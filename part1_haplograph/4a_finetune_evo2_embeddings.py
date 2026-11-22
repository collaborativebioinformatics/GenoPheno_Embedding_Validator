"""
Step 4 - fine-tune Evo 2 (7B) on the node embeddings from 3_node_embeddings.py (node2vec, gnn or
xgboost). No DNA sequence is used. (For hash-phenotype associations, see 4b_finetune_evo2_hash.py.)

  - A training example is one individual on one chromosome: the clusters that person carries,
    ordered by genomic position, each replaced by its node embedding.
  - A linear layer projects the embeddings to Evo 2's hidden size; a learned per-chromosome vector
    is added (useful for node2vec, whose vectors aren't aligned across chromosomes). These vectors
    are fed into Evo 2 IN PLACE OF its nucleotide token embeddings.
  - Evo 2 runs up to its read-out layer; the last --train-blocks blocks before it are unfrozen, the
    rest stay frozen (--train-blocks 0 = frozen-Evo 2 baseline, only the new layers train).
  - Objective: predict the NEXT cluster along the chromosome (softmax over all clusters of that
    chromosome), mirroring Evo 2's next-token pre-training.
  - Only TRAINING individuals are used: --train-individuals FILE_OR_URL if given, else
    <graph-dir>/train_individuals.txt (written by 1_build_graph.py from the split of
    download_haplograph.py), else all individuals. They are split again into train / validation
    (--val-frac); validation loss and accuracy are reported before and after training. The test
    individuals are never seen; step 5 evaluates the model on them.

BACKBONE
  default      Evo 2 through the `evo2` package (Arc Institute), plain PyTorch on one CUDA GPU
               (install_evo2.sh). Options: --evo2-model, --layer, --embed-module.
  --bionemo    Evo 2 through NVIDIA BioNeMo / NeMo 2 (Megatron Core), from a NeMo2 checkpoint
               (--ckpt-dir, --model-size, --layer-index). Run inside the BioNeMo Framework container.
               EXPERIMENTAL: not yet run on a GPU; code that may need adjusting for your BioNeMo
               version is marked VERSION-SENSITIVE in evo2_model.py.
  --mock-evo2  a small random causal transformer instead of Evo 2, on CPU or GPU. For checking the
               pipeline end-to-end without a GPU (what --test uses); its results mean nothing.

INPUT  (in --graph-dir)
  nodes.csv.gz              CSV, N rows: node, id, chrom, region, start, end, support
                            (genomic order within a chromosome is taken from start)
  individuals.txt           I lines, one individual per line (columns of carriers.npz)
  carriers.npz              scipy.sparse CSR uint8 [N, I], node x individual carriers
  node_emb_<method>.npy     float32 [N, dim], row i = node i of nodes.csv.gz      (step 3)
  node_emb_<method>.json    training set the embeddings were made from (must match)
  train_individuals.txt     (optional) training individuals, one per line (step 1)

OUTPUT  --out, default <graph-dir>/evo2[_bionemo|_mock]_next_<method>.pt   (torch.save dict, CPU tensors)
  mode ("embeddings"), backend ("evo2" / "bionemo" / "mock"), objective ("next"), graph_method,
  chroms, n_chrom_slots, embeddings_sha1, evo2_model + layer (evo2), ckpt_dir + model_size +
  layer_index (bionemo), mock_config (mock)
  trained_blocks   {parameter name in the backbone: tensor} for the unfrozen blocks
                   (mock: the whole backbone, so it can be rebuilt exactly)
  proj_in          state_dict: Linear(dim -> hidden) + LayerNorm
  chrom_emb        state_dict: Embedding(n_chroms, hidden), rows in chromosome order of the graph
  head             state_dict: LayerNorm + Linear(hidden -> dim)
  train_individuals, val_individuals   who the model was trained / validated on (step 5 checks)
  train_set_id, train_source;  metrics {before, after};  config
  Rebuild with evo2_model.load_finetuned(torch.load(path), graph_dir) (what step 5 does).

USAGE
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir haplograph/genome_train_half_A
  python 4a_finetune_evo2_embeddings.py --method node2vec --graph-dir $G --chroms chr21,chr22
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --train-blocks 0     # frozen baseline
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --bionemo --ckpt-dir nemo2_evo2_7b_8k
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --mock-evo2          # CPU check
  python 4a_finetune_evo2_embeddings.py --test                                           # synthetic, CPU

REQUIRES  numpy, pandas, scipy, torch; evo2 (default: Linux, CUDA 12.8+ and driver >= 570 (install_evo2.sh), Python 3.11/3.12, an
          80 GB GPU recommended for 7B) or the BioNeMo container (--bionemo).

NOTE  The official Evo 2 training paths are Savanna and NVIDIA BioNeMo. The default backbone uses
      the `evo2` inference package in plain PyTorch and, after the first step, checks that gradients
      reach the new layers and the unfrozen blocks, stopping with a clear message if they don't.
"""
import argparse
import json
import os

import numpy as np

from evo2_model import GenomeData, NextClusterModel, add_backbone_args, config_of, fit, make_runner
from haplo_common import (die, graph_id, log, parse_chroms, read_graph_individuals, require_files,
                          selftest_graph_dir, train_columns)

METHODS = ("node2vec", "gnn", "xgboost")


def evaluate_fn(model, val_ex):
    def evaluate():
        tot = correct = count = 0
        for chrom, i in val_ex:
            loss, (k, n) = model.loss(chrom, i)
            tot += loss.item(); correct += k; count += n
        return {"loss": tot / max(len(val_ex), 1), "accuracy": correct / max(count, 1), "examples": len(val_ex)}
    return evaluate


def train(args):
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = GenomeData(args.graph_dir, args.method, args.chroms)
    individuals = read_graph_individuals(args.graph_dir)
    if data.by_ind.shape[0] != len(individuals):
        die("individuals.txt and carriers.npz disagree; rebuild the graph with step 1")
    train_cols, train_src, train_id = train_columns(args, individuals)
    emb_info = os.path.join(args.graph_dir, f"node_emb_{args.method}.json")
    if os.path.exists(emb_info):
        info = json.load(open(emb_info))
        emb_id = info.get("train_set_id", "all")
        if emb_id != train_id:
            die(f"node_emb_{args.method}.npy was made from another training set ({emb_id}) than this run uses "
                f"({train_id}, {train_src}); rerun 3_node_embeddings.py with the same --train-individuals")
        if info.get("graph_id", graph_id(args.graph_dir)) != graph_id(args.graph_dir):
            die(f"node_emb_{args.method}.npy was made from an earlier build of this graph; rerun "
                f"3_node_embeddings.py --method {args.method} --graph-dir {args.graph_dir}")

    perm = rng.permutation(train_cols)
    n_val = max(1, int(args.val_frac * len(perm)))
    if len(perm) - n_val < 1:
        die(f"only {len(perm)} training individuals: too few to hold out {n_val} for validation")
    val_inds, tr_inds = perm[:n_val], perm[n_val:]
    train_ex, val_ex = data.examples(tr_inds), data.examples(val_inds)
    if not train_ex or not val_ex:
        die("no training / validation sequences with at least 2 clusters; check --chroms and the graph")
    if len(val_ex) > args.val_examples:
        val_ex = [val_ex[j] for j in rng.choice(len(val_ex), args.val_examples, replace=False)]
    log(f"[evo2] embeddings={args.method}, {len(data.chroms)} chromosome(s), {len(tr_inds):,} train / "
        f"{len(val_inds):,} val individuals (both from the training set: {train_src}), "
        f"{len(train_ex):,} training sequences")

    rnr = make_runner(args, args.train_blocks)
    model = NextClusterModel(rnr, data, args.temperature, args.max_len)
    describe = lambda r: f"val loss {r['loss']:.4f}, val next-cluster top-1 accuracy {r['accuracy']:.3f}"
    before, after = fit(args, [p for _, p in rnr.trainable], model.modules, len(train_ex),
                        lambda j: model.loss(*train_ex[j], rng)[0], evaluate_fn(model, val_ex), describe,
                        model.train, rng, ("the input projection", list(model.proj_in.parameters())))

    backend = rnr.backend
    trained = (rnr.net.state_dict() if backend == "mock"
               else {n: p for n, p in rnr.trainable})
    tag = "" if backend == "evo2" else f"_{backend}"
    out = args.out or os.path.join(args.graph_dir, f"evo2{tag}_next_{args.method}.pt")
    torch.save({
        "mode": "embeddings", "backend": backend, "objective": "next", "graph_method": args.method,
        "chroms": data.chroms, "n_chrom_slots": data.n_chrom_slots, "embeddings_sha1": data.emb_sha1,
        "graph_id": graph_id(args.graph_dir),
        "evo2_model": args.evo2_model if backend == "evo2" else None,
        "layer": args.layer if backend == "evo2" else None,
        **rnr.describe(),
        "trained_blocks": {n: p.detach().cpu() for n, p in trained.items()},
        **{k: {n: t.detach().cpu() for n, t in v.items()} for k, v in model.state().items()},
        "val_individuals": [individuals[i] for i in val_inds],
        "train_individuals": [individuals[i] for i in tr_inds],
        "train_set_id": train_id, "train_source": train_src,
        "metrics": {"before": before, "after": after}, "config": config_of(args),
    }, out)
    log(f"  saved to {out}")
    return out


def self_test(args):
    """Synthetic graph + embeddings (3_node_embeddings.py --test), then a few steps with --mock-evo2."""
    import importlib
    importlib.import_module("3_node_embeddings").main(["--test"])
    args.graph_dir = selftest_graph_dir()
    args.method = args.method or "gnn"
    args.mock_evo2, args.bionemo, args.device = True, False, "cpu"
    args.mock_hidden, args.mock_layers, args.train_blocks = 32, 2, 1
    args.batch, args.epochs, args.max_len, args.log_every, args.eval_every = 4, 2, 64, 5, 10_000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, help="which node embeddings from 3_node_embeddings.py to use")
    ap.add_argument("--graph-dir", default="haplograph/genome_train_half_A", help="output folder of 1_build_graph.py")
    ap.add_argument("--chroms", help="train on these chromosomes of the graph only (default: all)")
    ap.add_argument("--train-individuals",
                    help="file or URL of training individuals (default: <graph-dir>/train_individuals.txt, else all)")
    ap.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--max-len", type=int, default=1024, help="max clusters per training sequence")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--out", help="output .pt (default <graph-dir>/evo2[_backend]_next_<method>.pt)")
    ap.add_argument("--test", action="store_true", help="self-test on synthetic data with --mock-evo2 (CPU, seconds)")
    add_backbone_args(ap)
    e = ap.add_argument_group("training")
    e.add_argument("--train-blocks", type=int, default=4, help="0 = Evo 2 frozen, train new layers only")
    e.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
    e.add_argument("--epochs", type=int, default=1)
    e.add_argument("--lr", type=float, default=1e-5, help="for unfrozen Evo 2 blocks")
    e.add_argument("--head-lr", type=float, default=1e-3, help="for the new layers")
    e.add_argument("--val-frac", type=float, default=0.1, help="fraction of training individuals held out")
    e.add_argument("--val-examples", type=int, default=300, help="max validation examples per evaluation")
    e.add_argument("--log-every", type=int, default=10)
    e.add_argument("--eval-every", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.test:
        self_test(args)
    if not args.method:
        ap.error("--method is required (node2vec, gnn or xgboost)")
    args.chroms = parse_chroms(args.chroms)
    for name in ("max_len", "batch", "epochs", "val_examples", "log_every", "eval_every"):
        if getattr(args, name) < 1:
            die(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_len < 2:
        die("--max-len must be >= 2")
    if args.train_blocks < 0:
        die("--train-blocks must be >= 0")
    if not 0 < args.val_frac < 1:
        die("--val-frac must be between 0 and 1")
    require_files(args.graph_dir, ["nodes.csv.gz", "individuals.txt", "carriers.npz"], "1_build_graph.py")
    require_files(args.graph_dir, [f"node_emb_{args.method}.npy"], f"3_node_embeddings.py --method {args.method}")
    train(args)
    log("done.")


if __name__ == "__main__":
    main()
