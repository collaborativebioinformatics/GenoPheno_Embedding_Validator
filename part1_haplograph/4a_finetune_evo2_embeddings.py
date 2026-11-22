"""
Step 4 - fine-tune Evo 2 on the step-3 node embeddings (no DNA). One example = one training
individual on one chromosome: their clusters in genomic order, each replaced by its embedding,
projected into Evo 2 in place of its nucleotide embeddings (plus a per-chromosome vector). Evo 2
runs up to its read-out layer, the last --train-blocks blocks are fine-tuned (fp32 master weights;
0 = frozen baseline), and a head predicts the NEXT cluster among the chromosome's tokens. Clusters
carried by fewer than --min-support (default 2) of these training individuals share one token per
haploblock region.
Only the graph's training individuals are used (split again into train / validation).

Backbone: Evo 2 via the `evo2` package (default; one CUDA GPU, install_evo2.sh), via NVIDIA
BioNeMo (--bionemo --ckpt-dir NeMo2 checkpoint; experimental, VERSION-SENSITIVE parts in
evo2_model.py), or --mock-evo2 (a small random stand-in for CPU checks; results mean nothing).

OUTPUT  --out, default <graph-dir>/evo2[_bionemo|_mock]_next_<method>.pt (torch dict, CPU tensors):
  mode "embeddings", objective "next", backend and backbone settings, graph_method, chroms,
  n_chrom_slots, graph_id, embeddings_sha1, trained_blocks, proj_in, chrom_emb, head,
  train_individuals, val_individuals, train_set_id, train_source, metrics {before, after}, config.
  Step 5 rebuilds it with evo2_model.load_finetuned().

USAGE
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --train-blocks 0       # baseline
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --bionemo --ckpt-dir nemo2_evo2_7b_8k
  python 4a_finetune_evo2_embeddings.py --test                                             # CPU
"""
import argparse
import json
import os

import numpy as np

from evo2_model import GenomeData, NextClusterModel, add_backbone_args, config_of, fit, make_runner
from haplo_common import (METHODS, check_args, die, graph_id, log, parse_chroms, read_graph_individuals, require_files,
                          selftest_graph_dir, train_columns)

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
    individuals = read_graph_individuals(args.graph_dir)
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
    data = GenomeData(args.graph_dir, args.method, args.chroms, args.min_support, tr_inds)
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
    ap.add_argument("--min-support", type=int, default=2,
                    help="clusters carried by fewer training individuals become one token per haploblock region "
                         "(0 = no merging; 2 = default, so the merged token also occurs in training)")
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
    check_args(args, training=True)
    if args.max_len < 2:
        die("--max-len must be >= 2")
    require_files(args.graph_dir, ["nodes.csv.gz", "individuals.txt", "carriers.npz"], "1_build_graph.py")
    require_files(args.graph_dir, [f"node_emb_{args.method}.npy"], f"3_node_embeddings.py --method {args.method}")
    train(args)
    log("done.")


if __name__ == "__main__":
    main()
