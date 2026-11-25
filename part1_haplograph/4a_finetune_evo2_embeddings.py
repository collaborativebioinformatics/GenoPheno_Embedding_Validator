"""
Step 4a - fine-tune Evo 2 on the step-3 node embeddings (no DNA). One example = one training
haplotype on one chromosome (--units haplotype, default: exactly one token per haploblock), or one
training individual with every cluster it carries (--units individual). Each token is the node's
embedding with its genomic position (sin/cos, --pos-freqs) and its haploblock's length appended,
projected into the backbone in place of its nucleotide embeddings, plus a per-chromosome vector.
Clusters carried by fewer than --min-support training haplotypes share one token per haploblock.

  --objective next     predict the next haploblock's cluster from the ones before it
  --objective masked   predict masked spans of 1..--mask-max haploblocks from both sides (the
                       backbone runs forward and on the reversed sequence)
  --head               scaled-cosine (learned logit scale), table (learned output vector per token)
                       or cosine (fixed temperature)
  --softmax block      the softmax runs over the clusters of the target's haploblock only (exact for
                       haplotype units); all = over every token of the chromosome
  --backbone           evo2 (default, GPU), bionemo (--bionemo, NeMo2 checkpoint, experimental:
                       VERSION-SENSITIVE parts in evo2_model.py) or small (a transformer trained
                       from scratch); --init random keeps Evo 2's architecture without its pretraining
The last --train-blocks backbone blocks are fine-tuned (fp32 master weights; 0 = frozen backbone;
default 4 for Evo 2, every block for small). Only the graph's training individuals are used, split
again into train / validation; the parameters of the best validation loss are kept. --smoke loads
the backbone, checks one forward / backward and measures memory and time per step, then exits.

OUTPUT  --out, default <graph-dir>/<backbone>[-random]_<objective>_<method>[_individual].pt (torch dict,
  CPU tensors): mode "embeddings", backend, init, units, objective, head, softmax, graph_method, chroms,
  n_chrom_slots, graph_id, embeddings_sha1, trained_blocks (the trained backbone parameters; the whole
  small backbone), proj_in, chrom_emb, head_layers, extra, train_individuals, val_individuals, train_set_id,
  train_source, trainable_params, metrics {before, best, last, best_step, history}, config.
  <out>.resume holds the training state after every evaluation (for --resume); removed at the end.
  Step 6 rebuilds the model with evo2_model.load_finetuned().

USAGE
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G                      # Evo 2, GPU
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --objective masked
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --init random         # control
  python 4a_finetune_evo2_embeddings.py --method gnn --graph-dir $G --backbone small      # control
  python 4a_finetune_evo2_embeddings.py --smoke                                          # GPU check
  python 4a_finetune_evo2_embeddings.py --test                                           # CPU
"""
import argparse
import json
import os
import time

import numpy as np

from evo2_model import (HEADS, GenomeData, HaploModel, add_backbone_args, backbone_fingerprint, backbone_state,
                        config_of, fit, make_runner)
from haplo_common import (METHODS, check_args, die, graph_id, log, parse_chroms, read_graph_individuals, require_files,
                          selftest_graph_dir, train_columns)


def evaluate_fn(model, val_ex):
    def evaluate():
        tot = correct = count = 0
        for chrom, ex in val_ex:
            loss, (k, n) = model.loss(chrom, ex, mask=model.evaluation_mask(chrom, ex)
                                      if model.objective == "masked" else None)
            tot += loss.item() * n; correct += k; count += n
        return {"loss": tot / max(count, 1), "accuracy": correct / max(count, 1), "examples": len(val_ex)}
    return evaluate


def check_embeddings(args, train_id):
    info_file = os.path.join(args.graph_dir, f"node_emb_{args.method}.json")
    if not os.path.exists(info_file):
        return
    info = json.load(open(info_file))
    if info.get("train_set_id", "all") != train_id:
        die(f"node_emb_{args.method}.npy was made from another training set ({info.get('train_set_id')}) than "
            f"this run uses ({train_id}); rerun 3_node_embeddings.py with the same training set")
    if info.get("graph_id") != graph_id(args.graph_dir):
        die(f"node_emb_{args.method}.npy was made from an earlier build of this graph; rerun "
            f"3_node_embeddings.py --method {args.method} --graph-dir {args.graph_dir}")


def smoke(args):
    """Load the backbone, run one forward / backward on 16 tokens, check that gradients reach the input
    projection and the trainable blocks, and time one step at --max-len tokens."""
    import torch
    rnr = make_runner(args, args.train_blocks)
    proj = torch.nn.Linear(16, rnr.hidden_in).to(rnr.dev)
    report = {"backend": rnr.backend, "hidden_in": rnr.hidden_in, "hidden_out": rnr.hidden_out,
              "trainable_params": int(sum(p.numel() for _, p in rnr.trainable))}
    for T, key in ((16, "check"), (args.max_len, "step")):
        if rnr.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = rnr.run_soft(proj(torch.randn(T, 16, device=rnr.dev)))
        out.float().pow(2).mean().backward()
        if rnr.dev.type == "cuda":
            torch.cuda.synchronize()
            report[f"{key}_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        report[f"{key}_seconds"] = round(time.time() - t0, 3)
        if key == "check":
            if proj.weight.grad is None:
                die("smoke: no gradient reached the input projection through the backbone")
            if rnr.trainable and all(p.grad is None for _, p in rnr.trainable):
                die("smoke: no gradient reached the trainable backbone blocks")
    report["seconds_per_step_at_max_len"] = report.pop("step_seconds")
    log("smoke: " + json.dumps({"max_len": args.max_len, **report}))
    return report


def train(args):
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    individuals = read_graph_individuals(args.graph_dir)
    train_cols, train_src, train_id = train_columns(args, individuals)
    check_embeddings(args, train_id)
    perm = rng.permutation(train_cols)
    n_val = max(1, int(args.val_frac * len(perm)))
    if len(perm) - n_val < 1:
        die(f"only {len(perm)} training individuals: too few to hold out {n_val} for validation")
    val_inds, tr_inds = perm[:n_val], perm[n_val:]
    data = GenomeData(args.graph_dir, args.method, args.chroms, args.min_support, tr_inds, args.pos_freqs, args.units)
    train_ex, val_ex = data.examples(tr_inds), data.examples(val_inds)
    if not train_ex or not val_ex:
        die("no training / validation sequences with at least 2 tokens; check --chroms and the graph")
    if len(val_ex) > args.val_examples:
        val_ex = [val_ex[j] for j in rng.choice(len(val_ex), args.val_examples, replace=False)]
    log(f"[4a] embeddings={args.method}, units={args.units}, objective={args.objective}, head={args.head}, "
        f"softmax={args.softmax}; {len(data.chroms)} chromosome(s), {len(tr_inds):,} train / {len(val_inds):,} val "
        f"individuals (both from the training set: {train_src}), {len(train_ex):,} training sequences")

    rnr = make_runner(args, args.train_blocks)
    fingerprint = backbone_fingerprint(rnr.net)
    model = HaploModel(rnr, data, args.head, args.softmax, args.objective, args.temperature, args.max_len,
                       args.mask_frac, args.mask_max)
    blocks = [p for _, p in rnr.trainable]
    n_trainable = int(sum(p.numel() for p in blocks + model.parameters()))
    log(f"  trainable parameters: {n_trainable:,} (backbone {sum(p.numel() for p in blocks):,})")
    tag = rnr.backend + ("-random" if args.init == "random" else "")
    out = args.out or os.path.join(args.graph_dir, f"{tag}_{args.objective}_{args.method}"
                                                   f"{'_individual' if args.units == 'individual' else ''}.pt")
    describe = lambda r: f"val loss {r['loss']:.4f}, val top-1 accuracy {r['accuracy']:.3f}"
    metrics = fit(args, blocks, model.parameters(), len(train_ex), lambda j: model.loss(*train_ex[j], rng)[0],
                  evaluate_fn(model, val_ex), describe, model.train,
                  ("the input projection", list(model.proj_in.parameters())), resume_path=out + ".resume", rng=rng)

    torch.save({
        "mode": "embeddings", "backend": rnr.backend, "init": args.init, "units": args.units,
        "objective": args.objective, "head": args.head, "softmax": args.softmax, "graph_method": args.method,
        "chroms": data.chroms, "n_chrom_slots": data.n_chrom_slots, "embeddings_sha1": data.emb_sha1,
        "graph_id": graph_id(args.graph_dir), "backbone_fingerprint": fingerprint, **rnr.describe(),
        "trained_blocks": {n: p.detach().cpu() for n, p in backbone_state(rnr).items()},
        **{k: {n: t.detach().cpu() for n, t in v.items()} for k, v in model.state().items()},
        "val_individuals": [individuals[i] for i in val_inds],
        "train_individuals": [individuals[i] for i in tr_inds],
        "train_set_id": train_id, "train_source": train_src, "trainable_params": n_trainable,
        "metrics": metrics, "config": config_of(args),
    }, out)
    if os.path.exists(out + ".resume"):
        os.remove(out + ".resume")
    log(f"  saved to {out}")
    return out


def self_test(args):
    """Synthetic graph + embeddings (3_node_embeddings.py --test), then a few steps with a small backbone."""
    import importlib
    importlib.import_module("3_node_embeddings").main(["--test"])
    args.graph_dir = selftest_graph_dir()
    args.method = args.method or "gnn"
    args.backbone, args.bionemo, args.device, args.small_hidden, args.small_layers = "small", False, "cpu", 32, 2
    args.batch, args.epochs, args.max_len, args.log_every, args.eval_every = 4, 2, 64, 5, 10_000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, help="which node embeddings from 3_node_embeddings.py to use")
    ap.add_argument("--graph-dir", default="haplograph/genome_train_half_A", help="output folder of 2_build_graph.py")
    ap.add_argument("--chroms", help="train on these chromosomes of the graph only (default: all)")
    ap.add_argument("--train-individuals",
                    help="file or URL of training individuals (default: <graph-dir>/train_individuals.txt, else all)")
    ap.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--units", choices=["haplotype", "individual"], default="haplotype",
                    help="haplotype = one token per haploblock per haplotype (needs step 1 --haplotypes); individual = "
                         "every cluster an individual carries")
    ap.add_argument("--objective", choices=["next", "masked"], default="next")
    ap.add_argument("--head", choices=HEADS, default="scaled-cosine")
    ap.add_argument("--softmax", choices=["block", "all"],
                    help="block (default with haplotype units) = over the target haploblock's clusters; all = over "
                         "the chromosome's tokens (default with individual units)")
    ap.add_argument("--mask-frac", type=float, default=0.15, help="--objective masked: share of tokens masked")
    ap.add_argument("--mask-max", type=int, default=3, help="--objective masked: longest masked span (haploblocks)")
    ap.add_argument("--max-len", type=int, default=1024, help="max tokens per training sequence")
    ap.add_argument("--temperature", type=float, default=0.1, help="--head cosine: fixed softmax temperature")
    ap.add_argument("--pos-freqs", type=int, default=16,
                    help="genomic-position encoding: sin/cos at this many wavelengths appended to every "
                         "embedding (0 = none)")
    ap.add_argument("--min-support", type=int, default=2,
                    help="clusters carried by fewer training haplotypes (individuals with --units individual) become "
                         "one token per haploblock (0 = no merging)")
    ap.add_argument("--out", help="output .pt (default <graph-dir>/<backbone>[-random]_<objective>_<method>"
                                  "[_individual].pt)")
    ap.add_argument("--smoke", action="store_true",
                    help="load the backbone, check one forward/backward, report memory and time per step, exit")
    ap.add_argument("--test", action="store_true", help="self-test on synthetic data with a small backbone (CPU, seconds)")
    add_backbone_args(ap)
    e = ap.add_argument_group("training")
    e.add_argument("--train-blocks", type=int, help="backbone blocks to train (default 4 for Evo 2, all for small; "
                                                    "0 = frozen backbone, train the new layers only)")
    e.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
    e.add_argument("--epochs", type=int, default=3)
    e.add_argument("--lr", type=float, default=1e-5, help="for the trainable Evo 2 blocks")
    e.add_argument("--head-lr", type=float, default=1e-3, help="for the new layers (and the whole small backbone)")
    e.add_argument("--val-frac", type=float, default=0.1, help="fraction of training individuals held out")
    e.add_argument("--val-examples", type=int, default=300, help="max validation examples per evaluation")
    e.add_argument("--log-every", type=int, default=10)
    e.add_argument("--eval-every", type=int, default=200, help="steps between evaluations (also after every epoch)")
    e.add_argument("--resume", action="store_true", help="continue from <out>.resume")
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.test:
        self_test(args)
    if args.backbone == "small" and args.lr == 1e-5:
        args.lr = args.head_lr                       # a model trained from scratch needs the larger rate
    if args.train_blocks is None:
        args.train_blocks = None if args.backbone == "small" else 4
    if args.max_len < 2:
        die("--max-len must be >= 2")
    if args.smoke:
        smoke(args)
        return
    if not args.method:
        ap.error("--method is required")
    args.softmax = args.softmax or ("block" if args.units == "haplotype" else "all")
    args.chroms = parse_chroms(args.chroms)
    check_args(args, ("mask_max",), training=True)
    if args.pos_freqs < 0 or not 0 < args.mask_frac < 1 or args.temperature <= 0:
        die("--pos-freqs must be >= 0, --mask-frac between 0 and 1 and --temperature > 0")
    require_files(args.graph_dir, ["nodes.csv.gz", "individuals.txt", "carriers.npz"], "2_build_graph.py")
    require_files(args.graph_dir, [f"node_emb_{args.method}.npy"], f"3_node_embeddings.py --method {args.method}")
    train(args)
    log("done.")


if __name__ == "__main__":
    main()
