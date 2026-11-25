"""
Step 4b - fine-tune Evo 2 to predict a phenotype from an individual's genome.

  --input tokens (default)  both haplotypes of every chromosome as step-4a tokens (the step-3 node
                            embedding of each haploblock's cluster, --method), each run through the
                            backbone; attention pooling over the two haplotypes of a chromosome, then
                            over the chromosomes, then a phenotype head. --init-from starts from a
                            step-4a checkpoint (its backbone blocks and input projection)
  --input hash              the 'unlinked hash' baseline: a 0/1 genomic hash per row (--hash-pheno),
                            cut into tokens of --hash-bits bits (one haploblock hash each), so the
                            model sees cluster identities without the haplograph's linkage
Numbers are treated as regression (R^2, Pearson r), labels as classification (accuracy vs majority);
--task forces one. The training rows are split again into train / validation.

PHENOTYPES (tokens): --phenotype FILE (individual <tab> value, optional header) of TRAINING
individuals, or --simulate on the graph's real haplotypes (1000 Genomes has no measured traits):
  additive  --causal-blocks haploblocks, one causal cluster each (training frequency 5-50%), effect
            N(0, 1) per copy (dosage 0-2)
  cis       --causal-blocks pairs of adjacent haploblocks with a cluster pair (a, b) found together on
            5-50% of training haplotypes; the effect counts the haplotypes carrying a AND b (cis); a
            and b on different haplotypes of the same individual do nothing, so an additive model of
            cluster dosages cannot represent it exactly
  null      noise only
  phenotype = genetic value scaled to heritability --h2 + Gaussian noise. Writes
  <graph-dir>/phenotypes/<trait>_train.tsv (training individuals only), <trait>_test.tsv and
  <trait>_answer.json (causal haploblocks, clusters, effects) for step 6's phenotype_prediction.

OUTPUT  --out, default <graph-dir>/<backbone>[-random]_pheno_<trait>_<method>.pt (hash input:
  <TSV folder>/<backbone>_hash_<trait>.pt): mode "phenotype", input, backend, init, task, trait, classes,
  target_mean/std, n_out, graph_method, chroms, embeddings_sha1, graph_id, trained_blocks, proj_in,
  chrom_emb, head_layers, query, train_individuals, val_individuals (val_rows for hash input),
  trainable_params, metrics {before, best, last, best_step, history}, config.

USAGE
  python 4b_finetune_evo2_phenotype.py --graph-dir $G --method gnn --simulate cis
  python 4b_finetune_evo2_phenotype.py --graph-dir $G --method gnn --phenotype height_train.tsv \\
      --init-from $G/evo2_next_gnn.pt
  python 4b_finetune_evo2_phenotype.py --input hash --hash-pheno hash_height.tsv
  python 4b_finetune_evo2_phenotype.py --test        # synthetic data, small backbone, CPU
"""
import argparse
import importlib
import json
import os
import re

import numpy as np
import pandas as pd

from evo2_model import GenomeData, PhenotypeModel, add_backbone_args, backbone_state, config_of, fit, make_runner
from haplo_common import (METHODS, check_args, die, graph_id, log, parse_chroms, read_graph_individuals,
                          read_individual_list, require_files, selftest_graph_dir, test_workdir)


# ---------------------------------------------------------------------------
# Phenotypes
# ---------------------------------------------------------------------------

def simulate(graph_dir, kind, h2, n_causal, seed):
    """Simulated trait on the graph's haplotypes; writes the phenotype files and the answer key and
    returns the training file."""
    rng = np.random.default_rng(seed)
    h = np.load(os.path.join(graph_dir, "haplotypes.npz"))
    H, blocks, bchrom = h["node"], h["blocks"], h["block_chrom"]
    inds = read_graph_individuals(graph_dir)
    pos = {x: i for i, x in enumerate(inds)}
    train = [pos[x] for x in read_individual_list(os.path.join(graph_dir, "train_individuals.txt")) if x in pos]
    tr_rows = np.ravel([[2 * i, 2 * i + 1] for i in train])
    freq = np.bincount(H[tr_rows].ravel(), minlength=H.max() + 1) / len(tr_rows)
    g, causal = np.zeros(len(inds)), []
    if kind == "additive":
        ok = [b for b in range(H.shape[1]) if ((freq[np.unique(H[:, b])] >= 0.05) & (freq[np.unique(H[:, b])] <= 0.5)).any()]
        for b in rng.choice(ok, min(n_causal, len(ok)), replace=False):
            c = rng.choice([n for n in np.unique(H[:, b]) if 0.05 <= freq[n] <= 0.5])
            beta = rng.normal()
            g += beta * ((H[0::2, b] == c).astype(float) + (H[1::2, b] == c))
            causal.append({"haploblocks": [str(blocks[b])], "clusters": [int(c)], "effect": float(beta)})
    elif kind == "cis":
        pairs = []
        for b in range(H.shape[1] - 1):
            if bchrom[b] != bchrom[b + 1]:
                continue
            key, cnt = np.unique(H[tr_rows, b].astype(np.int64) * (H.max() + 1) + H[tr_rows, b + 1], return_counts=True)
            good = key[(cnt / len(tr_rows) >= 0.05) & (cnt / len(tr_rows) <= 0.5)]
            if len(good):
                pairs.append((b, rng.choice(good)))
        for j in rng.choice(len(pairs), min(n_causal, len(pairs)), replace=False):
            b, key = pairs[j]
            a, c = divmod(int(key), H.max() + 1)
            beta = rng.normal()
            cis = (H[:, b] == a) & (H[:, b + 1] == c)                    # per haplotype
            g += beta * (cis[0::2].astype(float) + cis[1::2])
            causal.append({"haploblocks": [str(blocks[b]), str(blocks[b + 1])], "clusters": [int(a), int(c)],
                           "effect": float(beta)})
    elif kind != "null":
        die(f"unknown --simulate {kind!r}")
    noise = rng.normal(size=len(inds))
    y = (np.sqrt(h2) * (g - g.mean()) / g.std() + np.sqrt(1 - h2) * noise) if g.std() > 0 else noise
    out = os.path.join(graph_dir, "phenotypes")
    os.makedirs(out, exist_ok=True)
    test = [pos[x] for x in read_individual_list(os.path.join(graph_dir, "test_individuals.txt")) if x in pos]
    for name, idx in (("train", train), ("test", test)):
        pd.DataFrame({"individual": [inds[i] for i in idx], kind: np.round(y[idx], 6)}).to_csv(
            os.path.join(out, f"{kind}_{name}.tsv"), sep="\t", index=False)
    with open(os.path.join(out, f"{kind}_answer.json"), "w") as f:
        json.dump({"trait": kind, "h2": float(h2) if causal else 0.0, "causal": causal, "seed": int(seed)}, f, indent=1)
    log(f"  simulated '{kind}' (h2 {h2 if causal else 0}, {len(causal)} causal effects) -> {out}/{kind}_*.tsv")
    return os.path.join(out, f"{kind}_train.tsv")


def read_table(path, task, first_pattern):
    """2-column TSV (key, phenotype), optional header -> keys, y, task, classes, name, n_skipped."""
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3)
    if raw.shape[1] < 2 or len(raw) == 0:
        die(f"{path}: expected a tab-separated file with 2 columns")
    raw = raw.iloc[:, :2].apply(lambda c: c.str.strip())
    name = "phenotype"
    if not re.fullmatch(first_pattern, raw.iloc[0, 0]) or raw.iloc[0, 0].lower() == "individual":
        name, raw = raw.iloc[0, 1] or name, raw.iloc[1:]
    k, v = raw.iloc[:, 0], raw.iloc[:, 1]
    ok = k.str.fullmatch(first_pattern) & (v != "") & ~v.str.lower().isin(["na", "nan", "none", "null"])
    keys, v = k[ok].tolist(), v[ok]
    if len(keys) < 2:
        die(f"{path}: fewer than 2 valid rows")
    num = pd.to_numeric(v, errors="coerce")
    if task == "auto":
        task = "regression" if num.notna().all() and num.nunique() > 10 else "classification"
    if task == "regression":
        if num.isna().any():
            die(f"non-numeric phenotype values {v[num.isna()].unique()[:5].tolist()}; use --task classification")
        return keys, num.to_numpy(np.float64), task, None, name, int((~ok).sum())
    classes = np.array(sorted(v.unique()))
    return keys, np.searchsorted(classes, v.to_numpy()).astype(np.int64), task, classes, name, int((~ok).sum())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokens = args.input == "tokens"
    if tokens:
        individuals = read_graph_individuals(args.graph_dir)
        pos = {x: i for i, x in enumerate(individuals)}
        graph_train = set(read_individual_list(os.path.join(args.graph_dir, "train_individuals.txt")))
        keys, y, task, classes, name, n_bad = read_table(args.phenotype, args.task, r"\S+")
        if args.simulate:
            name = args.simulate
        leaked = [k for k in keys if k in pos and k not in graph_train]
        if leaked:
            die(f"{len(leaked)} individuals of {args.phenotype} are not in the graph's training set (e.g. {leaked[:3]}): "
                "4b must not train on test individuals")
        keep = [j for j, k in enumerate(keys) if k in pos]
        if len(keep) < len(keys):
            log(f"  warning: {len(keys) - len(keep)} individuals of {args.phenotype} are not in the graph, ignored")
        examples, y = [pos[keys[j]] for j in keep], y[keep]
    else:
        examples, y, task, classes, name, n_bad = read_table(args.hash_pheno, args.task, r"[01]+")
    n = len(examples)
    perm = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val, tr = perm[:n_val], perm[n_val:]
    if len(tr) == 0:
        die(f"only {n} rows: too few to hold out {n_val} for validation; lower --val-frac")
    if task == "regression":
        mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
    data = GenomeData(args.graph_dir, args.method, args.chroms, args.min_support, [examples[j] for j in tr],
                      args.pos_freqs, "haplotype") if tokens else None
    rnr = make_runner(args, args.train_blocks)
    n_out = 1 if task == "regression" else len(classes)
    model = PhenotypeModel(rnr, n_out, data, args.hash_bits, args.max_len)
    if args.init_from:
        import torch as T
        meta = T.load(args.init_from, map_location="cpu", weights_only=True)
        if meta.get("mode") != "embeddings" or meta.get("graph_method") != args.method or meta.get("backend") != rnr.backend:
            die(f"--init-from {args.init_from} is not a step-4a model on the same embeddings and backbone")
        with T.no_grad():
            rnr.net.load_state_dict({k: v.to(rnr.dev) for k, v in meta["trained_blocks"].items()}, strict=False)
        model.proj_in.load_state_dict(meta["proj_in"])
        model.chrom_emb.load_state_dict(meta["chrom_emb"])
        log(f"  started from the step-4a model {args.init_from}")
    log(f"[4b] '{name}' as {task} from {'haplotype tokens (' + args.method + ')' if tokens else 'genomic hashes'}; "
        f"{n:,} rows{f' ({n_bad:,} invalid skipped)' if n_bad else ''}, {len(tr):,} train / {len(val):,} val")

    def example_loss(j):
        out = model.predict(examples[j])
        if task == "regression":
            return F.mse_loss(out.squeeze(-1), torch.tensor((y[j] - mu) / sd, dtype=out.dtype, device=rnr.dev))
        return F.cross_entropy(out.unsqueeze(0), torch.tensor([y[j]], device=rnr.dev))

    def evaluate():
        pred = np.array([model.predict(examples[j]).cpu().numpy() for j in val])
        yt = y[val]
        if task == "regression":
            p = pred[:, 0] * sd + mu
            r2 = float(1 - np.sum((p - yt) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-12))
            r = float(np.corrcoef(p, yt)[0, 1]) if p.std() > 0 and yt.std() > 0 else float("nan")
            return {"loss": float(np.mean(((p - yt) / sd) ** 2)), "r2": r2, "pearson_r": r}
        m = pred.max(1, keepdims=True)
        lp = pred - m - np.log(np.exp(pred - m).sum(1, keepdims=True))
        return {"loss": float(-lp[np.arange(len(yt)), yt].mean()), "accuracy": float((pred.argmax(1) == yt).mean()),
                "baseline": float((yt == np.bincount(y[tr]).argmax()).mean())}
    describe = (lambda r: f"val loss {r['loss']:.4f}, R^2 {r['r2']:.3f}, Pearson r {r['pearson_r']:.3f}") \
        if task == "regression" else (lambda r: f"val loss {r['loss']:.4f}, accuracy {r['accuracy']:.3f} "
                                                f"(majority {r['baseline']:.3f})")
    blocks = [p for _, p in rnr.trainable]
    n_trainable = int(sum(p.numel() for p in blocks + model.parameters()))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "phenotype"
    tag = rnr.backend + ("-random" if args.init == "random" else "")
    out = args.out or (os.path.join(args.graph_dir, f"{tag}_pheno_{safe}_{args.method}.pt") if tokens else
                       os.path.join(os.path.dirname(os.path.abspath(args.hash_pheno)), f"{tag}_hash_{safe}.pt"))
    metrics = fit(args, blocks, model.parameters(), len(tr), lambda j: example_loss(tr[j]), evaluate, describe,
                  model.train, ("the phenotype head", list(model.head.parameters())), resume_path=out + ".resume")
    ind = (lambda idx: [individuals[examples[j]] for j in idx]) if tokens else None
    torch.save({
        "mode": "phenotype", "input": args.input, "backend": rnr.backend, "init": args.init, "task": task,
        "trait": name, "classes": None if classes is None else classes.tolist(), "n_out": n_out,
        "target_mean": mu if task == "regression" else None, "target_std": sd if task == "regression" else None,
        "graph_method": args.method if tokens else None, "chroms": data.chroms if tokens else None,
        "embeddings_sha1": data.emb_sha1 if tokens else None, "graph_id": graph_id(args.graph_dir) if tokens else None,
        "simulated": args.simulate, **rnr.describe(),
        "trained_blocks": {k: v.detach().cpu() for k, v in backbone_state(rnr).items()},
        **{k: ({n: t.detach().cpu() for n, t in v.items()} if isinstance(v, dict) else v) for k, v in model.state().items()},
        "train_individuals": ind(tr) if tokens else None, "val_individuals": ind(val) if tokens else None,
        "val_rows": None if tokens else val.tolist(), "trainable_params": n_trainable,
        "metrics": metrics, "config": config_of(args),
    }, out)
    if os.path.exists(out + ".resume"):
        os.remove(out + ".resume")
    log(f"  saved to {out}")
    return out


def write_test_hashes(path, n=60, bits=48, seed=0):
    """Synthetic hash-phenotype TSV: the phenotype is the number of 1s in the first 8 bits + noise."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        f.write("hash\theight\n")
        for _ in range(n):
            h = rng.integers(0, 2, bits)
            f.write("".join(map(str, h)) + f"\t{160 + 4 * h[:8].sum() + rng.normal(0, 2):.1f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["tokens", "hash"], default="tokens")
    ap.add_argument("--graph-dir", default="haplograph/genome_train_half_A", help="tokens: output folder of step 2")
    ap.add_argument("--method", choices=METHODS, help="tokens: which step-3 node embeddings")
    ap.add_argument("--chroms", help="tokens: use these chromosomes of the graph only (default: all)")
    ap.add_argument("--phenotype", help="tokens: TSV individual <tab> value of training individuals")
    ap.add_argument("--simulate", choices=["additive", "cis", "null"], help="tokens: simulate the phenotype instead")
    ap.add_argument("--h2", type=float, default=0.5, help="--simulate: heritability")
    ap.add_argument("--causal-blocks", type=int, default=10, help="--simulate: causal haploblocks (pairs for cis)")
    ap.add_argument("--sim-seed", type=int, default=0, help="--simulate: seed")
    ap.add_argument("--init-from", help="tokens: start from this step-4a checkpoint (same embeddings and backbone)")
    ap.add_argument("--hash-pheno", help="hash: TSV hash (0/1 string) <tab> phenotype")
    ap.add_argument("--hash-bits", type=int, default=20, help="hash: bits per token (one haploblock hash)")
    ap.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    ap.add_argument("--min-support", type=int, default=2, help="tokens: as in step 4a")
    ap.add_argument("--pos-freqs", type=int, default=16, help="tokens: as in step 4a")
    ap.add_argument("--max-len", type=int, default=1024, help="max tokens per haplotype and chromosome")
    ap.add_argument("--out", help="output .pt")
    ap.add_argument("--test", action="store_true", help="self-test: simulated cis trait, small backbone, CPU")
    add_backbone_args(ap)
    e = ap.add_argument_group("training")
    e.add_argument("--train-blocks", type=int, help="backbone blocks to train (default 4 for Evo 2, all for small)")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--epochs", type=int, default=3)
    e.add_argument("--lr", type=float, default=1e-5, help="for the trainable Evo 2 blocks")
    e.add_argument("--head-lr", type=float, default=1e-3, help="for the new layers (and the whole small backbone)")
    e.add_argument("--val-frac", type=float, default=0.1)
    e.add_argument("--log-every", type=int, default=10)
    e.add_argument("--eval-every", type=int, default=200)
    e.add_argument("--resume", action="store_true", help="continue from <out>.resume")
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.test:
        importlib.import_module("3_node_embeddings").main(["--test"])
        args.graph_dir, args.method, args.simulate = selftest_graph_dir(), "gnn", "cis"
        args.backbone, args.device, args.small_hidden, args.small_layers = "small", "cpu", 32, 2
        args.batch, args.epochs, args.log_every, args.causal_blocks, args.h2 = 4, 2, 1000, 3, 0.8
    if args.backbone == "small" and args.lr == 1e-5:
        args.lr = args.head_lr
    if args.train_blocks is None:
        args.train_blocks = None if args.backbone == "small" else 4
    check_args(args, ("hash_bits", "max_len", "causal_blocks", "batch", "epochs", "log_every", "eval_every"))
    if args.input == "hash":
        if not args.hash_pheno or not os.path.exists(args.hash_pheno):
            die("--input hash needs --hash-pheno (an existing TSV)")
        args.chroms = None
    else:
        if not args.method:
            ap.error("--method is required with --input tokens")
        if bool(args.phenotype) == bool(args.simulate):
            ap.error("give either --phenotype FILE or --simulate additive|cis|null")
        args.chroms = parse_chroms(args.chroms)
        require_files(args.graph_dir, ["haplotypes.npz", "train_individuals.txt"], "1_download_haplograph.py --haplotypes, "
                      "then 2_build_graph.py")
        require_files(args.graph_dir, [f"node_emb_{args.method}.npy"], f"3_node_embeddings.py --method {args.method}")
        if args.simulate:
            if not 0 <= args.h2 < 1:
                die("--h2 must be in [0, 1)")
            args.phenotype = simulate(args.graph_dir, args.simulate, args.h2, args.causal_blocks, args.sim_seed)
    train(args)
    log("done.")


if __name__ == "__main__":
    main()
