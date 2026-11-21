"""
Step 4b - fine-tune Evo 2 7B on hash-phenotype associations (genomic hash -> phenotype).
Standalone: does not need steps 1-3. (For node embeddings, use 4a_finetune_evo2_embeddings.py.)

  - Each genomic hash (a string of 0s and 1s) is fed to Evo 2 as text: Evo 2's tokenizer is
    character-level, so every bit is one token. Hashes longer than --hash-window bits are split
    into windows; Evo 2's layer output is mean-pooled over all tokens of all windows.
  - Evo 2 runs up to --layer (default blocks.28.mlp.l3); the last --train-blocks blocks before it
    are unfrozen, the rest stay frozen.
  - A head on the pooled vector predicts the phenotype:
      regression     (numeric phenotype, e.g. height): MSE on the standardized value; reports RMSE
                     (original units), Pearson r and R^2 on held-out rows
      classification (text labels, or numbers with <= 10 distinct values): cross-entropy; reports
                     accuracy vs the majority-class baseline
    --task auto (default) picks one by these rules; --task regression|classification forces it.
  - Rows are split train/validation; validation metrics are reported before and after training.

INPUT
  --hash-pheno FILE.tsv   tab-separated, 2 columns, one row per sample; optional header row
                          (detected when its first field is not a 0/1 string; the second header
                          field is used as the phenotype name):
                            column 1  genomic hash, only the characters 0 and 1, any length
                            column 2  phenotype value, numeric (e.g. 172.5) or a label (e.g. tall)
                          Rows with an invalid hash or an empty / NA phenotype are skipped.
                          Example:
                            hash<TAB>height
                            0110100111010<TAB>172.5
                            1100101000111<TAB>165.0

OUTPUT  --out, default <TSV folder>/evo2_hash_<phenotype name>.pt   (torch.save dict)
  mode ("hash"), evo2_model, layer, task, phenotype, hash_window
  classes          class labels (classification) or None
  target_mean, target_std   (regression) prediction = head output * target_std + target_mean
  trained_blocks   {parameter name in evo.model: tensor} for the unfrozen Evo 2 blocks
  head             state_dict: LayerNorm + Linear(hidden -> 1 or n_classes)
  val_rows         0-based data-row indices (header excluded) of the held-out rows
  metrics {before, after};  config
  To reuse: Evo2(evo2_model), copy trained_blocks into evo.model (load_state_dict(strict=False)),
  rebuild the new layers with the same shapes and load their state_dicts.

USAGE
  python 4b_finetune_evo2_hash.py --test                                           # synthetic, CPU
  python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv                     # regression
  python 4b_finetune_evo2_hash.py --hash-pheno hash_pop.tsv --task classification
  python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv --train-blocks 0    # frozen baseline

  --mock-evo2 replaces Evo 2 by a small random stand-in model (CPU check of the script only).

REQUIRES  numpy, pandas, torch, evo2 (pip install evo2; Linux, CUDA 12.1+, Python 3.11/3.12;
          the 7B models run in bf16 without Transformer Engine; an 80 GB GPU is recommended).

NOTE  The official Evo 2 training paths are Savanna and NVIDIA BioNeMo. This script uses the `evo2`
      inference package in plain PyTorch and, after the first step, checks that gradients reach the
      new layers and the unfrozen blocks, stopping with a clear message if they don't.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from evo2_model import Evo2Runner, MockEvo2Runner, config_of, fit
from haplo_common import die, log, test_workdir


def load_hash_pheno(path, task="auto"):
    """Returns hashes (list of str), y (float64 or int64), task, classes, phenotype name, n_skipped."""
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3)
    if raw.shape[1] < 2 or len(raw) == 0:
        die(f"{path}: expected a tab-separated file with 2 columns (hash, phenotype)")
    raw = raw.iloc[:, :2].apply(lambda c: c.str.strip())
    name = "phenotype"
    if not re.fullmatch(r"[01]+", raw.iloc[0, 0]):            # header row
        name = raw.iloc[0, 1] or name
        raw = raw.iloc[1:]
    h, v = raw.iloc[:, 0], raw.iloc[:, 1]
    ok = h.str.fullmatch(r"[01]+") & (v != "") & ~v.str.lower().isin(["na", "nan", "none", "null"])
    n_skipped = int((~ok).sum())
    hashes, v = h[ok].tolist(), v[ok]
    if len(hashes) < 2:
        die(f"{path}: fewer than 2 valid rows")

    num = pd.to_numeric(v, errors="coerce")
    if task == "auto":
        task = "regression" if num.notna().all() and num.nunique() > 10 else "classification"
    if task == "regression":
        if num.isna().any():
            bad = v[num.isna()].unique()[:5].tolist()
            die(f"non-numeric phenotype values {bad}; use --task classification")
        return hashes, num.to_numpy(np.float64), task, None, name, n_skipped
    classes = np.array(sorted(v.unique()))
    return hashes, np.searchsorted(classes, v.to_numpy()).astype(np.int64), task, classes, name, n_skipped


def train_hash(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    hashes, y, task, classes, name, n_skipped = load_hash_pheno(args.hash_pheno, args.task)
    n = len(hashes)
    perm = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val, tr = perm[:n_val], perm[n_val:]
    if len(tr) == 0:
        die(f"only {n} valid rows: too few to hold out {n_val} for validation and still train; lower --val-frac")
    val_eval = val if len(val) <= args.val_examples else np.sort(rng.choice(val, args.val_examples, replace=False))
    lengths = np.array([len(h) for h in hashes])
    windows = np.ceil(lengths / args.hash_window).astype(int)
    target = (f"{len(classes)} classes {classes.tolist()[:10]}" if task == "classification"
              else f"mean {y.mean():.3g}, sd {y.std():.3g}")
    log(f"[evo2] hash-phenotype mode: '{name}' as {task} ({target}); {n:,} rows"
        f"{f' ({n_skipped:,} invalid rows skipped)' if n_skipped else ''}, {len(tr):,} train / {len(val):,} val")
    log(f"  hash length: min {lengths.min():,}, median {int(np.median(lengths)):,}, max {lengths.max():,} bits"
        f" -> up to {windows.max()} window(s) of {args.hash_window:,} tokens per hash")

    if task == "regression":
        mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
    if args.mock_evo2:
        rnr = MockEvo2Runner(args.mock_hidden, args.mock_layers, args.train_blocks, args.device, args.seed)
        tokenize = lambda h: list(h.encode())
    else:
        rnr = Evo2Runner(args.evo2_model, args.layer, args.train_blocks, args.embed_module)
        tokenize = rnr.evo.tokenizer.tokenize
    dev = rnr.dev
    head = nn.Sequential(nn.LayerNorm(rnr.hidden_out),
                         nn.Linear(rnr.hidden_out, 1 if task == "regression" else len(classes))).to(dev)

    def encode(h, train_mode):
        """Mean of Evo 2's --layer output over every token (bit) of the hash."""
        ids = torch.tensor(tokenize(h), dtype=torch.int, device=dev)
        starts = np.arange(0, len(ids), args.hash_window)
        if args.hash_max_windows and len(starts) > args.hash_max_windows:
            starts = (np.sort(rng.choice(starts, args.hash_max_windows, replace=False)) if train_mode
                      else starts[np.linspace(0, len(starts) - 1, args.hash_max_windows).astype(int)])
        total, count = 0, 0
        for s in starts:
            hs = rnr.run_ids(ids[s:s + args.hash_window])
            total = total + hs.sum(0)
            count += hs.shape[0]
        return total / count

    def example_loss(i, train_mode):
        """-> (loss, prediction in original units / class index)."""
        out = head(encode(hashes[i], train_mode))
        if task == "regression":
            pred = out.squeeze(-1)
            t = torch.tensor((y[i] - mu) / sd, dtype=pred.dtype, device=dev)
            return F.mse_loss(pred, t), pred.item() * sd + mu
        logits = out.unsqueeze(0)
        return F.cross_entropy(logits, torch.tensor([y[i]], device=dev)), int(logits.argmax(-1).item())

    def evaluate():
        losses, preds = [], []
        for i in val_eval:
            loss, p = example_loss(i, False)
            losses.append(loss.item()); preds.append(p)
        preds, yt = np.array(preds), y[val_eval]
        res = {"loss": float(np.mean(losses))}
        if task == "regression":
            res["rmse"] = float(np.sqrt(np.mean((preds - yt) ** 2)))
            res["pearson_r"] = float(np.corrcoef(preds, yt)[0, 1]) if preds.std() > 0 and yt.std() > 0 else float("nan")
            res["r2"] = float(1 - np.sum((preds - yt) ** 2) / max(np.sum((yt - mu) ** 2), 1e-12))
        else:
            res["accuracy"] = float((preds == yt).mean())
            res["baseline"] = float((yt == np.bincount(y[tr]).argmax()).mean())
        return res

    if task == "regression":
        describe = lambda r: (f"val loss {r['loss']:.4f}, RMSE {r['rmse']:.4g}, "
                              f"Pearson r {r['pearson_r']:.3f}, R^2 {r['r2']:.3f}")
    else:
        describe = lambda r: (f"val loss {r['loss']:.4f}, accuracy {r['accuracy']:.3f} "
                              f"(majority-class baseline {r['baseline']:.3f})")
    def set_train(mode):
        rnr.net.train(mode)
        head.train(mode)

    before, after = fit(args, [p for _, p in rnr.trainable], [head], len(tr),
                        lambda j: example_loss(tr[j], True)[0], evaluate, describe, set_train,
                        np.random.default_rng(args.seed + 1), ("the phenotype head", list(head.parameters())))

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "phenotype"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.hash_pheno)), f"evo2_hash_{safe}.pt")
    trained = rnr.net.state_dict() if args.mock_evo2 else dict(rnr.trainable)
    torch.save({
        "mode": "hash", "backend": rnr.backend, "evo2_model": args.evo2_model, "layer": args.layer, "task": task,
        "phenotype": name, "classes": None if classes is None else classes.tolist(),
        "target_mean": mu if task == "regression" else None,
        "target_std": sd if task == "regression" else None,
        "hash_window": args.hash_window,
        "trained_blocks": {n: p.detach().cpu() for n, p in trained.items()},
        "head": head.state_dict(), "val_rows": val.tolist(),
        "metrics": {"before": before, "after": after}, "config": config_of(args),
    }, out)
    log(f"  saved to {out}")


# ===========================================================================

def write_test_tsv(path, n=60, bits=48, seed=0):
    """Synthetic hash-phenotype TSV: the phenotype is the number of 1s in the first 8 bits + noise."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        f.write("hash\theight\n")
        for _ in range(n):
            h = rng.integers(0, 2, bits)
            f.write("".join(map(str, h)) + f"\t{160 + 4 * h[:8].sum() + rng.normal(0, 2):.1f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hash-pheno", help="TSV with 2 columns: genomic hash (0/1 string), phenotype")
    ap.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    ap.add_argument("--hash-window", type=int, default=4096, help="bits (tokens) per Evo 2 forward pass")
    ap.add_argument("--hash-max-windows", type=int, default=0,
                    help="cap windows per hash (0 = use all; random windows in training, evenly spaced in eval)")
    ap.add_argument("--out", help="output .pt (default <TSV folder>/evo2_hash_<phenotype>.pt)")
    e = ap.add_argument_group("Evo 2 and training")
    e.add_argument("--evo2-model", default="evo2_7b_base", help="evo2_7b_base (8K ctx) or evo2_7b")
    e.add_argument("--layer", default="blocks.28.mlp.l3", help="read-out layer; forward stops here")
    e.add_argument("--embed-module", help="name of Evo 2's token embedding module (auto-detected)")
    e.add_argument("--mock-evo2", action="store_true",
                   help="small CPU stand-in model instead of Evo 2, to check the script without a GPU")
    e.add_argument("--mock-hidden", type=int, default=64)
    e.add_argument("--mock-layers", type=int, default=4)
    e.add_argument("--device", help="--mock-evo2: cpu / cuda")
    e.add_argument("--train-blocks", type=int, default=4, help="0 = Evo 2 frozen, train new layers only")
    e.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
    e.add_argument("--epochs", type=int, default=1)
    e.add_argument("--lr", type=float, default=1e-5, help="for unfrozen Evo 2 blocks")
    e.add_argument("--head-lr", type=float, default=1e-3, help="for the new layers")
    e.add_argument("--val-frac", type=float, default=0.1, help="fraction of rows held out")
    e.add_argument("--val-examples", type=int, default=300, help="max validation examples per evaluation")
    e.add_argument("--log-every", type=int, default=10)
    e.add_argument("--eval-every", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="self-test on a synthetic TSV with --mock-evo2 (CPU)")
    args = ap.parse_args(argv)
    if args.test:
        args.hash_pheno = str(test_workdir("hash") / "hash_height.tsv")
        write_test_tsv(args.hash_pheno)
        args.mock_evo2, args.device, args.mock_hidden, args.mock_layers = True, "cpu", 32, 2
        args.train_blocks, args.batch, args.epochs, args.log_every, args.hash_window = 1, 8, 3, 5, 32
    if not args.hash_pheno:
        ap.error("--hash-pheno is required")
    if not os.path.exists(args.hash_pheno):
        die(f"{args.hash_pheno} not found")
    if not 0 < args.val_frac < 1:
        die("--val-frac must be between 0 and 1")
    if args.train_blocks < 0:
        die("--train-blocks must be >= 0")
    for name in ("hash_window", "batch", "epochs", "val_examples", "log_every", "eval_every"):
        if getattr(args, name) < 1:
            die(f"--{name.replace('_', '-')} must be >= 1")
    train_hash(args)
    log("done.")


if __name__ == "__main__":
    main()