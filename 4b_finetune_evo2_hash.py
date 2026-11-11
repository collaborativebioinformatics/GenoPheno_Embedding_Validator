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
  python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv                     # regression
  python 4b_finetune_evo2_hash.py --hash-pheno hash_pop.tsv --task classification
  python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv --train-blocks 0    # frozen baseline

REQUIRES  numpy, pandas, torch, evo2 (pip install evo2; Linux, CUDA 12.1+, Python 3.11/3.12;
          the 7B models run in bf16 without Transformer Engine; an 80 GB GPU is recommended).

NOTE  The official Evo 2 training paths are Savanna and NVIDIA BioNeMo. This script uses the `evo2`
      inference package in plain PyTorch and, after the first step, checks that gradients reach the
      new layers and the unfrozen blocks, stopping with a clear message if they don't.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd


def load_hash_pheno(path, task="auto"):
    """Returns hashes (list of str), y (float64 or int64), task, classes, phenotype name, n_skipped."""
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3)
    if raw.shape[1] < 2 or len(raw) == 0:
        raise SystemExit(f"{path}: expected a tab-separated file with 2 columns (hash, phenotype)")
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
        raise SystemExit(f"{path}: fewer than 2 valid rows")

    num = pd.to_numeric(v, errors="coerce")
    if task == "auto":
        task = "regression" if num.notna().all() and num.nunique() > 10 else "classification"
    if task == "regression":
        if num.isna().any():
            bad = v[num.isna()].unique()[:5].tolist()
            raise SystemExit(f"non-numeric phenotype values {bad}; use --task classification")
        return hashes, num.to_numpy(np.float64), task, None, name, n_skipped
    classes = np.array(sorted(v.unique()))
    return hashes, np.searchsorted(classes, v.to_numpy()).astype(np.int64), task, classes, name, n_skipped


def log(msg=""):
    print(msg, flush=True)


class StopForward(Exception):
    pass


def find_embedding_module(net, name):
    import torch.nn as nn
    mods = dict(net.named_modules())
    if name:
        return mods[name]
    for n, m in mods.items():
        if n.endswith("embedding_layer"):
            return m
    for n, m in mods.items():
        if isinstance(m, nn.Embedding):
            return m
    raise SystemExit("could not find Evo 2's token embedding layer; pass --embed-module")


def convert_inference_tensors(net):
    """The evo2 package loads its weights under torch.inference_mode(), which makes them
    'inference tensors' that autograd can neither train nor save for backward (needed even for
    frozen blocks, so gradients can flow back to the input projection). Replace every parameter,
    buffer and plain tensor attribute that is an inference tensor with a normal clone, one tensor
    at a time (peak extra memory = one tensor). Tied parameters stay tied."""
    import torch
    memo, n = {}, 0

    def fresh(t, as_param):
        nonlocal n
        if id(t) not in memo:
            c = t.clone()                      # outside inference mode -> normal tensor
            memo[id(t)] = torch.nn.Parameter(c, requires_grad=False) if as_param else c
            n += 1
        return memo[id(t)]

    with torch.no_grad():
        for mod in net.modules():
            for name, p in list(mod._parameters.items()):
                if p is not None and p.is_inference():
                    mod._parameters[name] = fresh(p, True)
            for name, b in list(mod._buffers.items()):
                if b is not None and b.is_inference():
                    mod._buffers[name] = fresh(b, False)
            for name, v in list(vars(mod).items()):
                if (isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter)
                        and v.is_inference()):
                    setattr(mod, name, fresh(v, False))
    memo.clear()
    torch.cuda.empty_cache()
    return n


class Evo2Runner:
    """Loads Evo 2, unfreezes the last --train-blocks blocks up to --layer, and runs the model up to
    --layer either on its own tokens (run_ids) or on replacement input vectors (run_soft)."""

    def __init__(self, args):
        import torch
        from evo2 import Evo2
        self.torch = torch
        self.evo = Evo2(args.evo2_model)
        net = self.net = self.evo.model
        self.dev = next(net.parameters()).device
        n_conv = convert_inference_tensors(net)
        if n_conv:
            log(f"  converted {n_conv:,} inference-mode tensors from the evo2 loader to trainable tensors")

        m = re.match(r"blocks\.(\d+)\.", args.layer)
        if not m:
            raise SystemExit("--layer must look like blocks.<i>.<...>, e.g. blocks.28.mlp.l3")
        last = int(m.group(1))
        for p in net.parameters():
            p.requires_grad_(False)
        self.trainable = []
        if args.train_blocks > 0:
            for i in range(max(0, last - args.train_blocks + 1), last + 1):
                for name, p in net.blocks[i].named_parameters():
                    p.requires_grad_(True)
                    self.trainable.append((f"blocks.{i}.{name}", p))

        self.state = {"soft": None, "h": None, "embed_shape": None}
        find_embedding_module(net, args.embed_module).register_forward_hook(self._embed_hook)
        dict(net.named_modules())[args.layer].register_forward_hook(self._layer_hook)

        with torch.no_grad():                  # discover sizes with one plain forward pass
            self._forward(torch.zeros(1, 8, dtype=torch.int, device=self.dev))
        self.hidden_in = self.state["embed_shape"][-1]
        self.hidden_out = self.state.pop("h").shape[-1]
        self.state["h"] = None
        n_train = sum(p.numel() for _, p in self.trainable) / 1e6
        blocks = f"{last - args.train_blocks + 1}-{last}" if self.trainable else "none"
        log(f"  {args.evo2_model}: hidden {self.hidden_in}, read-out {args.layer}, "
            f"unfrozen blocks {blocks} ({n_train:.0f}M params)")

    def _embed_hook(self, _m, _inp, out):
        self.state["embed_shape"] = tuple(out.shape)
        soft = self.state["soft"]
        if soft is None:
            return None
        if soft.shape != out.shape:
            raise RuntimeError(f"soft input {tuple(soft.shape)} != embedding output {tuple(out.shape)}")
        return soft.to(out.dtype)

    def _layer_hook(self, _m, _inp, out):
        self.state["h"] = out[0] if isinstance(out, tuple) else out
        raise StopForward

    def _forward(self, ids):
        try:
            self.net(ids)
        except StopForward:
            pass

    def run_ids(self, ids):
        """ids: int tensor [T] of Evo 2 tokens -> hidden states at --layer, [T, hidden_out] float32."""
        self._forward(ids.unsqueeze(0))
        return self.state.pop("h").float().squeeze(0)

    def run_soft(self, soft):
        """soft: [T, hidden_in] replacement input vectors -> [T, hidden_out] float32."""
        self.state["soft"] = soft.unsqueeze(0)
        dummy = self.torch.zeros(1, soft.shape[0], dtype=self.torch.int, device=self.dev)
        try:
            self._forward(dummy)
        finally:
            self.state["soft"] = None
        return self.state.pop("h").float().squeeze(0)


def fit(args, rnr, train_ex, example_loss, evaluate, describe, new_modules, grad_check):
    """Shared training loop. example_loss(ex, train_mode) -> (loss, info); evaluate() -> metrics."""
    torch = rnr.torch
    rng = np.random.default_rng(args.seed + 1)
    new_params = [p for m in new_modules for p in m.parameters()]
    opt = torch.optim.AdamW([
        {"params": [p for _, p in rnr.trainable], "lr": args.lr},
        {"params": new_params, "lr": args.head_lr},
    ], weight_decay=0.01)

    def run_eval():
        rnr.net.eval()
        for m in new_modules:
            m.eval()
        with torch.no_grad():
            res = evaluate()
        rnr.net.train()
        for m in new_modules:
            m.train()
        return res

    before = run_eval()
    log(f"  before training: {describe(before)}")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_ex))
        for b in range(0, len(order) - args.batch + 1, args.batch):
            opt.zero_grad()
            batch_loss = 0.0
            for j in order[b:b + args.batch]:            # gradient accumulation over the batch
                loss = example_loss(train_ex[j], True)[0]
                (loss / args.batch).backward()
                batch_loss += loss.item() / args.batch
            if step == 0:
                what, params = grad_check
                if all(p.grad is None for p in params):
                    raise RuntimeError(f"No gradient reached {what}: this evo2 version's forward pass is "
                                       "not differentiable. Fine-tune via BioNeMo instead.")
                if rnr.trainable and all(p.grad is None for _, p in rnr.trainable):
                    raise RuntimeError("No gradient reached the unfrozen Evo 2 blocks; try --train-blocks 0 "
                                       "(trains only the new layers through frozen Evo 2).")
            torch.nn.utils.clip_grad_norm_([p for _, p in rnr.trainable] + new_params, 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                log(f"  epoch {epoch + 1} step {step}: loss {batch_loss:.4f}  {(time.time() - t0) / step:.1f}s/step")
            if step % args.eval_every == 0:
                log(f"  [eval] {describe(run_eval())}")
    after = run_eval()
    log(f"  final: {describe(after)}")
    log(f"  (before training: {describe(before)})")
    return before, after


def config_of(args):
    return {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))}


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
    rnr = Evo2Runner(args)
    dev = rnr.dev
    head = nn.Sequential(nn.LayerNorm(rnr.hidden_out),
                         nn.Linear(rnr.hidden_out, 1 if task == "regression" else len(classes))).to(dev)

    def encode(h, train_mode):
        """Mean of Evo 2's --layer output over every token (bit) of the hash."""
        ids = torch.tensor(rnr.evo.tokenizer.tokenize(h), dtype=torch.int, device=dev)
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
    before, after = fit(args, rnr, list(tr), example_loss, evaluate, describe,
                        [head], ("the phenotype head", list(head.parameters())))

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "phenotype"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.hash_pheno)), f"evo2_hash_{safe}.pt")
    torch.save({
        "mode": "hash", "evo2_model": args.evo2_model, "layer": args.layer, "task": task,
        "phenotype": name, "classes": None if classes is None else classes.tolist(),
        "target_mean": mu if task == "regression" else None,
        "target_std": sd if task == "regression" else None,
        "hash_window": args.hash_window,
        "trained_blocks": {n: p.detach().cpu() for n, p in rnr.trainable},
        "head": head.state_dict(), "val_rows": val.tolist(),
        "metrics": {"before": before, "after": after}, "config": config_of(args),
    }, out)
    log(f"  saved to {out}")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hash-pheno", required=True,
                    help="TSV with 2 columns: genomic hash (0/1 string), phenotype")
    ap.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    ap.add_argument("--hash-window", type=int, default=4096, help="bits (tokens) per Evo 2 forward pass")
    ap.add_argument("--hash-max-windows", type=int, default=0,
                    help="cap windows per hash (0 = use all; random windows in training, evenly spaced in eval)")
    ap.add_argument("--out", help="output .pt (default <TSV folder>/evo2_hash_<phenotype>.pt)")
    e = ap.add_argument_group("Evo 2 and training")
    e.add_argument("--evo2-model", default="evo2_7b_base", help="evo2_7b_base (8K ctx) or evo2_7b")
    e.add_argument("--layer", default="blocks.28.mlp.l3", help="read-out layer; forward stops here")
    e.add_argument("--embed-module", help="name of Evo 2's token embedding module (auto-detected)")
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
    args = ap.parse_args()
    if not os.path.exists(args.hash_pheno):
        raise SystemExit(f"{args.hash_pheno} not found")
    train_hash(args)
    log("done.")


if __name__ == "__main__":
    main()