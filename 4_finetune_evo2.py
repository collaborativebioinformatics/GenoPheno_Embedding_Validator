"""
Step 4 of 4 - fine-tune Evo 2 7B on the node embeddings from 3_node_embeddings.py
(node2vec, gnn or xgboost). No DNA sequence is used.

  - A training example is one individual on one chromosome: the clusters that person carries,
    ordered by genomic position, each replaced by its node embedding.
  - A linear layer projects the embeddings to Evo 2's hidden size; a learned per-chromosome vector
    is added (useful for node2vec, whose vectors aren't aligned across chromosomes). These vectors
    are fed into Evo 2 IN PLACE OF its nucleotide token embeddings (a hook swaps the output of
    Evo 2's embedding layer).
  - Evo 2 runs up to --layer (default blocks.28.mlp.l3); the last --train-blocks blocks before it
    are unfrozen, the rest stay frozen.
  - --objective next     (default): predict the NEXT cluster along the chromosome (softmax over all
                          clusters of that chromosome), mirroring Evo 2's next-token pre-training.
    --objective ancestry : predict the individual's super-population from the mean-pooled sequence.
  - Individuals are split train/validation; validation loss and accuracy are reported before and
    after training.

INPUT  (in --graph-dir)
  nodes.csv.gz              CSV, N rows: node, id, chrom, region, start, end, support
                            (genomic order within a chromosome is taken from start)
  individuals.txt           I lines, one individual per line (columns of carriers.npz)
  carriers.npz              scipy.sparse CSR uint8 [N, I], node x individual carriers
  node_emb_<method>.npy     float32 [N, dim], row i = node i of nodes.csv.gz
  individual_ancestry.csv   CSV: individual, ancestry (empty = unknown)   (--objective ancestry)

OUTPUT (in --graph-dir)
  evo2_<objective>_<method>.pt   torch.save dict:
      evo2_model, layer, objective, graph_method, chroms
      classes          list of ancestry labels (ancestry objective) or None
      trained_blocks   {parameter name in evo.model: tensor} for the unfrozen Evo 2 blocks
      proj_in          state_dict: Linear(dim -> hidden) + LayerNorm
      chrom_emb        state_dict: Embedding(n_chroms, hidden), row order = chroms
      head             state_dict: next -> LayerNorm + Linear(hidden -> dim);
                                   ancestry -> LayerNorm + Linear(hidden -> n_classes)
      val_individuals  held-out individuals;  metrics {before, after};  config
  To reuse: Evo2(evo2_model), copy trained_blocks into evo.model (load_state_dict(strict=False)),
  rebuild proj_in / chrom_emb / head with the same shapes and load their state_dicts.

USAGE
  python 4_finetune_evo2.py --method gnn
  python 4_finetune_evo2.py --method node2vec --chroms chr21,chr22        # train on 2 chromosomes
  python 4_finetune_evo2.py --method xgboost --objective ancestry
  python 4_finetune_evo2.py --method gnn --train-blocks 0                  # frozen Evo 2 baseline

REQUIRES  numpy, pandas, scipy, torch, evo2 (pip install evo2; Linux, CUDA 12.1+, Python 3.11/3.12;
          the 7B models run in bf16 without Transformer Engine; an 80 GB GPU is recommended).

NOTE  The official Evo 2 training paths are Savanna and NVIDIA BioNeMo. This script uses the `evo2`
      inference package in plain PyTorch and, after the first step, checks that gradients reach the
      input projection and the unfrozen blocks, stopping with a clear message if they don't.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


def log(msg=""):
    print(msg, flush=True)


def gfile(args, name):
    return os.path.join(args.graph_dir, name)


def chrom_key(c):
    m = re.fullmatch(r"chr(\d+)", c)
    return (0, int(m.group(1)), "") if m else (1, 0, c)


class GenomeData:
    """Node embeddings + each individual's clusters per chromosome, in genomic order."""

    def __init__(self, args):
        nodes = pd.read_csv(gfile(args, "nodes.csv.gz"))
        Z = np.load(gfile(args, f"node_emb_{args.method}.npy")).astype(np.float32)
        if len(Z) != len(nodes):
            raise SystemExit(f"node_emb_{args.method}.npy has {len(Z)} rows, nodes.csv.gz has {len(nodes)}")
        self.Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

        present = sorted(nodes["chrom"].unique(), key=chrom_key)
        self.chroms = args.chroms or present
        unknown = set(self.chroms) - set(present)
        if unknown:
            raise SystemExit(f"chromosomes not in this graph: {sorted(unknown)}; graph has {present}")
        self.cidx = {c: present.index(c) for c in self.chroms}
        self.n_chrom_slots = len(present)

        code = nodes["chrom"].map({c: i for i, c in enumerate(present)}).to_numpy()
        start = nodes["start"].fillna(0).to_numpy()
        self.order = np.lexsort((np.arange(len(nodes)), start, code))   # rank -> node index
        code_sorted = code[self.order]
        self.bounds = {c: (int(np.searchsorted(code_sorted, i)), int(np.searchsorted(code_sorted, i, "right")))
                       for c, i in ((c, present.index(c)) for c in present)}
        C = sp.load_npz(gfile(args, "carriers.npz")).tocsr()
        self.by_ind = C[self.order].T.tocsr()                            # [I, N], columns = ranks
        self.by_ind.sort_indices()

    def tokens(self, chrom, i):
        """(node indices, positions among the chromosome's nodes) of individual i, genomic order."""
        row = self.by_ind.indices[self.by_ind.indptr[i]:self.by_ind.indptr[i + 1]]
        s, e = self.bounds[chrom]
        lo, hi = np.searchsorted(row, [s, e])
        ranks = row[lo:hi].astype(np.int64)
        return self.order[ranks], ranks - s

    def candidates(self, chrom):
        s, e = self.bounds[chrom]
        return self.order[s:e]


def load_individual_ancestry(args, individuals):
    f = gfile(args, "individual_ancestry.csv")
    if not os.path.exists(f):
        raise SystemExit(f"{f} not found - run 2_ancestry_labels.py first (needed for --objective ancestry)")
    a = pd.read_csv(f, dtype=str, keep_default_na=False)
    lookup = dict(zip(a["individual"], a["ancestry"]))
    return np.array([lookup.get(i) or None for i in individuals], dtype=object)


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


def train(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from evo2 import Evo2

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = GenomeData(args)
    individuals = open(gfile(args, "individuals.txt")).read().split("\n")
    n_ind = len(individuals)

    y_ind, classes = None, None
    if args.objective == "ancestry":
        anc = load_individual_ancestry(args, individuals)
        known = np.array([a is not None for a in anc])
        classes = np.array(sorted(set(anc[known])))
        y_ind = np.full(n_ind, -1)
        y_ind[known] = np.searchsorted(classes, anc[known])
        pool = np.where(known)[0]
    else:
        pool = np.arange(n_ind)

    perm = rng.permutation(pool)
    n_val = max(1, int(args.val_frac * len(perm)))
    val_inds, tr_inds = perm[:n_val], perm[n_val:]
    examples = lambda inds: [(c, i) for c in data.chroms for i in inds if len(data.tokens(c, i)[0]) >= 2]
    train_ex, val_ex = examples(tr_inds), examples(val_inds)
    if len(val_ex) > args.val_examples:
        val_ex = [val_ex[j] for j in rng.choice(len(val_ex), args.val_examples, replace=False)]
    log(f"[evo2] objective={args.objective}, embeddings={args.method}, {len(data.chroms)} chromosome(s), "
        f"{len(tr_inds):,} train / {len(val_inds):,} val individuals, {len(train_ex):,} training sequences")

    evo = Evo2(args.evo2_model)
    net = evo.model
    dev = next(net.parameters()).device
    n_conv = convert_inference_tensors(net)
    if n_conv:
        log(f"  converted {n_conv:,} inference-mode tensors from the evo2 loader to trainable tensors")

    # ---- freeze all, unfreeze the last `train_blocks` blocks up to the read-out layer
    m = re.match(r"blocks\.(\d+)\.", args.layer)
    if not m:
        raise SystemExit("--layer must look like blocks.<i>.<...>, e.g. blocks.28.mlp.l3")
    last = int(m.group(1))
    for p in net.parameters():
        p.requires_grad_(False)
    trainable = []
    if args.train_blocks > 0:
        for i in range(max(0, last - args.train_blocks + 1), last + 1):
            for name, p in net.blocks[i].named_parameters():
                p.requires_grad_(True); trainable.append((f"blocks.{i}.{name}", p))

    # ---- hooks: replace token embeddings with projected node embeddings; stop at --layer
    class StopForward(Exception):
        pass

    state = {"soft": None, "h": None, "embed_shape": None}

    def embed_hook(_m, _inp, out):
        state["embed_shape"] = tuple(out.shape)
        if state["soft"] is None:
            return None
        if state["soft"].shape != out.shape:
            raise RuntimeError(f"soft input {tuple(state['soft'].shape)} != embedding output {tuple(out.shape)}")
        return state["soft"].to(out.dtype)

    def layer_hook(_m, _inp, out):
        state["h"] = out[0] if isinstance(out, tuple) else out
        raise StopForward

    find_embedding_module(net, args.embed_module).register_forward_hook(embed_hook)
    dict(net.named_modules())[args.layer].register_forward_hook(layer_hook)

    def run(soft):
        """soft: [T, hidden] -> Evo 2 hidden states at --layer, [T, hidden_out] float32."""
        state["soft"] = soft.unsqueeze(0)
        dummy = torch.zeros(1, soft.shape[0], dtype=torch.int, device=dev)  # ids are ignored
        try:
            net(dummy)
        except StopForward:
            pass
        state["soft"] = None
        return state.pop("h").float().squeeze(0)

    with torch.no_grad():                       # discover sizes with one plain forward pass
        try:
            net(torch.zeros(1, 8, dtype=torch.int, device=dev))
        except StopForward:
            pass
    hidden_in, hidden_out = state["embed_shape"][-1], state["h"].shape[-1]
    state["h"] = None

    # ---- new trainable layers
    dim = data.Z.shape[1]
    proj_in = nn.Sequential(nn.Linear(dim, hidden_in), nn.LayerNorm(hidden_in)).to(dev)
    chrom_emb = nn.Embedding(data.n_chrom_slots, hidden_in).to(dev)
    nn.init.normal_(chrom_emb.weight, std=0.02)
    if args.objective == "next":
        head = nn.Sequential(nn.LayerNorm(hidden_out), nn.Linear(hidden_out, dim)).to(dev)
    else:
        head = nn.Sequential(nn.LayerNorm(hidden_out), nn.Linear(hidden_out, len(classes))).to(dev)
    new_params = list(proj_in.parameters()) + list(chrom_emb.parameters()) + list(head.parameters())

    Z_t = torch.from_numpy(data.Z).to(dev)
    cand_n = {c: F.normalize(Z_t[torch.from_numpy(data.candidates(c)).to(dev)], dim=-1) for c in data.chroms}

    opt = torch.optim.AdamW([
        {"params": [p for _, p in trainable], "lr": args.lr},
        {"params": new_params, "lr": args.head_lr},
    ], weight_decay=0.01)
    log(f"  {args.evo2_model}: hidden {hidden_in}, read-out {args.layer}, unfrozen blocks "
        f"{f'{last - args.train_blocks + 1}-{last}' if trainable else 'none'} "
        f"({sum(p.numel() for _, p in trainable) / 1e6:.0f}M params)")

    def window(n_tok, train_mode):
        if n_tok <= args.max_len:
            return 0, n_tok
        s = int(rng.integers(0, n_tok - args.max_len + 1)) if train_mode else (n_tok - args.max_len) // 2
        return s, s + args.max_len

    def example_loss(chrom, i, train_mode):
        """(loss, n_correct, n_predictions) for one individual-chromosome sequence."""
        node_idx, local = data.tokens(chrom, i)
        s, e = window(len(node_idx), train_mode)
        tok = torch.from_numpy(node_idx[s:e]).to(dev)
        x = proj_in(Z_t[tok]) + chrom_emb.weight[data.cidx[chrom]]
        h = run(x)                                                        # [T, hidden_out]
        if args.objective == "next":
            pred = F.normalize(head(h[:-1]), dim=-1)                      # predict t+1 from <= t
            logits = pred @ cand_n[chrom].T / args.temperature           # over the chromosome's nodes
            target = torch.from_numpy(local[s + 1:e]).to(dev)
            return F.cross_entropy(logits, target), (logits.argmax(-1) == target).sum().item(), len(target)
        logits = head(h.mean(0, keepdim=True))
        target = torch.tensor([y_ind[i]], device=dev)
        return F.cross_entropy(logits, target), int(logits.argmax(-1).item() == y_ind[i]), 1

    def evaluate():
        net.eval(); proj_in.eval(); head.eval()
        tot = correct = count = 0
        with torch.no_grad():
            for c, i in val_ex:
                l, k, n = example_loss(c, i, False)
                tot += l.item(); correct += k; count += n
        net.train(); proj_in.train(); head.train()
        return tot / max(len(val_ex), 1), correct / max(count, 1)

    what = "next-cluster top-1" if args.objective == "next" else "ancestry"
    l0, a0 = evaluate()
    log(f"  before training: val loss {l0:.4f}, val {what} accuracy {a0:.3f}")

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_ex))
        for b in range(0, len(order) - args.batch + 1, args.batch):
            opt.zero_grad()
            batch_loss = 0.0
            for j in order[b:b + args.batch]:            # gradient accumulation over the batch
                loss, _, _ = example_loss(*train_ex[j], True)
                (loss / args.batch).backward()
                batch_loss += loss.item() / args.batch

            if step == 0:
                if all(p.grad is None for p in proj_in.parameters()):
                    raise RuntimeError("No gradient reached the input projection: this evo2 version's "
                                       "forward pass is not differentiable. Fine-tune via BioNeMo instead.")
                if trainable and all(p.grad is None for _, p in trainable):
                    raise RuntimeError("No gradient reached the unfrozen Evo 2 blocks; try --train-blocks 0 "
                                       "(trains only the new layers through frozen Evo 2).")
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable] + new_params, 1.0)
            opt.step(); step += 1
            if step % args.log_every == 0:
                log(f"  epoch {epoch + 1} step {step}: loss {batch_loss:.4f}  {(time.time() - t0) / step:.1f}s/step")
            if step % args.eval_every == 0:
                l, a = evaluate()
                log(f"  [eval] val loss {l:.4f}, val {what} accuracy {a:.3f}")

    l, a = evaluate()
    log(f"  final: val loss {l:.4f} (was {l0:.4f}), val {what} accuracy {a:.3f} (was {a0:.3f})")
    out = gfile(args, f"evo2_{args.objective}_{args.method}.pt")
    torch.save({
        "evo2_model": args.evo2_model, "layer": args.layer, "objective": args.objective,
        "graph_method": args.method, "chroms": data.chroms,
        "classes": None if classes is None else classes.tolist(),
        "trained_blocks": {n: p.detach().cpu() for n, p in trainable},
        "proj_in": proj_in.state_dict(), "chrom_emb": chrom_emb.state_dict(), "head": head.state_dict(),
        "val_individuals": [individuals[i] for i in val_inds],
        "metrics": {"before": {"loss": l0, "accuracy": a0}, "after": {"loss": l, "accuracy": a}},
        "config": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))},
    }, out)
    log(f"  saved to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["node2vec", "gnn", "xgboost"], required=True,
                    help="which embeddings from 3_node_embeddings.py to use")
    ap.add_argument("--graph-dir", default="haplograph/genome", help="output folder of 1_build_graph.py")
    ap.add_argument("--chroms", help="train on these chromosomes of the graph only (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["next", "ancestry"], default="next")
    ap.add_argument("--evo2-model", default="evo2_7b_base", help="evo2_7b_base (8K ctx) or evo2_7b")
    ap.add_argument("--layer", default="blocks.28.mlp.l3", help="read-out layer; forward stops here")
    ap.add_argument("--embed-module", help="name of Evo 2's token embedding module (auto-detected)")
    ap.add_argument("--train-blocks", type=int, default=4, help="0 = Evo 2 frozen, train new layers only")
    ap.add_argument("--max-len", type=int, default=1024, help="max clusters per training sequence")
    ap.add_argument("--batch", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="for unfrozen Evo 2 blocks")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="for projection, chromosome vectors, head")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of individuals held out")
    ap.add_argument("--val-examples", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
    args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()] if args.chroms else None

    for f, hint in [("nodes.csv.gz", "1_build_graph.py"), ("individuals.txt", "1_build_graph.py"),
                    ("carriers.npz", "1_build_graph.py"),
                    (f"node_emb_{args.method}.npy", f"3_node_embeddings.py --method {args.method}")]:
        if not os.path.exists(gfile(args, f)):
            raise SystemExit(f"missing {f} in {args.graph_dir} - run {hint} first")
    train(args)
    log("done.")


if __name__ == "__main__":
    main()