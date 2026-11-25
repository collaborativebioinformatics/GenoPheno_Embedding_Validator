"""
Step 3 - one embedding per graph node (--method: a list, or all = node2vec, gnn, carrier_svd). No phenotype or population labels are used; everything learned from individuals uses
the graph's training individuals only.
  carrier_svd  the SVD-compressed carrier profile of each node (which training individuals carry it):
               the input of gnn and xgboost, kept as the reference they have to beat in step 6
  node2vec     biased random walks on the graph (--p, --q, weights log(1 + lift)) + skip-gram
  gnn          2-layer GCN autoencoder (link reconstruction) on the carrier_svd features
  xgboost      trees predict each node's cis partner in the next haploblock (the frequency rank,
               within that haploblock, of the cluster that most often follows it on a training
               haplotype; --xgb-classes classes) from its own and its neighbourhood's carrier
               profile; one-hot leaves -> SVD = the embedding. Memory-bounded (sampled training /
               SVD, chunked embedding); trees = --xgb-trees x classes. Not in --method all: on chr21
               its target is near chance and it does not beat carrier_svd in step 6
  evo2_seq     (GPU) the backbone (Evo 2 at --layer) mean-pooled over each cluster's representative
               haplotype sequence (step 1 --sequences; --seq-windows windows of --seq-window bp),
               centred within its haploblock (clusters of a block differ by a few variants) and
               reduced to --dim by PCA. Defined for every cluster, including those never seen in
               training. Cached per haploblock in <graph-dir>/evo2_seq_cache/, so a run can resume.
               Cost: one --seq-window pass per cluster (chr21: 167k clusters)
--rapids runs the two SVDs on the GPU (RAPIDS cuML + CuPy; the carrier SVD is fitted on
--rapids-fit-rows sampled rows). Graphs have within-chromosome edges only, so node2vec vectors are
comparable within a chromosome; the others are comparable genome-wide.

OUTPUT (in --graph-dir)
  node_emb_<method>.npy    float32 [nodes, --dim], row i = node i of nodes.csv.gz
  node_emb_<method>.json   settings, training summary, train_set_id, graph_id, settings_id
                           (an output is reused only while all three still match)
  node_emb_<method>.csv.gz (--csv) id, chrom, e0 ... e<dim-1>
  features_svd*.npy        cached carrier features
  evo2_seq_cache/<id>/     per-haploblock backbone outputs of evo2_seq (one folder per backbone setting)

USAGE
  python 3_node_embeddings.py --graph-dir haplograph/genome_train_half_A
  python 3_node_embeddings.py --method xgboost --graph-dir $G --device cuda --rapids
  python 3_node_embeddings.py --method evo2_seq --graph-dir $G                 # GPU, step 1 --sequences
  python 3_node_embeddings.py --test            # synthetic data, CPU, seconds
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from haplo_common import (DEFAULT_METHODS, METHODS, check_args, chrom_key, die, graph_id, log, read_graph_individuals,
                          require_files, selftest_graph_dir, torch_device, train_columns)



def gfile(args, name):
    return os.path.join(args.graph_dir, name)


# ---------------------------------------------------------------------------
# SVD: scipy on CPU, or RAPIDS cuML on the GPU (--rapids)
# ---------------------------------------------------------------------------

def import_rapids():
    try:
        import cupy  # noqa: F401
        from cuml.decomposition import TruncatedSVD  # noqa: F401
    except ImportError as e:
        die(f"--rapids needs RAPIDS cuML and CuPy ({e}); see DOCUMENTATION.md, or run without --rapids")


def gpu_svd_tall(M, k, seed=0, fit_rows=150_000, chunk=200_000):
    """cuML TruncatedSVD for a TALL sparse M [n, d] (d small), VRAM-safe: fit on a dense row sample,
    then project every row through the components in dense chunks. cuML needs dense input, so only
    the sample / chunk slices are densified. Returns scores U*S [n, k], largest component first."""
    import cupy as cp
    from cuml.decomposition import TruncatedSVD
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    samp = np.sort(rng.choice(n, min(n, fit_rows), replace=False))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(cp.asarray(M[samp].toarray().astype("float32")))
    order = np.argsort(cp.asnumpy(svd.singular_values_))[::-1]
    Vo = cp.asarray(svd.components_)[cp.asarray(order)]           # [k, d], largest first
    scores = np.empty((n, k), dtype="float32")
    for b in range(0, n, chunk):
        Xb = cp.asarray(M[b:b + chunk].toarray().astype("float32"))
        scores[b:b + chunk] = cp.asnumpy(Xb @ Vo.T)              # U*S = X @ V^T
    return scores


def svd_basis(M, k, seed, rapids):
    """Right singular vectors of M -> V [n_cols, k], largest component first."""
    if rapids:
        import cupy as cp
        from cuml.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k, random_state=seed)
        svd.fit(cp.asarray(M.toarray().astype(np.float32)))
        S, Vt = cp.asnumpy(svd.singular_values_), cp.asnumpy(svd.components_)
    else:
        _, S, Vt = svds(M, k=k, random_state=seed)
    return Vt[np.argsort(S)[::-1]].T.astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def carrier_features(args):
    """SVD-compressed, standardized carrier profiles of the TRAINING individuals [N, svd_dim] (cached)."""
    tag = f"_rapids{args.rapids_fit_rows}" if args.rapids else ""
    f = gfile(args, f"features_svd{args.svd_dim}_{args.train_id}_{args.graph_id}_s{args.seed}{tag}.npy")
    if os.path.exists(f) and not args.force:
        return np.load(f)
    C = sp.load_npz(gfile(args, "carriers.npz")).tocsc()[:, args.train_idx].tocsr().astype(np.float32)
    k = min(args.svd_dim, min(C.shape) - 1)
    if k < 1:
        die(f"carrier matrix {C.shape} is too small for an SVD")
    log(f"  SVD of carrier matrix {C.shape} (training individuals) -> {k} features"
        + (" on the GPU (RAPIDS cuML)" if args.rapids else ""))
    if args.rapids:
        X = gpu_svd_tall(C, k, args.seed, args.rapids_fit_rows, args.rapids_chunk)
    else:
        U, S, _ = svds(C, k=k, random_state=args.seed)
        X = (U * S)[:, np.argsort(S)[::-1]].astype(np.float32)              # largest component first
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    np.save(f, X)
    return X


def normalized_adjacency(A):
    """D^-1/2 (A + I) D^-1/2"""
    A_hat = A + sp.identity(A.shape[0], dtype=np.float32, format="csr")
    d = np.asarray(A_hat.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    return (Dm @ A_hat @ Dm).tocsr().astype(np.float32)


# ---------------------------------------------------------------------------
# node2vec
# ---------------------------------------------------------------------------

class Node2VecWalker:
    """Vectorized (optionally weighted) node2vec random walks on a CSR adjacency.

    Second-order bias by rejection sampling: a candidate next node x (drawn proportional to edge
    weight from the current node v, coming from t) is accepted with probability
    (1/p if x == t, 1 if x is a neighbour of t, 1/q otherwise) / max(1/p, 1, 1/q).
    Isolated nodes stay where they are.
    """

    def __init__(self, A, p=1.0, q=1.0, weighted=True, seed=0, max_tries=50):
        A = A.tocsr()
        A.sort_indices()
        self.n = A.shape[0]
        self.indptr = A.indptr.astype(np.int64)
        self.indices = A.indices.astype(np.int64)
        self.deg = np.diff(self.indptr)
        w = A.data.astype(np.float64) if weighted else np.ones(A.nnz)
        w = np.maximum(w, 1e-12)
        self.cum = np.concatenate([[0.0], np.cumsum(w)])          # cum[k] = weight before edge k
        self.row_w = self.cum[self.indptr[1:]] - self.cum[self.indptr[:-1]]
        rows = np.repeat(np.arange(self.n, dtype=np.int64), self.deg)
        self.keys = rows * self.n + self.indices                  # sorted: row-major CSR
        self.p, self.q = p, q
        self.biased = not (p == 1 and q == 1)
        self.max_prob = max(1.0 / p, 1.0, 1.0 / q)
        self.max_tries = max_tries
        self.rng = np.random.default_rng(seed)

    def sample_neighbor(self, v):
        u = self.cum[self.indptr[v]] + self.rng.random(len(v)) * self.row_w[v]
        k = np.searchsorted(self.cum, u, side="right") - 1
        k = np.clip(k, self.indptr[v], self.indptr[v + 1] - 1)
        return self.indices[k]

    def is_edge(self, a, b):
        key = a * self.n + b
        pos = np.searchsorted(self.keys, key)
        pos = np.minimum(pos, len(self.keys) - 1)
        return self.keys[pos] == key

    def walk(self, starts, length):
        starts = np.asarray(starts, dtype=np.int64)
        walks = np.empty((len(starts), length), dtype=np.int64)
        walks[:, 0] = cur = starts
        prev = None
        for step in range(1, length):
            nxt = cur.copy()
            todo = np.where(self.deg[cur] > 0)[0]
            if prev is None or not self.biased:
                nxt[todo] = self.sample_neighbor(cur[todo])
            else:
                for attempt in range(self.max_tries):
                    if len(todo) == 0:
                        break
                    cand = self.sample_neighbor(cur[todo])
                    t = prev[todo]
                    prob = np.where(cand == t, 1.0 / self.p,
                                    np.where(self.is_edge(t, cand), 1.0, 1.0 / self.q)) / self.max_prob
                    acc = self.rng.random(len(todo)) < prob
                    if attempt == self.max_tries - 1:
                        acc[:] = True
                    nxt[todo[acc]] = cand[acc]
                    todo = todo[~acc]
            walks[:, step] = nxt
            prev, cur = cur, nxt
        return walks


def skipgram_pairs(walks, window):
    """(center, context) pairs within `window` positions, both directions, self-pairs removed."""
    c, o = [], []
    for k in range(1, window + 1):
        a, b = walks[:, :-k].ravel(), walks[:, k:].ravel()
        c += [a, b]; o += [b, a]
    c, o = np.concatenate(c), np.concatenate(o)
    keep = c != o
    return c[keep], o[keep]


def embed_node2vec(args, A):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch_device(args.device)
    walker = Node2VecWalker(A, p=args.p, q=args.q, weighted=not args.n2v_unweighted, seed=args.seed)
    n = walker.n
    log(f"  node2vec on {dev}: {n:,} nodes, p={args.p}, q={args.q}, "
        f"{args.walks_per_node} walks/node x {args.walk_length} steps, window {args.window}, "
        f"{'unweighted' if args.n2v_unweighted else 'weighted'}")

    emb = torch.nn.Embedding(n, args.dim, sparse=True).to(dev)
    ctx = torch.nn.Embedding(n, args.dim, sparse=True).to(dev)
    torch.nn.init.uniform_(emb.weight, -0.5 / args.dim, 0.5 / args.dim)
    torch.nn.init.zeros_(ctx.weight)
    opt = torch.optim.SparseAdam(list(emb.parameters()) + list(ctx.parameters()), lr=args.n2v_lr)

    # negatives drawn from the usual unigram^0.75 distribution (walk frequency ~ degree)
    noise = torch.from_numpy((np.maximum(walker.deg, 1) ** 0.75).astype(np.float64))
    noise = (noise / noise.sum()).float().to(dev)

    t0, step, last_loss = time.time(), 0, float("nan")
    for epoch in range(1, args.n2v_epochs + 1):
        starts = rng.permutation(np.repeat(np.arange(n), args.walks_per_node))
        for b in range(0, len(starts), args.walk_batch):
            walks = walker.walk(starts[b:b + args.walk_batch], args.walk_length)
            c, o = skipgram_pairs(walks, args.window)
            perm = rng.permutation(len(c))
            c, o = torch.from_numpy(c[perm]).to(dev), torch.from_numpy(o[perm]).to(dev)
            for j in range(0, len(c), args.pair_batch):
                cb, ob = c[j:j + args.pair_batch], o[j:j + args.pair_batch]
                neg = torch.multinomial(noise, len(cb) * args.negatives, replacement=True).view(len(cb), -1)
                ec = emb(cb)                                                   # [B, d]
                pos = (ec * ctx(ob)).sum(-1)                                   # [B]
                negs = torch.bmm(ctx(neg), ec.unsqueeze(-1)).squeeze(-1)       # [B, k]
                loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-negs).sum(-1).mean())
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1; last_loss = loss.item()
            done_walks = min(b + args.walk_batch, len(starts))
            log(f"    epoch {epoch} walks {done_walks:,}/{len(starts):,}  loss {last_loss:.4f}  "
                f"({time.time() - t0:.0f}s)")
    with torch.no_grad():
        Z = emb.weight.detach().cpu().numpy().astype(np.float32)
    return Z, {"final_loss": last_loss, "steps": step}


# ---------------------------------------------------------------------------
# GNN (GCN graph autoencoder)
# ---------------------------------------------------------------------------

def embed_gnn(args, A):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch_device(args.device)
    X = carrier_features(args)
    n = A.shape[0]
    An = normalized_adjacency(A).tocoo()
    A_t = torch.sparse_coo_tensor(np.vstack([An.row, An.col]), An.data, (n, n)).coalesce().to(dev)
    Xt = torch.from_numpy(X).to(dev)
    up = sp.triu(A, k=1).tocoo()
    pos = np.vstack([up.row, up.col]).astype(np.int64)
    if pos.shape[1] == 0:
        die("graph has no edges; rebuild it with a lower --min-lift")

    class GCN(nn.Module):
        def __init__(s, d_in, d_hid, d_out):
            super().__init__()
            s.l1, s.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_out)

        def forward(s, X, A):
            H = F.dropout(F.relu(torch.sparse.mm(A, s.l1(X))), 0.1, s.training)
            return torch.sparse.mm(A, s.l2(H))

    model = GCN(X.shape[1], args.gnn_hidden, args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)
    batch = min(args.edge_batch, pos.shape[1])
    log(f"  GCN on {dev}: {n:,} nodes, {pos.shape[1]:,} edges, {X.shape[1]} input features")
    acc = float("nan")
    for ep in range(1, args.gnn_epochs + 1):
        model.train(); opt.zero_grad()
        Z = model(Xt, A_t)
        p = torch.from_numpy(pos[:, rng.choice(pos.shape[1], batch, replace=False)]).to(dev)
        q = torch.from_numpy(rng.integers(0, n, (2, batch))).to(dev)
        pl, nl = (Z[p[0]] * Z[p[1]]).sum(-1), (Z[q[0]] * Z[q[1]]).sum(-1)
        loss = (F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl)))
        loss.backward(); opt.step()
        if ep == 1 or ep % 50 == 0 or ep == args.gnn_epochs:
            acc = (((pl > 0).float().mean() + (nl < 0).float().mean()) / 2).item()
            log(f"    epoch {ep:4d}  loss {loss.item():.4f}  link acc {acc:.3f}")
    model.eval()
    with torch.no_grad():
        Z = model(Xt, A_t).cpu().numpy().astype(np.float32)
    return Z, {"final_loss": loss.item(), "link_accuracy": acc}


def embed_carrier_svd(args, A):
    X = carrier_features(args)
    return X[:, :args.dim].copy(), {"svd_dim": int(X.shape[1])}


# ---------------------------------------------------------------------------
# XGBoost (self-supervised leaf embeddings: predict the cis partner in the next haploblock)
# ---------------------------------------------------------------------------

def propagate(A, X):
    """D^-1/2 (A + I) D^-1/2 @ X (GCN-normalized neighbourhood average) without copying A."""
    d = np.asarray(A.sum(axis=1)).ravel() + 1.0
    s = (1.0 / np.sqrt(d)).astype(np.float32)[:, None]
    Y = s * X
    return s * (A @ Y + Y)


def leaf_onehot(leaves, width):
    """[n, n_trees] leaf ids -> sparse one-hot [n, n_trees * width] (tree j uses columns j*width...)."""
    n, n_trees = leaves.shape
    cols = (np.arange(n_trees, dtype=np.int64) * width)[None, :] + leaves
    rows = np.repeat(np.arange(n, dtype=np.int64), n_trees)
    return sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols.ravel())),
                         shape=(n, n_trees * width))


def cis_partner_class(args, nodes, n_classes):
    """Per-node class = frequency rank (by training carriers, 0 = most frequent) within the next
    haploblock of the cluster that most often follows the node on the same training haplotype
    (junctions.npz); ranks >= n_classes - 1 share the last class. Without haplotype data, the
    next-haploblock cluster most often carried by the same training individuals stands in. -1 for
    nodes with no training partner (last haploblock of a chromosome, never seen in training):
    they are not trained on, but still embedded. This target is about linkage, which the genomic
    position encoding of step 4 does not already carry."""
    freq = nodes["support_train"].to_numpy() if "support_train" in nodes else nodes["support"].to_numpy()
    rank = (pd.DataFrame({"r": nodes["region"], "f": -freq}).groupby("r")["f"]
            .rank(method="first").to_numpy().astype(np.int64) - 1)
    best = np.full(len(nodes), -1, np.int64)
    jf = gfile(args, "junctions.npz")
    if os.path.exists(jf):
        j = np.load(jf)
        if len(j["a"]):
            srt = np.lexsort((-j["count"], j["a"]))               # per a: largest count first
            first = np.r_[True, j["a"][srt][1:] != j["a"][srt][:-1]]
            best[j["a"][srt][first]] = j["b"][srt][first]
        source = "same training haplotype (junctions.npz)"
    else:
        C = sp.load_npz(gfile(args, "carriers.npz")).tocsc()[:, args.train_idx].tocsr().astype(np.float32)
        regs = (nodes.assign(ck=nodes["chrom"].map(lambda c: chrom_key(str(c))))
                .groupby("region", sort=False).agg(ck=("ck", "first"), start=("start", "first"), chrom=("chrom", "first"))
                .sort_values(["ck", "start"]).reset_index())
        members = nodes.groupby("region")["node"].apply(np.asarray)
        for (r0, c0), (r1, c1) in zip(regs[["region", "chrom"]].values[:-1], regs[["region", "chrom"]].values[1:]):
            if c0 != c1:
                continue
            a, b = members[r0], members[r1]
            M = (C[a] @ C[b].T).toarray()
            ok = M.max(1) > 0
            best[a[ok]] = b[M[ok].argmax(1)]
        source = "same training individuals (carriers; no haplotype data)"
    y = np.where(best >= 0, np.minimum(rank[np.maximum(best, 0)], n_classes - 1), -1)
    y[freq == 0] = -1
    return y, source


def embed_xgboost(args, A):
    import xgboost as xgb
    rng = np.random.default_rng(args.seed)
    nodes = pd.read_csv(gfile(args, "nodes.csv.gz"), usecols=["node", "chrom", "region", "start"])
    if len(nodes) != A.shape[0]:
        die(f"nodes.csv.gz has {len(nodes)} rows but graph.npz has {A.shape[0]} nodes; rebuild with step 2")
    nodes = nodes.join(pd.read_csv(gfile(args, "nodes.csv.gz"), usecols=lambda c: c.startswith("support")))
    y, source = cis_partner_class(args, nodes, args.xgb_classes)
    labelled = np.where(y >= 0)[0]
    n_cls = int(y.max()) + 1 if len(labelled) else 0
    if n_cls < 2:
        die("xgboost needs nodes with at least 2 different cis-partner classes; use --method gnn / carrier_svd")
    X = carrier_features(args)
    H = propagate(A, X)                                   # graph neighbourhood average
    n = len(X)

    def feats(idx):                                       # own profile + neighbourhood, built per batch
        return np.hstack([X[idx], H[idx]])

    order = labelled[rng.permutation(len(labelled))]
    n_val = min(max(1, int(0.1 * len(order))), args.xgb_val_nodes)
    val, tr = order[:n_val], order[n_val:n_val + args.xgb_train_nodes]
    if len(tr) == 0:
        die("too few nodes to train xgboost")

    device = "cuda" if torch_device(args.device).type == "cuda" else "cpu"
    params = {
        "objective": "multi:softprob", "num_class": n_cls, "eval_metric": "mlogloss",
        "max_depth": args.xgb_depth, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "max_bin": args.xgb_max_bin, "device": device, "seed": args.seed,
    }
    log(f"  XGBoost on {device}: {len(tr):,} train / {len(val):,} val nodes ({n:,} total, {len(labelled):,} with a "
        f"partner), {2 * X.shape[1]} features, target = cis partner's frequency rank in the next haploblock "
        f"({n_cls} classes; partner from {source}; {n_cls * args.xgb_trees:,} trees)")
    dtrain = xgb.QuantileDMatrix(feats(tr), label=y[tr], max_bin=args.xgb_max_bin)
    dval = xgb.QuantileDMatrix(feats(val), label=y[val], ref=dtrain, max_bin=args.xgb_max_bin)
    booster = xgb.train(params, dtrain, num_boost_round=args.xgb_trees,
                        evals=[(dval, "val")], verbose_eval=False)
    acc = float((np.asarray(booster.predict(dval)).reshape(len(val), -1).argmax(1) == y[val]).mean())
    base = float(np.bincount(y[tr], minlength=n_cls).max() / len(tr))
    log(f"  cis-partner accuracy on held-out nodes {acc:.3f} (majority-class baseline {base:.3f})")
    del dtrain, dval
    # leaf prediction below runs on CPU numpy input; a cuda booster would take XGBoost's slow
    # device-mismatch fallback (and can exhaust host RAM at genome scale)
    if device == "cuda":
        booster.set_param({"device": "cpu"})

    width = 2 ** (args.xgb_depth + 1)                     # depth-d tree node ids are < 2^(d+1) - 1

    def leaves(idx):
        L = np.asarray(booster.predict(xgb.DMatrix(feats(idx)), pred_leaf=True)).astype(np.int64)
        L = L.reshape(len(idx), -1)
        if L.size and L.max() >= width:
            raise RuntimeError(f"leaf id {L.max()} >= {width}; trees are deeper than --xgb-depth")
        return L

    sample = np.sort(rng.choice(n, min(n, args.xgb_svd_nodes), replace=False))
    M = leaf_onehot(leaves(sample), width)
    k = min(args.dim, min(M.shape) - 1)
    V = svd_basis(M, k, args.seed, args.rapids)
    log(f"  leaf SVD fitted on {len(sample):,} nodes ({V.shape[0]:,} leaf columns -> {V.shape[1]} dims)")
    Z = np.empty((n, V.shape[1]), np.float32)
    for b in range(0, n, args.chunk_nodes):
        idx = np.arange(b, min(b + args.chunk_nodes, n))
        Z[idx] = leaf_onehot(leaves(idx), width) @ V
        if (b // args.chunk_nodes) % 20 == 0:
            log(f"    embedded {idx[-1] + 1:,}/{n:,} nodes")
    return Z, {"target": "cis_partner_rank", "partner_source": source, "n_classes": n_cls, "val_accuracy": acc,
               "baseline_accuracy": base, "train_nodes": int(len(tr)), "svd_nodes": int(len(sample)),
               "n_trees_total": int(V.shape[0] // width), "rapids": bool(args.rapids)}


# ---------------------------------------------------------------------------
# evo2_seq: the backbone on each cluster's representative haplotype sequence
# ---------------------------------------------------------------------------

def read_fasta(text):
    name, seq = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq)
            name, seq = line[1:].split()[0], []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def embed_evo2_seq(args, A):
    """See the module docstring. Reads <raw-dir>/<chrom>/rep_seq.tar.gz (one FASTA per haploblock,
    one record per cluster named after its representative haplotype) and checks every member
    against rep_seq.md5."""
    import torch
    from evo2_model import make_runner
    nodes = pd.read_csv(gfile(args, "nodes.csv.gz"), usecols=["node", "chrom", "region"])
    hf = gfile(args, "haplotypes.npz")
    if not os.path.exists(hf):
        die("evo2_seq needs haplotypes.npz (step 1 --haplotypes, then step 2) to know each sequence's cluster")
    with np.load(hf) as h:                                          # read once: the file is compressed
        Hn, blocks = h["node"], h["blocks"]
    col = {r: j for j, r in enumerate(blocks)}
    row = {x: i for i, x in enumerate(read_graph_individuals(args.graph_dir))}
    raw = args.raw_dir or os.path.join(os.path.dirname(os.path.abspath(args.graph_dir)), "raw")
    key = {"graph_id": graph_id(args.graph_dir)} | {k: getattr(args, k) for k in ("backbone", "bionemo", "init", "seed", "evo2_model", "layer", "ckpt_dir",
                                         "model_size", "layer_index", "small_hidden", "small_layers", "seq_window",
                                         "seq_windows")}
    cache = os.path.join(gfile(args, "evo2_seq_cache"), hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8])
    os.makedirs(cache, exist_ok=True)
    rnr, V, t0, disk = None, {}, time.time(), 0
    for chrom in sorted(nodes["chrom"].unique(), key=chrom_key):
        arc = os.path.join(raw, chrom, "rep_seq.tar.gz")
        if not os.path.exists(arc):
            die(f"{arc} not found - run 1_download_haplograph.py --chroms {chrom} --sequences")
        disk += os.path.getsize(arc)
        sums = dict(line.split()[::-1] for line in open(os.path.join(raw, chrom, "rep_seq.md5")) if line.strip())
        with tarfile.open(arc, "r|gz") as tar:                       # streamed: one member at a time
            for m in tar:
                if not m.name.endswith("_rep_seq.fasta"):
                    continue
                region = os.path.basename(m.name)[:-len("_rep_seq.fasta")]
                f = os.path.join(cache, f"{region}.npz")
                if os.path.exists(f):
                    with np.load(f) as z:
                        V.update(zip(*z.values()))
                    continue
                data = tar.extractfile(m).read()
                if sums.get(m.name) not in (None, hashlib.md5(data).hexdigest()):
                    die(f"{arc}: {m.name} does not match rep_seq.md5; delete the archive and download it again")
                rnr = rnr or make_runner(argparse.Namespace(**vars(args)), 0)
                ids, vecs = [], []
                for name, seq in read_fasta(data.decode()):
                    sample, hap = name.split("_")[0], int(name[-1])
                    if sample not in row or region not in col:
                        continue
                    seq = seq.upper().encode()
                    L = min(len(seq), args.seq_window)
                    starts = np.linspace(0, len(seq) - L, args.seq_windows).astype(int) if len(seq) > L else [0]
                    with torch.no_grad():
                        v = torch.stack([rnr.run_ids(torch.frombuffer(bytearray(seq[s:s + L]), dtype=torch.uint8))
                                         .mean(0) for s in starts]).mean(0)
                    ids.append(int(Hn[2 * row[sample] + hap, col[region]]))
                    vecs.append(v.cpu().numpy())
                np.savez(f, node=np.array(ids, np.int64), vec=np.array(vecs, np.float32))
                V.update(zip(ids, vecs))
        log(f"    {chrom}: {len(V):,} clusters embedded so far ({time.time() - t0:.0f}s)")
    if not V:
        die("no representative sequences matched the graph's clusters")
    dim = len(next(iter(V.values())))
    M = np.zeros((len(nodes), dim), np.float32)
    have = np.zeros(len(nodes), bool)
    for k, v in V.items():
        M[k], have[k] = v, True
    blk = pd.factorize(nodes["region"])[0]
    mean = np.zeros((blk.max() + 1, dim)); np.add.at(mean, blk[have], M[have])
    mean /= np.maximum(np.bincount(blk[have], minlength=len(mean)), 1)[:, None]
    M = np.where(have[:, None], M - mean[blk], 0).astype(np.float32)        # within-haploblock differences
    U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False) if len(M) < 50_000 else (None, None, None)
    if U is None:
        U, S, _ = svds(sp.csr_matrix(M - M.mean(0)), k=min(args.dim, dim - 1), random_state=args.seed)
        o = np.argsort(S)[::-1]; U, S = U[:, o], S[o]
    Z = (U[:, :args.dim] * S[:args.dim]).astype(np.float32)
    return Z, {"backbone": args.backbone, "clusters_with_sequence": int(have.sum()), "clusters": len(nodes),
               "seq_window": args.seq_window, "seq_windows": args.seq_windows, "archive_bytes": int(disk),
               "seconds": round(time.time() - t0)}


# ---------------------------------------------------------------------------

NOT_RESULT_SETTINGS = ("force", "csv", "train_idx", "test", "device", "method", "graph_dir", "train_src",
                       "train_individuals", "sample_map", "graph_id", "train_id")


def settings_id(args):
    """Id of every setting that can change the embeddings (not where or on which device they run)."""
    keep = {k: v for k, v in vars(args).items() if k not in NOT_RESULT_SETTINGS}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_method(args, method, A):
    out = gfile(args, f"node_emb_{method}.npy")
    info_file = gfile(args, f"node_emb_{method}.json")
    current = {"train_set_id": args.train_id, "graph_id": args.graph_id, "settings_id": settings_id(args)}
    if os.path.exists(out) and not args.force:
        old = json.load(open(info_file)) if os.path.exists(info_file) else {}
        if all(old.get(k) == v for k, v in current.items()):
            log(f"{out} already exists for this graph, training set and settings (use --force to recompute)")
            return
        log(f"existing {os.path.basename(out)} was made from another graph build, training set or settings; "
            "recomputing")
    t0 = time.time()
    log(f"[{method}] graph {args.graph_dir}: {A.shape[0]:,} nodes, {A.nnz // 2:,} edges; "
        f"training individuals: {len(args.train_idx):,} ({args.train_src})")
    fn = {"node2vec": embed_node2vec, "gnn": embed_gnn, "xgboost": embed_xgboost,
          "carrier_svd": embed_carrier_svd, "evo2_seq": embed_evo2_seq}[method]
    Z, summary = fn(args, A)
    if not np.isfinite(Z).all():
        die(f"{method} produced non-finite embeddings; try other settings (e.g. a lower learning rate)")

    np.save(out, Z)
    settings = {k: v for k, v in vars(args).items() if k not in ("force", "csv", "train_idx", "test")}
    with open(info_file, "w") as f:
        json.dump({"method": method, "shape": list(Z.shape), "settings": settings,
                   "train_source": args.train_src, "n_train": int(len(args.train_idx)),
                   **current, "summary": summary, "seconds": round(time.time() - t0)}, f, indent=2)
    if args.csv:
        nodes = pd.read_csv(gfile(args, "nodes.csv.gz"), usecols=["id", "chrom"])
        df = pd.DataFrame(Z, columns=[f"e{j}" for j in range(Z.shape[1])])
        pd.concat([nodes, df], axis=1).to_csv(gfile(args, f"node_emb_{method}.csv.gz"), index=False)
    log(f"  saved {Z.shape} embeddings to {out}")


def self_test_settings(args):
    """Build the synthetic test graph (2_build_graph.py --test) and shrink every method."""
    import importlib
    importlib.import_module("2_build_graph").main(["--test"])
    args.graph_dir = selftest_graph_dir()
    args.dim, args.svd_dim, args.walks_per_node, args.walk_length = 8, 8, 2, 10
    args.gnn_epochs, args.gnn_hidden, args.xgb_trees, args.xgb_depth = 20, 16, 5, 3
    args.xgb_classes, args.force, args.device = 4, True, "cpu"
    args.backbone, args.small_hidden, args.small_layers, args.seq_window = "small", 16, 1, 64


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all",
                    help="node2vec, gnn, xgboost, carrier_svd, evo2_seq, a comma-separated list, or all (default: "
                         "node2vec, gnn, carrier_svd)")
    ap.add_argument("--graph-dir", default="haplograph/genome_train_half_A", help="output folder of 2_build_graph.py")
    ap.add_argument("--dim", type=int, default=64, help="embedding size")
    ap.add_argument("--csv", action="store_true", help="also write node_emb_<method>.csv.gz")
    ap.add_argument("--device", help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute existing embeddings")
    ap.add_argument("--train-individuals",
                    help="file or URL of training individuals (default: <graph-dir>/train_individuals.txt, else all)")
    ap.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--test", action="store_true", help="self-test on a small synthetic graph (CPU, seconds)")

    n = ap.add_argument_group("node2vec")
    n.add_argument("--p", type=float, default=1.0, help="return parameter (low = walks stay local)")
    n.add_argument("--q", type=float, default=1.0, help="in-out parameter (low = walks explore outward)")
    n.add_argument("--walk-length", type=int, default=40)
    n.add_argument("--walks-per-node", type=int, default=10)
    n.add_argument("--window", type=int, default=5)
    n.add_argument("--negatives", type=int, default=5)
    n.add_argument("--n2v-epochs", type=int, default=1)
    n.add_argument("--n2v-lr", type=float, default=0.01)
    n.add_argument("--walk-batch", type=int, default=20_000, help="walks generated at a time")
    n.add_argument("--pair-batch", type=int, default=65_536, help="skip-gram pairs per step")
    n.add_argument("--n2v-unweighted", action="store_true", help="ignore edge weights in walks")

    g = ap.add_argument_group("gnn / xgboost features")
    g.add_argument("--svd-dim", type=int, default=128, help="carrier-profile features after SVD")
    g.add_argument("--gnn-hidden", type=int, default=256)
    g.add_argument("--gnn-epochs", type=int, default=300)
    g.add_argument("--gnn-lr", type=float, default=0.01)
    g.add_argument("--edge-batch", type=int, default=500_000)
    g.add_argument("--xgb-trees", type=int, default=50, help="boosting rounds (one tree per class per round)")
    g.add_argument("--xgb-depth", type=int, default=6)
    g.add_argument("--xgb-classes", type=int, default=8,
                   help="cis-partner frequency ranks 0 .. classes-2, then one class for every lower rank")
    g.add_argument("--xgb-train-nodes", type=int, default=1_000_000, help="max nodes to train on")
    g.add_argument("--xgb-val-nodes", type=int, default=100_000, help="max held-out nodes for accuracy")
    g.add_argument("--xgb-svd-nodes", type=int, default=300_000, help="nodes sampled to fit the leaf SVD")
    g.add_argument("--xgb-max-bin", type=int, default=256, help="histogram bins (lower = less memory)")
    g.add_argument("--chunk-nodes", type=int, default=200_000, help="nodes embedded per chunk")

    s = ap.add_argument_group("evo2_seq")
    s.add_argument("--raw-dir", help="folder with <chrom>/rep_seq.tar.gz from step 1 (default <graph-dir>/../raw)")
    s.add_argument("--seq-window", type=int, default=8192, help="bp per backbone pass (Evo 2 7B base: 8,192 context)")
    s.add_argument("--seq-windows", type=int, default=1, help="evenly spaced windows per sequence, averaged")
    from evo2_model import add_backbone_args
    add_backbone_args(ap, device=False)

    r = ap.add_argument_group("RAPIDS (GPU SVD)")
    r.add_argument("--rapids", action="store_true", help="run the carrier and leaf SVDs on the GPU with cuML/CuPy")
    r.add_argument("--rapids-fit-rows", type=int, default=150_000,
                   help="rows sampled to fit the cuML carrier SVD (higher = closer to exact)")
    r.add_argument("--rapids-chunk", type=int, default=200_000,
                   help="rows densified per GPU transform chunk (lower = less VRAM)")
    args = ap.parse_args(argv)

    if args.test:
        self_test_settings(args)
        args.method = ",".join(METHODS) if args.method == "all" else args.method
    methods = list(DEFAULT_METHODS) if args.method == "all" else [m.strip() for m in args.method.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad or not methods:
        die(f"unknown --method {bad or args.method!r}; choose from {', '.join(METHODS)} or all")
    if args.rapids:
        import_rapids()
    check_args(args, ("dim", "svd_dim", "xgb_classes", "chunk_nodes", "walk_batch", "seq_window", "seq_windows"))
    if args.xgb_classes < 2:
        die("--xgb-classes must be >= 2")

    require_files(args.graph_dir, ["graph.npz", "individuals.txt"]
                  + (["carriers.npz"] if {"gnn", "xgboost", "carrier_svd"} & set(methods) else [])
                  + (["nodes.csv.gz"] if {"xgboost", "evo2_seq"} & set(methods) or args.csv else []),
                  "2_build_graph.py")
    individuals = read_graph_individuals(args.graph_dir)
    args.train_idx, args.train_src, args.train_id = train_columns(args, individuals)
    args.graph_id = graph_id(args.graph_dir)
    A = sp.load_npz(gfile(args, "graph.npz")).astype(np.float32).tocsr()
    for method in methods:
        run_method(args, method, A)
    log("done.")


if __name__ == "__main__":
    main()
