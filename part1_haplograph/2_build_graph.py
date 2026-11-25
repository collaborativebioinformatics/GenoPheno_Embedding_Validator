"""
Step 2 - build the haplograph co-occurrence graph, genome-wide or for --chroms, from step 1's
downloads. Everything learned from data is counted on the TRAINING individuals only (step 1's split,
or --train-individuals; --no-split = all individuals), so the test set never shapes the graph.
Two kinds of edge (--lift):
  haplotype   (default when step 1 ran with --haplotypes) two clusters are linked when they sit on
              the same training haplotype, within --max-span bp of each other:
              lift = n_AB * H / (n_A * n_B) over the H training haplotypes; candidate pairs come from
              the training haplotypes themselves, pairs seen on fewer than --min-pair-count are dropped
  individual  two clusters are linked when the same training individuals carry them (either
              haplotype), for the pairs listed in the dataset's edge file:
              lift = n_AB / (support_A * support_B / n_train)
Edges with lift < --min-lift are dropped; the edge value is log(1 + lift).

With step 1's --haplotypes data, the graph folder also gets each haplotype's cluster per haploblock
(checked: each individual's two haplotypes must reproduce its carriers exactly, or the build stops)
and the training haplotypes' transitions between adjacent haploblocks.

OUTPUT (in --out; default <workdir>/genome or <workdir>/<chrA>_<chrB>, + _train_<split name>)
  nodes.csv.gz      row i = node i: node, id, chrom, region, start, end, support, support_train
  individuals.txt   one individual per line = the columns of carriers.npz
  train_individuals.txt / test_individuals.txt   the split, restricted to the graph's individuals
  populations.tsv, related_pairs.tsv             copied from step 1's splits/ (step 6 stratifies by
                                                 population and reports related pairs across the split)
  carriers.npz      scipy.sparse CSR uint8 [nodes, individuals], ALL individuals (step 6 needs the test ones)
  graph.npz         scipy.sparse CSR float32 [nodes, nodes], symmetric, value log(1 + lift);
                    within-chromosome edges only
  haplotypes.npz    node: int32 [2 x individuals, haploblocks], row 2i / 2i+1 = haplotype 0 / 1 of
                    individual i of individuals.txt, value = node index; blocks: haploblock regions
                    (columns, genomic order per chromosome); block_chrom; ALL individuals
  junctions.npz     block, a, b, count: count = training haplotypes with node a in haploblock column
                    `block` and node b in the next haploblock of the same chromosome
  graph_info.json   settings, counts, training/test set ids (rerunning with other settings stops
                    unless --force)
Parsed chromosomes are cached in <workdir>/parts/<chrom>/, keyed by the raw files.

USAGE
  python 2_build_graph.py                       # all downloaded chromosomes -> haplograph/genome_train_half_A
  python 2_build_graph.py --chroms chr21,chr22  # -> haplograph/chr21_chr22_train_half_A
  python 2_build_graph.py --lift individual     # co-carriage lift from the dataset's edge file
  python 2_build_graph.py --cudf --chunksize 0  # pandas parts on the GPU (RAPIDS cuDF)
  python 2_build_graph.py --test                # synthetic data, CPU, seconds
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

if "--cudf" in sys.argv:
    # GPU dataframes: cuDF's pandas accelerator must be switched on BEFORE pandas is imported.
    # pandas code then runs on the GPU where cuDF supports it and falls back to the CPU elsewhere.
    try:
        import cudf.pandas
        cudf.pandas.install()
    except ImportError:
        sys.exit("--cudf needs RAPIDS cuDF (pip install cudf-cu12); run without --cudf to use plain pandas")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from haplo_common import (ALL_CHROMS, EDGE_FILE, REGION_RE, chrom_key, die, load_sample_map, log,
                          parse_chroms, read_individual_list, set_id,
                          test_workdir)

OUTPUTS = ("nodes.csv.gz", "individuals.txt", "carriers.npz", "graph.npz", "graph_info.json")
HAP_OUTPUTS = ("haplotypes.npz", "junctions.npz")


def file_stamp(path):
    """Short id of a raw file (absolute path, size, modification time): caches built from another raw
    folder or an older download are never reused."""
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:8]


def raw_file(args, chrom, name):
    p = Path(args.raw_dir) / chrom / name
    if not p.exists():
        die(f"{p} not found - download it first: python 1_download_haplograph.py --chroms {chrom}"
            + (" --edges lift5" if name.startswith("edges_lift") else "")
            + (f" --workdir {args.workdir}" if args.workdir != "haplograph" else ""))
    return str(p)


# ---------------------------------------------------------------------------
# Train-only co-occurrence: bit-packed carriers
# ---------------------------------------------------------------------------

def pack_rows(C, rows_per_block=50_000):
    """Sparse 0/1 [N, k] -> bit-packed uint64 [N, ceil(k/64)] (one bit per individual)."""
    blocks = []
    for b in range(0, C.shape[0], rows_per_block):
        P = np.packbits(C[b:b + rows_per_block].toarray().astype(bool), axis=1)
        pad = (-P.shape[1]) % 8
        if pad:
            P = np.pad(P, ((0, 0), (0, pad)))
        blocks.append(np.ascontiguousarray(P).view(np.uint64))
    return np.vstack(blocks) if blocks else np.zeros((0, 1), np.uint64)


if hasattr(np, "bitwise_count"):
    def popcount_rows(x):
        return np.bitwise_count(x).sum(axis=1, dtype=np.int64)
else:
    _TABLE = np.array([bin(i).count("1") for i in range(256)], np.uint8)

    def popcount_rows(x):
        return _TABLE[np.ascontiguousarray(x).view(np.uint8)].sum(axis=1, dtype=np.int64)


def shared_carriers(packed, s, t, block=1_000_000):
    """Number of individuals (bits) carrying both s and t, for each pair."""
    out = np.empty(len(s), np.float64)
    for b in range(0, len(s), block):
        out[b:b + block] = popcount_rows(packed[s[b:b + block]] & packed[t[b:b + block]])
    return out


# ---------------------------------------------------------------------------
# Parse one chromosome
# ---------------------------------------------------------------------------

def read_nodes(nodes_path, chunksize=20_000):
    header = pd.read_csv(nodes_path, nrows=0).columns.tolist()
    if header[:2] != ["id", "high_dim_edge"]:
        die(f"{nodes_path}: expected columns id, high_dim_edge, <individuals>; got {header[:3]}")
    inds = [c for c in header if c not in ("id", "high_dim_edge")]
    dtypes = {c: np.uint8 for c in inds}
    dtypes.update({"id": str, "high_dim_edge": str})
    ids, regions, blocks = [], [], []
    try:
        for chunk in pd.read_csv(nodes_path, dtype=dtypes, chunksize=chunksize):
            ids.append(chunk["id"].to_numpy())
            regions.append(chunk["high_dim_edge"].to_numpy())
            blocks.append(sp.csr_matrix(chunk[inds].to_numpy(dtype=np.uint8)))
    except (EOFError, OSError, ValueError) as e:
        die(f"{nodes_path} is truncated or malformed ({type(e).__name__}: {e}); delete it and download it again")
    if any(b.nnz and b.data.max() > 1 for b in blocks):
        die(f"{nodes_path}: carrier values must be 0 or 1")
    ids = np.concatenate(ids)
    if len(pd.unique(ids)) != len(ids):
        die(f"{nodes_path}: duplicate cluster ids")
    return ids, np.concatenate(regions), sp.vstack(blocks).tocsr(), inds


def read_edges(edges_path, ids, support, n_ind, min_lift, chunksize, packed=None):
    """Stream an edges file, map ids -> row indices, compute lift, keep lift >= min_lift.
    With `packed` (bit-packed training carriers), the weight is recomputed from training individuals."""
    index = pd.Index(ids)
    S, T, L = [], [], []
    n_read = n_unknown = 0
    reader = pd.read_csv(edges_path, usecols=["source", "target", "weight"],
                         dtype={"source": str, "target": str, "weight": np.float64},
                         chunksize=chunksize or None)          # 0 = whole file at once
    for chunk in (reader if chunksize else [reader]):
        s = index.get_indexer(chunk["source"])
        t = index.get_indexer(chunk["target"])
        w = chunk["weight"].to_numpy()
        n_read += len(chunk)
        n_unknown += int(((s < 0) | (t < 0)).sum())
        ok = (s >= 0) & (t >= 0) & (s != t)
        s, t, w = s[ok], t[ok], w[ok]
        if packed is not None:
            w = shared_carriers(packed, s, t)
        expected = support[s].astype(np.float64) * support[t] / n_ind
        lift = np.divide(w, expected, out=np.zeros_like(w), where=expected > 0)
        keep = (lift >= min_lift) & (w > 0)
        S.append(s[keep].astype(np.int64)); T.append(t[keep].astype(np.int64))
        L.append(lift[keep].astype(np.float32))
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt)
    return cat(S, np.int64), cat(T, np.int64), cat(L, np.float32), n_read, n_unknown


def read_haplotypes(path, ids, regions, carriers, inds):
    """raw/<chrom>/haplotypes.csv.gz -> (int32 [2 x individuals, haploblocks] node index within the
    chromosome, haploblock regions in column order). Stops unless every individual's two haplotypes
    reproduce its carriers exactly."""
    tab = pd.read_csv(path, index_col=0)
    rows = [f"{x}_hap{h}" for x in inds for h in (0, 1)]
    missing = sorted(set(rows) - set(tab.index))
    if missing:
        die(f"{path}: {len(missing)} haplotypes of nodes.csv.gz individuals are missing (e.g. {missing[:2]})")
    tab = tab.loc[rows]
    index = pd.Index(ids)
    H = np.empty(tab.shape, np.int32)
    for j, region in enumerate(tab.columns):
        H[:, j] = index.get_indexer(region + "_cluster" + tab[region].astype(str))
    if (H < 0).any():
        die(f"{path}: {int((H < 0).sum())} haplotype clusters are not nodes of nodes.csv.gz; re-run step 1")
    ind = np.repeat(np.arange(len(inds)), 2)
    U = sp.csr_matrix((np.ones(H.size, np.uint8), (H.ravel(), np.repeat(ind, H.shape[1]))),
                      shape=carriers.shape)
    U.data[:] = 1                                             # homozygous: both haplotypes, one carrier bit
    diff = (U != carriers.astype(bool)).tocoo()
    if diff.nnz:
        bad = sorted(set(regions[diff.row]))
        die(f"{path}: the haplotypes do not reproduce the carriers of nodes.csv.gz in {len(bad)} haploblock(s) "
            f"(e.g. {bad[:3]}); the two downloads disagree - delete raw/<chrom>/haplotypes.csv.gz and re-run step 1")
    return H, list(tab.columns)


def haplotype_edges(H, starts, hap_rows, n_nodes, args):
    """Same-haplotype co-occurrence over the training haplotypes `hap_rows` of H, for cluster pairs in
    different haploblocks at most --max-span bp apart (start to start). Returns s, t, lift, n_pairs."""
    Ht = H[hap_rows]
    n_A = np.bincount(Ht.ravel(), minlength=n_nodes).astype(np.float64)
    counts = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    for d in range(1, H.shape[1]):
        k = np.where(starts[d:] - starts[:-d] <= args.max_span)[0]
        if not len(k):
            break
        a, b = Ht[:, k].ravel(), Ht[:, k + d].ravel()
        counts = counts + sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n_nodes, n_nodes))
    c = counts.tocoo()
    lift = c.data * len(hap_rows) / (n_A[c.row] * n_A[c.col])
    keep = (c.data >= args.min_pair_count) & (lift >= args.min_lift)
    return c.row[keep].astype(np.int64), c.col[keep].astype(np.int64), lift[keep].astype(np.float32), c.nnz


def chrom_part(args, chrom, train_names=None, train_tag=""):
    """Parse one chromosome; cached in <workdir>/parts/<chrom>/.
    Returns ids, regions, carriers (all individuals), individual names, edges, lift, train support,
    haplotypes (node index within the chromosome, or None), haploblock regions of its columns, status."""
    d = os.path.join(args.workdir, "parts", chrom)
    os.makedirs(d, exist_ok=True)
    nodes_raw = raw_file(args, chrom, "nodes.csv.gz")
    hap_raw = Path(args.raw_dir) / chrom / "haplotypes.csv.gz"
    if args.lift == "haplotype":
        hap_raw = Path(raw_file(args, chrom, "haplotypes.csv.gz"))
        edges_raw, kind = str(hap_raw), f"hap_span{args.max_span}_min{args.min_pair_count}"
    else:
        edges_raw, kind = raw_file(args, chrom, EDGE_FILE[args.edges]), f"ind_{args.edges}"
    f_meta, f_car = os.path.join(d, "nodes.npz"), os.path.join(d, "carriers.npz")
    tag = f"_train-{train_tag}" if train_tag else ""
    f_edges = os.path.join(d, f"edges_{kind}_minlift{args.min_lift:g}{tag}_{file_stamp(edges_raw)}.npz")

    meta = np.load(f_meta) if os.path.exists(f_meta) and os.path.exists(f_car) else None
    if not args.force and meta is not None and str(meta.get("source", "")) == file_stamp(nodes_raw):
        ids, regions, inds = meta["ids"], meta["regions"], meta["individuals"].tolist()
        carriers = sp.load_npz(f_car).tocsr()
    else:
        ids, regions, carriers, inds = read_nodes(nodes_raw)
        sp.save_npz(f_car, carriers)
        np.savez(f_meta, ids=ids.astype(str), regions=regions.astype(str),
                 individuals=np.array(inds, dtype=str), source=np.array(file_stamp(nodes_raw)))

    cols = list(range(len(inds)))
    if train_names is not None:
        cols = [j for j, c in enumerate(inds) if c in train_names]
        if not cols:
            die(f"{chrom}: none of the training individuals match the columns of nodes.csv.gz "
                             f"(e.g. {inds[:2]}); pass --sample-map (CSV: graph name, sample ID)")
        C_train = carriers[:, cols]
    else:
        C_train = carriers
    support = np.asarray(C_train.sum(axis=1)).ravel()
    n_used = C_train.shape[1]
    H, blocks = read_haplotypes(hap_raw, ids, regions, carriers, inds) if hap_raw.exists() else (None, [])
    hap_rows = np.sort(np.concatenate([2 * np.asarray(cols), 2 * np.asarray(cols) + 1])) \
        if train_names is not None else np.arange(2 * len(inds))

    if not args.force and os.path.exists(f_edges):
        e = np.load(f_edges)
        s, t, lift = e["s"], e["t"], e["lift"]
        status = "cached"
    elif args.lift == "haplotype":
        starts = pd.Series(blocks).str.extract(REGION_RE)["start"].astype(np.int64).to_numpy()
        s, t, lift, n_pairs = haplotype_edges(H, starts, hap_rows, len(ids), args)
        np.savez(f_edges, s=s, t=t, lift=lift)
        status = f"{len(s):,}/{n_pairs:,} same-haplotype cluster pairs kept"
    else:
        packed = pack_rows(C_train) if train_names is not None else None
        s, t, lift, n_read, n_unk = read_edges(edges_raw, ids, support,
                                              n_used, args.min_lift, args.chunksize, packed)
        np.savez(f_edges, s=s, t=t, lift=lift)
        status = f"{len(s):,}/{n_read:,} edge rows kept" + (f", {n_unk:,} with unknown ids" if n_unk else "")
    return ids, regions, carriers, inds, s, t, lift, support, H, blocks, hap_rows, status


# ---------------------------------------------------------------------------
# Merge into one graph
# ---------------------------------------------------------------------------

def build(args):
    chroms = sorted(set(args.chroms), key=chrom_key)
    out = args.out
    train_names, train_tag, test_names = None, "", None
    to_graph = load_sample_map(args.sample_map) if args.sample_map else {}
    if args.train_individuals:
        raw = read_individual_list(args.train_individuals, os.path.join(args.workdir, "splits"))
        train_names = {to_graph.get(x, x) for x in raw}
        train_tag = set_id(train_names)
        log(f"train-only mode: {len(raw):,} training individuals from {args.train_individuals}")
        if args.test_individuals:
            test_names = {to_graph.get(x, x) for x in read_individual_list(args.test_individuals,
                                                                          os.path.join(args.workdir, "splits"))}
            overlap = train_names & test_names
            if overlap:
                die(f"{len(overlap):,} individuals are in both the train and the test set (e.g. {sorted(overlap)[:3]})")
        if args.lift == "individual":
            log(f"  note: --lift individual takes its candidate pairs from the dataset's {EDGE_FILE[args.edges]}, "
                "which was made from ALL individuals (weights are train-only); --lift haplotype avoids that")
    requested = {"chromosomes": chroms, "lift": args.lift, "min_lift": args.min_lift,
                 **({"edges_file": args.edges} if args.lift == "individual" else
                    {"max_span": args.max_span, "min_pair_count": args.min_pair_count}),
                 "train_file_set_id": train_tag or "all",
                 "test_file_set_id": set_id(test_names) if test_names is not None else None}
    if not args.force and all(os.path.exists(os.path.join(out, f)) for f in OUTPUTS):
        old = json.load(open(os.path.join(out, "graph_info.json")))
        differ = {k: (old.get(k), v) for k, v in requested.items() if old.get(k) != v}
        if not differ:
            log(f"graph already built in {out} with these settings (use --force to rebuild)")
            return
        die(f"{out} holds a graph built with other settings "
            + ", ".join(f"{k}: {a!r} there, {b!r} now" for k, (a, b) in differ.items())
            + ". Pass --force to rebuild it in place (steps 3, 4 and 6 outputs in that folder then become stale "
              "and are recomputed / refused), or --out for a new folder.")
    os.makedirs(out, exist_ok=True)
    log(f"building graph for {len(chroms)} chromosome(s) -> {out}")

    ref_inds = None
    ids_all, regions_all, chrom_all, carrier_blocks, sup_train, S, T, W = [], [], [], [], [], [], [], []
    haps, blocks_all, block_chrom, junc = [], [], [], []
    offset = 0
    for chrom in chroms:
        t0 = time.time()
        ids, regions, carriers, inds, s, t, lift, support, H, blocks, hap_rows, status = \
            chrom_part(args, chrom, train_names, train_tag)
        if ref_inds is None:
            ref_inds = inds
        elif inds != ref_inds:
            if set(inds) != set(ref_inds):
                die(f"{chrom}: individual columns differ from {chroms[0]}")
            pos = {c: j for j, c in enumerate(inds)}
            carriers = carriers[:, [pos[c] for c in ref_inds]].tocsr()
            if H is not None:
                order = np.ravel([[2 * pos[c], 2 * pos[c] + 1] for c in ref_inds])
                hap_rows = np.where(np.isin(order, hap_rows))[0]
                H = H[order]
        ids_all.append(ids); regions_all.append(regions); chrom_all.append(np.full(len(ids), chrom))
        carrier_blocks.append(carriers); sup_train.append(support)
        S.append(s + offset); T.append(t + offset); W.append(np.log1p(lift))
        if H is not None:
            col0 = len(blocks_all)
            for k in range(H.shape[1] - 1):           # training haplotypes' transitions between neighbours
                pair, cnt = np.unique(H[hap_rows, k].astype(np.int64) * len(ids) + H[hap_rows, k + 1],
                                      return_counts=True)
                junc.append((np.full(len(pair), col0 + k), pair // len(ids) + offset,
                             pair % len(ids) + offset, cnt))
            haps.append(H + offset); blocks_all += blocks; block_chrom += [chrom] * len(blocks)
        offset += len(ids)
        log(f"  {chrom}: {len(ids):,} nodes, {status}"
            + (f", {H.shape[0]:,} haplotypes x {H.shape[1]:,} haploblocks" if H is not None else "")
            + f"  ({time.time() - t0:.0f}s)")

    n = offset
    A = sp.coo_matrix((np.concatenate(W).astype(np.float32),
                       (np.concatenate(S), np.concatenate(T))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A = A.maximum(A.T).tocsr()          # co-occurrence is symmetric
    A.setdiag(0); A.eliminate_zeros(); A.sort_indices()
    carriers = sp.vstack(carrier_blocks).tocsr().astype(np.uint8)

    regions = np.concatenate(regions_all)
    parsed = pd.Series(regions).str.extract(REGION_RE)
    nodes = pd.DataFrame({
        "node": np.arange(n), "id": np.concatenate(ids_all), "chrom": np.concatenate(chrom_all),
        "region": regions,
        "start": pd.to_numeric(parsed["start"]).astype("Int64"),
        "end": pd.to_numeric(parsed["end"]).astype("Int64"),
        "support": np.asarray(carriers.sum(axis=1)).ravel(),
    })
    if train_names is not None:
        nodes["support_train"] = np.concatenate(sup_train).astype(np.int64)
    if nodes["start"].isna().any():
        log(f"  warning: {int(nodes['start'].isna().sum()):,} regions don't match <chrom>_<start>-<end>")

    nodes.to_csv(os.path.join(out, "nodes.csv.gz"), index=False)
    with open(os.path.join(out, "individuals.txt"), "w") as f:
        f.write("\n".join(ref_inds))
    train_in_graph = [x for x in ref_inds if train_names is not None and x in train_names]
    test_in_graph = [x for x in ref_inds if test_names is not None and x in test_names]
    for name, members, used in (("train_individuals.txt", train_in_graph, train_names is not None),
                                ("test_individuals.txt", test_in_graph, test_names is not None)):
        path = os.path.join(out, name)
        if used:
            with open(path, "w") as f:
                f.write("\n".join(members) + "\n")
        elif os.path.exists(path):
            os.remove(path)                 # a rebuild without a split must not leave a stale one
    for name in ("populations.tsv", "related_pairs.tsv"):
        src = os.path.join(args.workdir, "splits", name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, name))
    sp.save_npz(os.path.join(out, "carriers.npz"), carriers)
    sp.save_npz(os.path.join(out, "graph.npz"), A)
    with_haps = len(haps) == len(chroms)
    if with_haps:
        np.savez_compressed(os.path.join(out, "haplotypes.npz"), node=np.hstack(haps),
                            blocks=np.array(blocks_all), block_chrom=np.array(block_chrom))
        j = [np.concatenate(x) for x in zip(*junc)] if junc else [np.empty(0, np.int64)] * 4
        np.savez_compressed(os.path.join(out, "junctions.npz"), block=j[0], a=j[1], b=j[2], count=j[3])
    for f in HAP_OUTPUTS if not with_haps else ():
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))             # a rebuild without haplotypes must not leave stale ones
    deg = np.diff(A.indptr)
    info = {"edge_value": "log1p(lift)", "n_nodes": int(n), "n_edges": int(A.nnz // 2),
            "n_individuals": len(ref_inds), "isolated_nodes": int((deg == 0).sum()),
            "pair_source": "train_haplotypes" if args.lift == "haplotype" else f"dataset {EDGE_FILE[args.edges]}",
            "haplotypes": with_haps, "n_haploblocks": len(blocks_all) if with_haps else None,
            "train_only": train_names is not None, **requested}
    if train_names is not None:
        info.update({"train_individuals_source": args.train_individuals,
                     "train_individuals_in_file": len(train_names),
                     "train_individuals_in_graph": len(train_in_graph),
                     # same id as computed by steps 3-4 from train_individuals.txt
                     "train_set_id": set_id(train_in_graph)})
    if test_names is not None:
        info.update({"test_individuals_source": args.test_individuals,
                     "test_individuals_in_graph": len(test_in_graph), "test_set_id": set_id(test_in_graph)})
    with open(os.path.join(out, "graph_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"  graph: {n:,} nodes, {A.nnz // 2:,} edges, {len(ref_inds):,} individuals"
        + (f" ({len(train_in_graph):,} used for training)" if train_names is not None else "")
        + (f", {len(test_in_graph):,} held out for testing" if test_names is not None else "")
        + f", mean degree {deg.mean():.1f}, isolated {int((deg == 0).sum()):,}")
    if train_names is not None and len(train_in_graph) < len(train_names):
        log(f"  warning: {len(train_names) - len(train_in_graph):,} training IDs are not in the graph")


def resolve_split(args):
    """Default training/test sets: the split written by 1_download_haplograph.py (--workdir/splits)."""
    split_json = Path(args.workdir) / "splits" / "split.json"
    args.split_name = None
    if args.no_split:
        args.train_individuals = args.test_individuals = None
        return
    if args.train_individuals:
        args.split_name = re.sub(r"\.(txt|csv|tsv)(\.gz)?$", "", os.path.basename(args.train_individuals.rstrip("/")))
        if not args.test_individuals and split_json.exists():
            info = json.loads(split_json.read_text())
            split_train = split_json.parent / info["train_file"]
            if os.path.exists(args.train_individuals) and Path(args.train_individuals).resolve() == split_train.resolve():
                args.test_individuals = str(split_json.parent / info["test_file"])
                args.split_name = info.get("name", args.split_name)
        if not args.test_individuals:
            log("  warning: no --test-individuals given, so the graph folder gets no test_individuals.txt; "
                "step 6 will then need --test-individuals or --splits")
        return
    if not split_json.exists():
        die(f"no train/test split in {split_json.parent} - run 1_download_haplograph.py first, pass "
            "--train-individuals, or use --no-split to build the graph from all individuals")
    info = json.loads(split_json.read_text())
    args.train_individuals = str(split_json.parent / info["train_file"])
    args.test_individuals = args.test_individuals or str(split_json.parent / info["test_file"])
    args.split_name = info.get("name", "split")


def run_self_test(args):
    """Synthetic 2-chromosome haplograph -> graph, on CPU, in a few seconds."""
    import importlib
    wd = test_workdir("build_graph")
    importlib.import_module("1_download_haplograph").main(["--test"])                   # writes <tmp>/haplograph_selftest/download
    src = wd.parent / "download"
    shutil.copytree(src, wd, dirs_exist_ok=True)
    args.workdir, args.raw_dir, args.chroms, args.min_lift, args.force = str(wd), str(wd / "raw"), None, 1.2, True
    args.train_individuals = args.test_individuals = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", help="comma-separated selection, e.g. chr21,chr22 (default: every downloaded chromosome)")
    ap.add_argument("--workdir", default="haplograph", help="root folder of 1_download_haplograph.py (raw/, splits/); "
                                                            "the graph and cache are written here too")
    ap.add_argument("--raw-dir", help="folder with <chrom>/nodes.csv.gz etc. (default <workdir>/raw)")
    ap.add_argument("--out", help="output folder (default <workdir>/genome or <workdir>/<chroms>, "
                                  "+ _train_<split name> in train-only mode)")
    ap.add_argument("--train-individuals",
                    help="file or URL of training individuals (default: <workdir>/splits/train_individuals.txt); "
                         "edge weights, support and lift are computed from them only")
    ap.add_argument("--test-individuals",
                    help="file of test individuals, copied into the graph folder for step 6 "
                         "(default: <workdir>/splits/test_individuals.txt)")
    ap.add_argument("--no-split", action="store_true", help="build the graph from ALL individuals (no train/test split)")
    ap.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--lift", choices=["haplotype", "individual"],
                    help="edges from same-haplotype co-occurrence (default when step 1 downloaded --haplotypes) or "
                         "from individual co-carriage in the dataset's edge file")
    ap.add_argument("--max-span", type=int, default=500_000, help="--lift haplotype: max bp between linked haploblocks")
    ap.add_argument("--min-pair-count", type=int, default=3,
                    help="--lift haplotype: min training haplotypes carrying both clusters")
    ap.add_argument("--edges", choices=["all", "lift5"], default="all",
                    help="--lift individual: all = edges.csv.gz (~1 GB per chromosome); lift5 = pre-filtered lift>=5 file")
    ap.add_argument("--min-lift", type=float, default=2.0, help="keep edges with lift >= this")
    ap.add_argument("--chunksize", type=int, default=2_000_000,
                    help="edge rows read at a time; 0 = whole file at once (with --cudf, chunked reading "
                         "runs on the CPU, so 0 puts the parsing on the GPU if the file fits in GPU memory)")
    ap.add_argument("--force", action="store_true", help="re-parse cached chromosomes and rebuild")
    ap.add_argument("--cudf", action="store_true",
                    help="run the pandas parts (reading and matching nodes/edges) on the GPU with RAPIDS cuDF")
    ap.add_argument("--test", action="store_true", help="self-test on a small synthetic haplograph")
    args = ap.parse_args(argv)

    if args.test:
        run_self_test(args)
    args.raw_dir = args.raw_dir or os.path.join(args.workdir, "raw")
    if not os.path.isdir(args.raw_dir):
        die(f"{args.raw_dir} not found - run 1_download_haplograph.py first (or pass --raw-dir)")
    resolve_split(args)

    downloaded = sorted((p.name for p in Path(args.raw_dir).iterdir() if (p / "nodes.csv.gz").exists()),
                        key=chrom_key)
    args.chroms = parse_chroms(args.chroms) or [c for c in ALL_CHROMS if c in downloaded] or downloaded
    if not args.chroms:
        die(f"no <chrom>/nodes.csv.gz in {args.raw_dir} - run 1_download_haplograph.py first")
    if args.lift is None:
        has_haps = all((Path(args.raw_dir) / c / "haplotypes.csv.gz").exists() for c in args.chroms)
        args.lift = "haplotype" if has_haps else "individual"
    if args.max_span < 1 or args.min_pair_count < 1:
        die("--max-span and --min-pair-count must be >= 1")
    genome_wide = set(args.chroms) == set(ALL_CHROMS)
    name = "genome" if genome_wide else "_".join(args.chroms)
    if args.split_name:
        name += f"_train_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split_name)}"
    args.out = args.out or os.path.join(args.workdir, name)
    build(args)
    log("done.")


if __name__ == "__main__":
    main()
