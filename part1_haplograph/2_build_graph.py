"""
Step 2 - build the haplograph co-occurrence graph, genome-wide or for --chroms, from step 1's
downloads. Edge weight, support and lift are counted on the TRAINING individuals only (step 1's
split, or --train-individuals; --no-split = all individuals), so the test set never shapes the
graph: weight = training individuals carrying both clusters of a pair listed in the edge file,
lift = weight / (support_A * support_B / n_train), edges with lift < --min-lift are dropped.
Use --edges all for a train-only graph (the lift5 file was filtered on all individuals).

OUTPUT (in --out; default <workdir>/genome or <workdir>/<chrA>_<chrB>, + _train_<split name>)
  nodes.csv.gz      row i = node i: node, id, chrom, region, start, end, support, support_train
  individuals.txt   one individual per line = the columns of carriers.npz
  train_individuals.txt / test_individuals.txt   the split, restricted to the graph's individuals
  carriers.npz      scipy.sparse CSR uint8 [nodes, individuals], ALL individuals (step 6 needs the test ones)
  graph.npz         scipy.sparse CSR float32 [nodes, nodes], symmetric, value log(1 + lift);
                    within-chromosome edges only
  graph_info.json   settings, counts, training/test set ids (rerunning with other settings stops
                    unless --force)
Parsed chromosomes are cached in <workdir>/parts/<chrom>/, keyed by the raw files.

USAGE
  python 2_build_graph.py                       # all downloaded chromosomes -> haplograph/genome_train_half_A
  python 2_build_graph.py --chroms chr21,chr22  # -> haplograph/chr21_chr22_train_half_A
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


def file_stamp(path):
    """Short id of a raw file (absolute path, size, modification time): caches built from another raw
    folder or an older download are never reused."""
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:8]


def raw_file(args, chrom, name):
    p = Path(args.raw_dir) / chrom / name
    if not p.exists():
        die(f"{p} not found - download it first: python 1_download_haplograph.py --chroms {chrom}"
            + (" --edges lift5" if name.startswith("edges_lift") else ""))
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


def chrom_part(args, chrom, train_names=None, train_tag=""):
    """Parse one chromosome; cached in <workdir>/parts/<chrom>/.
    Returns ids, regions, carriers (all individuals), individual names, edges, lift, train support, status."""
    d = os.path.join(args.workdir, "parts", chrom)
    os.makedirs(d, exist_ok=True)
    edges_name = EDGE_FILE[args.edges]
    nodes_raw, edges_raw = raw_file(args, chrom, "nodes.csv.gz"), raw_file(args, chrom, edges_name)
    f_meta, f_car = os.path.join(d, "nodes.npz"), os.path.join(d, "carriers.npz")
    tag = f"_train-{train_tag}" if train_tag else ""
    f_edges = os.path.join(d, f"edges_{args.edges}_minlift{args.min_lift:g}{tag}_{file_stamp(edges_raw)}.npz")

    meta = np.load(f_meta) if os.path.exists(f_meta) and os.path.exists(f_car) else None
    if not args.force and meta is not None and str(meta.get("source", "")) == file_stamp(nodes_raw):
        ids, regions, inds = meta["ids"], meta["regions"], meta["individuals"].tolist()
        carriers = sp.load_npz(f_car).tocsr()
    else:
        ids, regions, carriers, inds = read_nodes(nodes_raw)
        sp.save_npz(f_car, carriers)
        np.savez(f_meta, ids=ids.astype(str), regions=regions.astype(str),
                 individuals=np.array(inds, dtype=str), source=np.array(file_stamp(nodes_raw)))

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

    if not args.force and os.path.exists(f_edges):
        e = np.load(f_edges)
        s, t, lift = e["s"], e["t"], e["lift"]
        status = "cached"
    else:
        packed = pack_rows(C_train) if train_names is not None else None
        s, t, lift, n_read, n_unk = read_edges(edges_raw, ids, support,
                                              n_used, args.min_lift, args.chunksize, packed)
        np.savez(f_edges, s=s, t=t, lift=lift)
        status = f"{len(s):,}/{n_read:,} edge rows kept" + (f", {n_unk:,} with unknown ids" if n_unk else "")
    return ids, regions, carriers, inds, s, t, lift, support, status


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
        if args.edges == "lift5":
            log("  warning: the lift5 edge file was pre-filtered on ALL individuals (incl. the test set); "
                "use --edges all for a strictly train-only graph")
    requested = {"chromosomes": chroms, "edges_file": args.edges, "min_lift": args.min_lift,
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
    offset = 0
    for chrom in chroms:
        t0 = time.time()
        ids, regions, carriers, inds, s, t, lift, support, status = chrom_part(args, chrom, train_names, train_tag)
        if ref_inds is None:
            ref_inds = inds
        elif inds != ref_inds:
            if set(inds) != set(ref_inds):
                die(f"{chrom}: individual columns differ from {chroms[0]}")
            pos = {c: j for j, c in enumerate(inds)}
            carriers = carriers[:, [pos[c] for c in ref_inds]].tocsr()
        ids_all.append(ids); regions_all.append(regions); chrom_all.append(np.full(len(ids), chrom))
        carrier_blocks.append(carriers); sup_train.append(support)
        S.append(s + offset); T.append(t + offset); W.append(np.log1p(lift))
        offset += len(ids)
        log(f"  {chrom}: {len(ids):,} nodes, {status}  ({time.time() - t0:.0f}s)")

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
    sp.save_npz(os.path.join(out, "carriers.npz"), carriers)
    sp.save_npz(os.path.join(out, "graph.npz"), A)
    deg = np.diff(A.indptr)
    info = {"chromosomes": chroms, "edges_file": args.edges, "min_lift": args.min_lift,
            "edge_value": "log1p(lift)", "n_nodes": int(n), "n_edges": int(A.nnz // 2),
            "n_individuals": len(ref_inds), "isolated_nodes": int((deg == 0).sum()),
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
    ap.add_argument("--edges", choices=["all", "lift5"], default="all",
                    help="all = edges.csv.gz (~1 GB per chromosome); lift5 = pre-filtered lift>=5 file")
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
    genome_wide = set(args.chroms) == set(ALL_CHROMS)
    name = "genome" if genome_wide else "_".join(args.chroms)
    if args.split_name:
        name += f"_train_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split_name)}"
    args.out = args.out or os.path.join(args.workdir, name)
    build(args)
    log("done.")


if __name__ == "__main__":
    main()
