"""
Step 2 of 5 - ancestry labels for the individuals and the nodes of the graph from 1_build_graph.py.

Needed for: 3_node_embeddings.py --method xgboost, and 4a --objective ancestry.
Skip it if you only use node2vec / gnn with --objective next.

  per individual : super-population (the "ancestry" phenotype), for every individual
  per node       : the super-population most ENRICHED among the cluster's carriers
                   (enrichment = share of carriers from a population / that population's share of
                   all individuals; enrichment, not majority, so the largest population doesn't win
                   everywhere), computed from the TRAINING individuals only (see below)

TRAINING INDIVIDUALS  node labels use only these, so a held-out test set never shapes them:
  --train-individuals FILE_OR_URL if given, else <graph-dir>/train_individuals.txt (written by
  1_build_graph.py --train-individuals), else all individuals. Format: one sample ID per line
  (e.g. BH_26 half_A.txt) or a CSV with an "individual" column.

INPUT
  --graph-dir/individuals.txt   I lines, one individual (column name of nodes.csv.gz) per line
  --graph-dir/carriers.npz      scipy.sparse CSR uint8 [N, I], node x individual carriers
  --graph-dir/train_individuals.txt   (optional) training individuals, one per line
  phenotypes_real.csv           (--phenotypes, or downloaded from the dataset root)
                                CSV of real labels from IGSR (ancestry, population, sex).
                                Long format (one row per individual-phenotype pair: sample id,
                                phenotype name, value) or wide format (one column per phenotype)
                                are both accepted; columns are auto-detected and can be set with
                                --pheno-id-col / --pheno-name-col / --pheno-value-col.
  --sample-map (optional)       CSV with 2 columns: graph individual name, sample ID; needed only
                                if the graph names individuals differently (e.g. IND1 vs HG00096).
                                Used for both the phenotype file and the train file.

OUTPUT (in --graph-dir)
  individual_ancestry.csv  CSV, I rows in individuals.txt order (all individuals):
                             individual  graph name
                             ancestry    super-population label, empty if unknown
  labels.npz               numpy archive:
                             labels      int64 [N], index into classes, -1 = no labelled carriers
                             classes     str [P], sorted population labels
                             enrichment  float32 [N, P], per-node enrichment of each population
                             train_set_id  str, id of the training set used (checked by step 3)
                             n_train     int, number of training individuals used

USAGE
  python 2_ancestry_labels.py --graph-dir haplograph/genome_train_half_A
  python 2_ancestry_labels.py --graph-dir haplograph/chr21_chr22 --phenotypes phenotypes_real.csv

REQUIRES  numpy, pandas, scipy.  Run 1_build_graph.py first.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

PHENO_URL = "https://data.haploblocks.org/haplograph/1000G/phenotypes_real.csv"


def log(msg=""):
    print(msg, flush=True)


def gfile(args, name):
    return os.path.join(args.graph_dir, name)


def download(url, dest):
    if os.path.exists(dest):
        return dest
    log(f"  downloading {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def is_url(p):
    return str(p).startswith(("http://", "https://"))


def read_individual_list(src, cache_dir):
    """Plain text (one ID per line, '#' comments) or CSV with an 'individual' column; file or URL."""
    if is_url(src):
        import urllib.request
        dest = os.path.join(cache_dir, os.path.basename(src.rstrip("/")))
        if not os.path.exists(dest):
            log(f"  downloading {src}")
            urllib.request.urlretrieve(src, dest + ".part")
            os.replace(dest + ".part", dest)
        src = dest
    elif not os.path.exists(src):
        raise SystemExit(f"file not found: {src}")
    if src.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(src, dtype=str)
        col = "individual" if "individual" in df.columns else df.columns[0]
        return [x.strip() for x in df[col].dropna()]
    out = []
    with open(src) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line.split()[0])
    return out


def load_sample_map(path):
    """CSV with 2 columns (graph individual name, sample ID) -> {sample ID: graph name}."""
    m = pd.read_csv(path, dtype=str)
    if m.shape[1] < 2:
        raise SystemExit(f"{path}: --sample-map needs 2 columns (graph name, sample ID)")
    return dict(zip(m.iloc[:, 1].str.strip(), m.iloc[:, 0].str.strip()))


def train_columns(args, individuals):
    """Training individuals as column indices into individuals.txt / carriers.npz.
    Source: --train-individuals if given, else the graph's train_individuals.txt (written by
    1_build_graph.py --train-individuals), else every individual.
    Returns (indices, source description, short id of the set)."""
    import hashlib
    src = getattr(args, "train_individuals", None)
    if not src and os.path.exists(os.path.join(args.graph_dir, "train_individuals.txt")):
        src = os.path.join(args.graph_dir, "train_individuals.txt")
    if not src:
        return np.arange(len(individuals)), "all individuals", "all"
    raw = read_individual_list(src, args.graph_dir)
    to_graph = load_sample_map(args.sample_map) if getattr(args, "sample_map", None) else {}
    names = {to_graph.get(x, x) for x in raw}
    pos = {x: i for i, x in enumerate(individuals)}
    idx = np.array(sorted(pos[x] for x in names if x in pos), dtype=np.int64)
    if len(idx) == 0:
        raise SystemExit(f"none of the training individuals in {src} (e.g. {sorted(names)[:2]}) match the "
                         f"graph's individuals (e.g. {individuals[:2]}); pass --sample-map")
    if len(idx) < len(names):
        log(f"  warning: {len(names) - len(idx):,} training IDs from {src} are not in the graph, ignored")
    set_id = hashlib.sha1("\n".join(sorted(individuals[i] for i in idx)).encode()).hexdigest()[:8]
    return idx, src, set_id


def individual_ancestry(args, individuals):
    """Ancestry label per individual, aligned to `individuals` (None where unknown)."""
    pheno_path = args.phenotypes or download(PHENO_URL, gfile(args, "phenotypes_real.csv"))
    ph = pd.read_csv(pheno_path, dtype=str)
    inds = set(individuals)
    to_col = None
    if args.sample_map:
        m = pd.read_csv(args.sample_map, dtype=str)
        to_col = dict(zip(m.iloc[:, 1], m.iloc[:, 0]))

    def overlap(col):
        vals = ph[col].dropna()
        if to_col is not None:
            vals = vals.map(lambda x: to_col.get(x, x))     # IDs already named like the graph stay as they are
        return vals.isin(inds).mean()

    id_col = args.pheno_id_col or max(ph.columns, key=overlap)
    if overlap(id_col) == 0:
        raise SystemExit(
            "No phenotype rows match the individual columns of nodes.csv.gz "
            f"(e.g. {individuals[:3]}). Pass --pheno-id-col and/or --sample-map.")

    ids = ph[id_col] if to_col is None else ph[id_col].map(lambda x: to_col.get(x, x))
    key = args.ancestry_key.lower()
    others = [c for c in ph.columns if c != id_col]
    wide = [c for c in others if c.lower() == key]
    if wide:                                   # wide format: a column called e.g. "ancestry"
        values = ph[wide[0]]
    else:                                      # long format: (id, phenotype name, value)
        name_col = args.pheno_name_col or max(
            others, key=lambda c: ph[c].astype(str).str.lower().eq(key).sum())
        value_col = args.pheno_value_col or next(c for c in others if c != name_col)
        rows = ph[name_col].astype(str).str.lower().eq(key)
        if not rows.any():
            raise SystemExit(f"no rows with phenotype '{args.ancestry_key}' in column '{name_col}'. "
                             f"Values seen: {ph[name_col].unique()[:10]}. Use --ancestry-key.")
        ids, values = ids[rows], ph.loc[rows, value_col]

    lookup = dict(zip(ids, values))
    return np.array([lookup.get(i) if isinstance(lookup.get(i), str) else None for i in individuals],
                    dtype=object)


def node_labels(carriers, anc):
    known = np.array([a is not None for a in anc])
    if not known.any():
        raise SystemExit("no training individual has a known ancestry: check that the phenotype IDs match the "
                         "graph's individuals (or --sample-map) and that the train set is right")
    classes = np.array(sorted(set(anc[known])))
    onehot = np.zeros((len(anc), len(classes)), np.float32)
    onehot[np.where(known)[0], np.searchsorted(classes, anc[known])] = 1
    counts = np.asarray(carriers @ onehot)                     # [N, P] carriers per population
    frac = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    enrichment = (frac / (onehot.sum(0) / onehot.sum())).astype(np.float32)
    labels = np.where(counts.sum(1) > 0, enrichment.argmax(1), -1).astype(np.int64)
    return labels, classes, enrichment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", default="haplograph/genome", help="output folder of 1_build_graph.py")
    ap.add_argument("--phenotypes", help="local phenotypes_real.csv (downloaded if omitted)")
    ap.add_argument("--ancestry-key", default="ancestry", help="phenotype to use as the label")
    ap.add_argument("--pheno-id-col"); ap.add_argument("--pheno-name-col"); ap.add_argument("--pheno-value-col")
    ap.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--train-individuals",
                    help="file or URL of training individuals (default: <graph-dir>/train_individuals.txt, else all)")
    ap.add_argument("--force", action="store_true", help="recompute existing labels")
    args = ap.parse_args()

    for f in ("individuals.txt", "carriers.npz"):
        if not os.path.exists(gfile(args, f)):
            raise SystemExit(f"missing {f} in {args.graph_dir} - run 1_build_graph.py first")
    individuals = open(gfile(args, "individuals.txt")).read().split("\n")
    train_idx, train_src, train_id = train_columns(args, individuals)

    out_files = ("individual_ancestry.csv", "labels.npz")
    if not args.force and all(os.path.exists(gfile(args, f)) for f in out_files):
        old = str(np.load(gfile(args, "labels.npz")).get("train_set_id", "all"))
        if old == train_id:
            log(f"labels already in {args.graph_dir} for this training set (use --force to recompute)")
            return
        log(f"existing labels were made with another training set ({old}); recomputing")

    anc = individual_ancestry(args, individuals)
    pd.DataFrame({"individual": individuals,
                  "ancestry": [a or "" for a in anc]}).to_csv(gfile(args, "individual_ancestry.csv"), index=False)
    log(f"individuals: ancestry known for {sum(a is not None for a in anc):,}/{len(individuals):,}")

    carriers = sp.load_npz(gfile(args, "carriers.npz")).tocsc()[:, train_idx].astype(np.float32)
    labels, classes, enrichment = node_labels(carriers, anc[train_idx])
    np.savez(gfile(args, "labels.npz"), labels=labels, classes=classes, enrichment=enrichment,
             train_set_id=np.array(train_id), n_train=np.array(len(train_idx)))
    dist = pd.Series(classes[labels[labels >= 0]]).value_counts().to_dict()
    log(f"nodes (from {len(train_idx):,} training individuals, {train_src}): {dist}  "
        f"(unlabelled {(labels < 0).sum():,})")
    log("done.")


if __name__ == "__main__":
    main()