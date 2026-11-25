"""
Helpers shared by the part 1 scripts (haplograph -> embeddings -> Evo 2): logging, downloads,
individual lists and the training set, the train/test split files, device selection, and a small
synthetic haplograph used by every script's --test flag.
"""
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://data.haploblocks.org/haplograph/1000G"
HASHES_URL = "https://data.haploblocks.org/haploblock_hashes/1000G"      # per-haplotype cluster of every block
SEQUENCES_URL = "https://data.haploblocks.org/haploblock_sequences/1000G"  # representative sequence per cluster
BH26_SPLITS = "https://data.haploblocks.org/BH_26/splits"
PEDIGREE_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/"
                "20130606_g1k_3202_samples_ped_population.txt")
ALL_CHROMS = [f"chr{i}" for i in range(1, 23)]      # chrX/chrY are not in the dataset (see its README)
METHODS = ("node2vec", "gnn", "xgboost", "carrier_svd", "evo2_seq")      # step-3 embedding methods
DEFAULT_METHODS = ("node2vec", "gnn", "carrier_svd")    # --method all (xgboost did not beat carrier_svd; evo2_seq: GPU)
EDGE_FILE = {"all": "edges.csv.gz", "lift5": "edges_lift_above_threshold.csv.gz"}
REGION_RE = re.compile(r"^(?P<chrom>.+)_(?P<start>\d+)-(?P<end>\d+)$")


def log(msg=""):
    print(msg, flush=True)


def die(msg):
    """Stop with a readable message instead of a traceback."""
    raise SystemExit(f"error: {msg}")


def chrom_key(c):
    m = re.fullmatch(r"chr(\d+)", c)
    return (0, int(m.group(1)), "") if m else (1, 0, c)


def parse_chroms(text):
    """'chr21,chr22' / '21,22' / 'all' / None -> sorted list of chromosome names (None = all)."""
    if not text or text.strip().lower() in ("all", "genome"):
        return None
    out = []
    for c in text.split(","):
        c = c.strip()
        if c:
            out.append(c if c.lower().startswith("chr") else f"chr{c}")
    return sorted(set(out), key=chrom_key)


def is_url(p):
    return str(p).startswith(("http://", "https://", "file://"))


def url_cache_path(url, cache_dir=None):
    """Where a downloaded URL is cached: <cache_dir>/downloads/<hash of the URL>/<file name>, so two
    URLs with the same file name (or a local file of that name) never stand in for each other."""
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    return Path(cache_dir or tempfile.gettempdir()) / "downloads" / key / os.path.basename(url.rstrip("/"))


def download(url, dest, quiet=False, hint=""):
    """Download url to dest (atomic, skipped if dest exists). Returns dest as str."""
    dest = Path(dest)
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        log(f"  downloading {url}")
    tmp = dest.with_name(dest.name + ".part")

    def hook(blocks, bs, total):
        if total > 0 and not quiet and sys.stdout.isatty():      # no \r progress lines in log files
            sys.stdout.write(f"\r    {min(blocks * bs, total) / 1e6:9.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        die(f"could not download {url} ({type(e).__name__}: {e})" + (f"\n{hint}" if hint else ""))
    os.replace(tmp, dest)
    if not quiet:
        log()
    return str(dest)


def read_individual_list(src, cache_dir=None):
    """Plain text (one ID per line, '#' comments) or CSV with an 'individual' column; file or URL."""
    if is_url(src):
        src = download(src, url_cache_path(src, cache_dir))
    elif not os.path.exists(src):
        die(f"file not found: {src}")
    src = str(src)
    if src.endswith((".csv", ".csv.gz")):
        import pandas as pd
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


def write_individual_list(path, names):
    with open(path, "w") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


def load_sample_map(path):
    """CSV with 2 columns (graph individual name, sample ID) -> {sample ID: graph name}."""
    import pandas as pd
    if not os.path.exists(path):
        die(f"--sample-map file not found: {path}")
    m = pd.read_csv(path, dtype=str)
    if m.shape[1] < 2:
        die(f"{path}: --sample-map needs 2 columns (graph name, sample ID)")
    return dict(zip(m.iloc[:, 1].str.strip(), m.iloc[:, 0].str.strip()))


def set_id(names):
    """Short, order-independent id of a set of individuals (stored by every step and cross-checked)."""
    return hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:8]


def read_graph_individuals(graph_dir):
    p = Path(graph_dir) / "individuals.txt"
    if not p.exists():
        die(f"missing {p} - run 2_build_graph.py first")
    return [x for x in p.read_text().split("\n") if x]


def train_columns(args, individuals):
    """Training individuals as column indices into individuals.txt / carriers.npz.
    Source: --train-individuals if given, else the graph's train_individuals.txt (written by
    2_build_graph.py), else every individual. It must be the set the graph was built from (so the
    test individuals can never enter training). Returns (indices, source description, set id)."""
    src = getattr(args, "train_individuals", None)
    default = Path(args.graph_dir) / "train_individuals.txt"
    if not src and default.exists():
        src = str(default)
    if not src:
        idx, src, sid = np.arange(len(individuals)), "all individuals", "all"
    else:
        idx, sid = _train_idx(args, src, individuals)
    info = Path(args.graph_dir) / "graph_info.json"
    graph_sid = json.loads(info.read_text()).get("train_set_id", "all") if info.exists() else sid
    if sid != graph_sid:
        die(f"the training set {sid} ({src}) is not the one the graph was built from ({graph_sid}): every step "
            "must train on the graph's train_individuals.txt; rebuild the graph with step 2 to use another set")
    return idx, src, sid


def _train_idx(args, src, individuals):
    raw = read_individual_list(src, args.graph_dir)
    to_graph = load_sample_map(args.sample_map) if getattr(args, "sample_map", None) else {}
    names = {to_graph.get(x, x) for x in raw}
    pos = {x: i for i, x in enumerate(individuals)}
    idx = np.array(sorted(pos[x] for x in names if x in pos), dtype=np.int64)
    if len(idx) == 0:
        die(f"none of the training individuals in {src} (e.g. {sorted(names)[:2]}) match the "
            f"graph's individuals (e.g. {individuals[:2]}); pass --sample-map")
    if len(idx) < len(names):
        log(f"  warning: {len(names) - len(idx):,} training IDs from {src} are not in the graph, ignored")
    return idx, set_id(individuals[i] for i in idx)


def graph_id(graph_dir):
    """Short id of a graph build (its graph_info.json: chromosomes, edge file, min lift, counts,
    training set). Stored by steps 3 and 4 so outputs of an older graph in the same folder are
    never reused."""
    p = Path(graph_dir) / "graph_info.json"
    if not p.exists():
        return "unknown"
    info = json.loads(p.read_text())
    return hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:8]


def torch_device(name=None):
    """--device value, else cuda if available, else cpu (MPS is not used: sparse ops are missing there)."""
    import torch
    if name:
        if name.startswith("cuda") and not torch.cuda.is_available():
            log(f"  warning: --device {name} requested but CUDA is not available; using cpu")
            return torch.device("cpu")
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_args(args, positive=(), training=False):
    """Stop on impossible option values: every name in `positive` must be >= 1; training=True adds
    the training options shared by steps 4 and 4b."""
    if training:
        positive = tuple(positive) + ("batch", "epochs", "val_examples", "log_every", "eval_every")
        if not 0 < args.val_frac < 1:
            die("--val-frac must be between 0 and 1")
        if args.train_blocks is not None and args.train_blocks < 0:
            die("--train-blocks must be >= 0")
    for name in positive:
        if getattr(args, name) < 1:
            die(f"--{name.replace('_', '-')} must be >= 1")


def require_files(folder, names, hint):
    missing = [n for n in names if not (Path(folder) / n).exists()]
    if missing:
        die(f"missing {', '.join(missing)} in {folder} - run {hint} first")


def test_workdir(name):
    """Fresh folder for a --test run (kept so the outputs can be inspected)."""
    d = Path(tempfile.gettempdir()) / "haplograph_selftest" / name
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def selftest_graph_dir():
    """Graph folder written by 2_build_graph.py --test."""
    hits = sorted((Path(tempfile.gettempdir()) / "haplograph_selftest" / "build_graph").glob("*_train_*"))
    if not hits:
        die("self-test graph was not built (run 2_build_graph.py --test)")
    return str(hits[0])


# ---------------------------------------------------------------------------
# Synthetic haplograph (same file layout as data.haploblocks.org) for --test runs
# ---------------------------------------------------------------------------

SUPERPOPS = ("AFR", "EUR", "EAS")                           # synthetic population groups


def make_synthetic_haplograph(raw_dir, chroms=("chr21", "chr22"), n_ind=60, n_regions=12,
                              clusters=(2, 5), seed=0, switch=0.3):
    """Synthetic 1000G-like data drawn from a known block-HMM, written in the layout of
    data.haploblocks.org: <raw_dir>/<chrom>/nodes.csv.gz, edges.csv.gz, edges_lift_above_threshold.csv.gz,
    the per-block haplotype files <raw_dir>/<chrom>/haplotype_hashes/<chrom>_individual_hashes_<region>.tsv,
    <raw_dir>/<chrom>/rep_seq.tar.gz + rep_seq.md5 (a representative sequence per cluster),
    <raw_dir>/phenotypes_real.csv and <raw_dir>/pedigree.txt (two parent-child trios).
    Each individual belongs to one of 3 populations. Along each haplotype, the cluster of block k is
    drawn from a transition matrix given the cluster of block k-1, or (probability `switch`) from
    the population's own cluster frequencies, so clusters are linked between neighbouring blocks and
    differ between populations. The generating parameters are saved in <raw_dir>/synthetic_truth.npz
    (see synthetic_next_block_ce). Returns the individual names."""
    rng = np.random.default_rng(seed)
    inds = [f"HG{90000 + i:05d}" for i in range(n_ind)]
    group = rng.integers(0, 3, n_ind)
    raw_dir = Path(raw_dir)
    truth = {"group": group, "switch": switch}
    for c, chrom in enumerate(chroms):
        d = raw_dir / chrom
        (d / "haplotype_hashes").mkdir(parents=True, exist_ok=True)
        pos, prev, haps = 1_000_000 * (c + 1), None, np.zeros((2 * n_ind, n_regions), np.int64)
        regions = []
        for r in range(n_regions):
            start, end = pos, pos + int(rng.integers(20_000, 80_000))
            pos = end
            k = int(rng.integers(clusters[0], clusters[1] + 1))
            pref = rng.dirichlet(np.ones(k) * 0.5, size=3)              # population cluster frequencies
            trans = rng.dirichlet(np.ones(k) * 0.3, size=prev) if prev else None
            for h in range(2 * n_ind):
                g = group[h // 2]
                p = pref[g] if trans is None or rng.random() < switch else trans[haps[h, r - 1]]
                haps[h, r] = rng.choice(k, p=p)
            truth[f"{chrom}_{r}_pref"], prev = pref, k
            if trans is not None:
                truth[f"{chrom}_{r}_trans"] = trans
            region = f"{chrom}_{start}-{end}"
            regions.append((region, k))
            with open(d / "haplotype_hashes" / f"{chrom}_individual_hashes_{start}-{end}.tsv", "w") as f:
                f.write("INDIVIDUAL\tHASH\n")
                for h in rng.permutation(2 * n_ind):                     # files are not sorted
                    f.write(f"{inds[h // 2]}_{chrom}_region_{start}-{end}_hap{h % 2}\t"
                            f"{c + 1:09b}{r:020b}{haps[h, r]:020b}\n")
        truth[f"{chrom}_haplotypes"] = haps
        import io
        import tarfile
        sums = []
        with tarfile.open(d / "rep_seq.tar.gz", "w:gz") as tar:          # cluster j of a block = its base
            for r, (region, k) in enumerate(regions):                    # sequence with j variants
                base = rng.choice(list("ACGT"), 300)
                recs = []
                for j in np.unique(haps[:, r]):                       # clusters nobody carries do not exist
                    rep = int(np.where(haps[:, r] == j)[0][0])
                    seq = base.copy()
                    seq[rng.choice(300, j, replace=False)] = rng.choice(list("ACGT"), j)
                    recs.append(f">{inds[rep // 2]}_{chrom}_region_{region.split('_', 1)[1]}_hap{rep % 2}\n"
                                + "".join(seq).lower() + "\n")
                data = "".join(recs).encode()
                name = f"{chrom}/clusters/{region}_rep_seq.fasta"
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                sums.append(f"{hashlib.md5(data).hexdigest()}  {name}\n")
        (d / "rep_seq.md5").write_text("".join(sums))
        ids, rows = [], []
        for r, (region, k) in enumerate(regions):
            for j in np.unique(haps[:, r]):
                ids.append(f"{region}_cluster{j + 1}")
                rows.append(((haps[0::2, r] == j) | (haps[1::2, r] == j)).astype(np.uint8))
        M = np.vstack(rows)
        with gzip.open(d / "nodes.csv.gz", "wt") as f:
            f.write("id,high_dim_edge," + ",".join(inds) + "\n")
            for i, cid in enumerate(ids):
                f.write(f"{cid},{cid.rsplit('_cluster', 1)[0]}," + ",".join(map(str, M[i])) + "\n")
        W = M.astype(np.int64) @ M.T.astype(np.int64)
        sup = M.sum(1)
        s, t = np.triu_indices(len(ids), k=1)
        w = W[s, t]
        keep = w > 0
        s, t, w = s[keep], t[keep], w[keep]
        lift = w / np.maximum(sup[s] * sup[t] / n_ind, 1e-9)
        with gzip.open(d / "edges.csv.gz", "wt") as f:
            f.write("source,target,weight\n")
            for a, b, x in zip(s, t, w):
                f.write(f"{ids[a]},{ids[b]},{x}\n")
        with gzip.open(d / "edges_lift_above_threshold.csv.gz", "wt") as f:
            f.write("source,target,weight,lift\n")
            for a, b, x, l in zip(s, t, w, lift):
                if l >= 5:
                    f.write(f"{ids[a]},{ids[b]},{x},{l:.2f}\n")
    np.savez(raw_dir / "synthetic_truth.npz", **truth)
    with open(raw_dir / "phenotypes_real.csv", "w") as f:
        f.write("individual_id,phenotype,value,source\n")
        for i, g in zip(inds, group):
            f.write(f"{i},ancestry,{SUPERPOPS[g]},synthetic\n{i},population,{SUPERPOPS[g]}1,synthetic\n")
    with open(raw_dir / "pedigree.txt", "w") as f:                     # 1000G pedigree format
        f.write("FamilyID SampleID FatherID MotherID Sex Population Superpopulation\n")
        for j, i in enumerate(inds):
            parents = (inds[j - 1], inds[j - 2]) if j in (4, 9) else ("0", "0")
            f.write(f"F{j} {i} {parents[0]} {parents[1]} {1 + j % 2} {SUPERPOPS[group[j]]}1 {SUPERPOPS[group[j]]}\n")
    return inds


def synthetic_next_block_ce(raw_dir, chrom, hap_rows):
    """Cross-entropy (nats per prediction) of the TRUE generating model of make_synthetic_haplograph
    when it predicts block k+1 from blocks 1..k of each haplotype in `hap_rows` (rows of the chromosome's
    haplotype matrix, 2*i and 2*i+1 for individual i). The population is hidden, so the true
    predictive distribution mixes the 3 populations by their posterior given the blocks so far."""
    t = np.load(Path(raw_dir) / "synthetic_truth.npz")
    haps, sw = t[f"{chrom}_haplotypes"][hap_rows], float(t["switch"])
    prior = np.bincount(t["group"], minlength=3) / len(t["group"])
    total = n = 0.0
    for x in haps:
        post = prior * t[f"{chrom}_0_pref"][:, x[0]]
        post /= post.sum()
        for r in range(1, len(x)):
            pref, trans = t[f"{chrom}_{r}_pref"], t[f"{chrom}_{r}_trans"]
            p_g = sw * pref[:, x[r]] + (1 - sw) * trans[x[r - 1], x[r]]   # P(x_r | x_{r-1}, population)
            total -= np.log(post @ p_g)
            n += 1
            post = post * p_g
            post /= post.sum()
    return total / max(n, 1)
