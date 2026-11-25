"""
Step 1 - download the 1000 Genomes haplograph and write the train / test split of individuals.

Downloads nodes.csv.gz and the edge file of every chromosome (chr1-chr22, or --chroms) from
https://data.haploblocks.org/haplograph/1000G/ and checks them against checksums.md5. With
--haplotypes (needed by the haplotype-level steps 2-6) it also downloads, for every haploblock, the
cluster of each of the 2 x 2,548 phased haplotypes (haploblock_hashes/1000G/<chrom>/, checked against
all_chr_tsv_webserver.md5), decodes them and writes one table per chromosome. It writes the TRAINING set (steps 2-4 learn from it) and the
TEST set (step 6 validates on it): the BH_26 halves (--split bh26, default) or a new
population-stratified random split that keeps related individuals together (--split random), plus
the population of every individual (used only to stratify step 6's metrics, never as model input)
and the related pairs found in the 1000 Genomes pedigree.

OUTPUT (in --workdir, default haplograph/)
  raw/<chrom>/nodes.csv.gz        id, high_dim_edge (haploblock region), one 0/1 column per individual
  raw/<chrom>/edges.csv.gz        source, target, weight   (or edges_lift_above_threshold.csv.gz, --edges lift5)
  raw/<chrom>/haplotypes.csv.gz   --haplotypes: haplotype (<sample>_hap0 / _hap1), then one column per
                                  haploblock region in genomic order = the cluster number N of node
                                  <region>_cluster<N>
  raw/<chrom>/rep_seq.tar.gz      --sequences: one FASTA per haploblock, the representative haplotype
  raw/<chrom>/rep_seq.md5         sequence of each cluster (for step 3 evo2_seq; chr21: 3.1 GB)
  splits/train_individuals.txt    one sample ID per line (only individuals present in the haplograph)
  splits/test_individuals.txt
  splits/populations.tsv          individual, superpopulation, population
  splits/related_pairs.tsv        individual_a, individual_b, relation (parent-child / siblings), across_split
  splits/split.json               split name, source, sizes, set ids, superpopulations per set, related pairs

USAGE
  python 1_download_haplograph.py --haplotypes             # whole genome (~15 GB + ~19 GB haplotype files)
  python 1_download_haplograph.py --chroms chr21,chr22 --haplotypes
  python 1_download_haplograph.py --split random --test-fraction 0.3 --seed 1
  python 1_download_haplograph.py --test                   # synthetic data, no network
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from haplo_common import (ALL_CHROMS, BASE_URL, BH26_SPLITS, EDGE_FILE, HASHES_URL, PEDIGREE_URL, REGION_RE,
                          SEQUENCES_URL, die,
                          download, log, make_synthetic_haplograph, parse_chroms, read_individual_list, set_id,
                          test_workdir, url_cache_path, write_individual_list)


def md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(workdir):
    path = download(f"{BASE_URL}/checksums.md5", Path(workdir) / "raw" / "checksums.md5", quiet=True)
    sums = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("./")] = parts[0]
    return sums


def fetch_chromosome(workdir, chrom, files, checksums):
    """Download missing files and check their MD5. Files already verified (same size and modification
    time as when they passed, recorded in raw/verified.json) are not hashed again."""
    record_path = Path(workdir) / "raw" / "verified.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name in files:
        key = f"{chrom}/{name}"
        dest = Path(workdir) / "raw" / chrom / name
        download(f"{BASE_URL}/{chrom}/{name}", dest)
        expected = checksums.get(key) if checksums else None
        if not expected:
            continue
        st = dest.stat()
        stamp = [st.st_size, st.st_mtime_ns, expected]
        if record.get(key) == stamp:
            continue
        got = md5(dest)
        if got != expected:
            dest.unlink()
            record.pop(key, None)
            record_path.write_text(json.dumps(record, indent=1))
            die(f"checksum mismatch for {dest} (got {got}, expected {expected}); the file was deleted, "
                "run again to re-download it")
        record[key] = stamp
        record_path.write_text(json.dumps(record, indent=1))


def haplograph_individuals(nodes_path):
    """Individual columns of a nodes.csv.gz (the header only)."""
    with gzip.open(nodes_path, "rt") as f:
        header = f.readline().rstrip("\n\r").split(",")
    if header[:2] != ["id", "high_dim_edge"]:
        die(f"{nodes_path}: unexpected header {header[:3]} (expected id, high_dim_edge, <individuals>)")
    return header[2:]


def fetch_haplotypes(workdir, chrom, url=HASHES_URL, jobs=8, verify=True):
    """--haplotypes: per haploblock of nodes.csv.gz, download <chrom>_individual_hashes_<region>.tsv
    (one row per haplotype, <sample>_<chrom>_region_<region>_hap<0/1>; the last 20 bits of HASH are
    the cluster index, node <region>_cluster<index + 1>), check that every haplotype of every
    individual has exactly one cluster that exists in the haplograph, and write
    raw/<chrom>/haplotypes.csv.gz. Files downloaded here are checked against <url>/all_chr_tsv_webserver.md5
    (unless verify=False). The per-block files are removed once the table is written."""
    raw = Path(workdir) / "raw" / chrom
    out = raw / "haplotypes.csv.gz"
    if out.exists():
        return
    nodes = pd.read_csv(raw / "nodes.csv.gz", usecols=["id", "high_dim_edge"])
    individuals = haplograph_individuals(raw / "nodes.csv.gz")
    regions = nodes["high_dim_edge"].drop_duplicates()
    regions = regions.iloc[np.argsort(regions.str.extract(REGION_RE)["start"].astype(np.int64).to_numpy(),
                                      kind="stable")].tolist()
    known = set(nodes["id"])
    parts = raw / "haplotype_hashes"
    log(f"  {chrom}: haplotype clusters of {len(regions):,} haploblocks")

    names = [f"{chrom}_individual_hashes_{r.split('_', 1)[1]}.tsv" for r in regions]
    new = [n for n in names if not (parts / n).exists()]
    with ThreadPoolExecutor(jobs) as ex:
        files = list(ex.map(lambda n: download(f"{url}/{chrom}/{n}", parts / n, quiet=True), names))
    if verify and new:
        sums_file = download(f"{url}/all_chr_tsv_webserver.md5", Path(workdir) / "raw" / "haplotype_hashes.md5", quiet=True)
        sums = {p.split("/")[-1]: h for h, p in (line.split() for line in open(sums_file) if line.strip())}
        for n in new:
            got = md5(parts / n)
            if sums.get(n, got) != got:
                (parts / n).unlink()
                die(f"checksum mismatch for {parts / n} (got {got}, expected {sums[n]}); the file was deleted, "
                    "run again to re-download it")
    rows = [f"{x}_hap{h}" for x in individuals for h in (0, 1)]
    cols = {}
    for region, path in zip(regions, files):
        df = pd.read_csv(path, sep="\t", dtype=str)
        parsed = df["INDIVIDUAL"].str.extract(r"^(?P<sample>[^_]+)_.*_hap(?P<hap>[01])$")
        if parsed.isna().any().any():
            die(f"{path}: malformed haplotype names (expected <sample>_..._hap0/1)")
        key = parsed["sample"] + "_hap" + parsed["hap"]
        if key.duplicated().any():
            die(f"{path}: a haplotype is listed twice (e.g. {key[key.duplicated()].iloc[0]})")
        if not df["HASH"].str.fullmatch(r"[01]{20,}", na=False).all():
            die(f"{path}: HASH must be a 0/1 string of at least 20 bits")
        col = pd.Series(df["HASH"].str[-20:].map(lambda b: int(b, 2) + 1).to_numpy(), index=key).reindex(rows)
        if col.isna().any():
            die(f"{path}: {int(col.isna().sum())} haplotypes of haplograph individuals are missing "
                f"(e.g. {col.index[col.isna()][:3].tolist()})")
        bad = [f"{region}_cluster{n}" for n in col.unique() if f"{region}_cluster{n}" not in known]
        if bad:
            die(f"{path}: clusters not in nodes.csv.gz (e.g. {bad[:3]}); the two sources disagree")
        cols[region] = col.astype(np.int32).to_numpy()
    tmp = out.with_name(out.name + ".part")
    pd.DataFrame(cols, index=pd.Index(rows, name="haplotype")).to_csv(tmp, compression="gzip")
    os.replace(tmp, out)
    shutil.rmtree(parts, ignore_errors=True)
    log(f"  {chrom}: {len(rows):,} haplotypes x {len(regions):,} haploblocks -> {out}")


def populations(workdir, individuals):
    """individual, superpopulation, population (from phenotypes_real.csv; 'unknown' if absent)."""
    path = download(f"{BASE_URL}/phenotypes_real.csv", Path(workdir) / "raw" / "phenotypes_real.csv")
    ph = pd.read_csv(path, dtype=str)
    if not {"individual_id", "phenotype", "value"} <= set(ph.columns):
        die(f"{path}: expected columns individual_id, phenotype, value")
    get = lambda kind: ph[ph["phenotype"].str.lower() == kind].set_index("individual_id")["value"]
    sup, pop = get("ancestry"), get("population")
    return pd.DataFrame({"individual": individuals, "superpopulation": [sup.get(i, "unknown") for i in individuals],
                         "population": [pop.get(i, "unknown") for i in individuals]})


def related_pairs(workdir, individuals, url=PEDIGREE_URL):
    """Parent-child and sibling (same father or mother) pairs among the haplograph individuals,
    from the 1000 Genomes pedigree."""
    path = download(url, Path(workdir) / "raw" / "pedigree.txt", quiet=True)
    ped = pd.read_csv(path, sep=r"\s+", dtype=str)
    if not {"SampleID", "FatherID", "MotherID"} <= set(ped.columns):
        die(f"{path}: expected the 1000 Genomes pedigree columns SampleID, FatherID, MotherID")
    present = set(individuals)
    ped = ped[ped["SampleID"].isin(present)]
    pairs = set()
    for col in ("FatherID", "MotherID"):
        pairs |= {(*sorted((c, p)), "parent-child") for c, p in zip(ped["SampleID"], ped[col]) if p in present}
        for _, kids in ped[ped[col] != "0"].groupby(col)["SampleID"]:
            kids = sorted(kids)
            pairs |= {(a, b, "siblings") for i, a in enumerate(kids) for b in kids[i + 1:]}
    return pd.DataFrame(sorted(pairs), columns=["individual_a", "individual_b", "relation"])


def random_split(individuals, test_fraction, seed, strata=None, pairs=None):
    """Random split stratified by `strata`; related individuals (connected through `pairs`) are kept
    on the same side."""
    rng = np.random.default_rng(seed)
    family = {x: x for x in individuals}

    def root(x):
        while family[x] != x:
            x = family[x]
        return x
    for a, b in ([] if pairs is None else zip(pairs["individual_a"], pairs["individual_b"])):
        family[root(a)] = root(b)
    stratum = dict(zip(individuals, strata if strata is not None else ["all"] * len(individuals)))
    units = {}
    for x in individuals:
        units.setdefault(root(x), []).append(x)
    train, test = [], []
    for s in sorted(set(stratum.values())):
        groups = [g for r, g in units.items() if stratum[r] == s]
        rng.shuffle(groups)
        target, n_test = test_fraction * sum(map(len, groups)), 0
        for g in groups:
            if n_test + len(g) / 2 <= target:
                test.extend(g)
                n_test += len(g)
            else:
                train.extend(g)
    return sorted(train), sorted(test)


def make_split(args, individuals):
    splits = Path(args.workdir) / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    present = set(individuals)
    info = {"split": args.split, "n_haplograph_individuals": len(individuals)}
    if args.split == "bh26":
        halves = []
        for name in ("half_A.txt", "half_B.txt"):      # cached per URL; the copy in splits/ is always current
            url = f"{args.splits_url.rstrip('/')}/{name}"
            shutil.copyfile(download(url, url_cache_path(url, splits)), splits / name)
            halves.append(read_individual_list(splits / name))
        a, b = halves
        overlap = set(a) & set(b)
        if overlap:
            die(f"the BH_26 halves overlap ({len(overlap)} IDs, e.g. {sorted(overlap)[:3]})")
        train, test = [x for x in a if x in present], [x for x in b if x in present]
        info.update(name="half_A", source=args.splits_url, train_in_file=len(a), test_in_file=len(b))
    else:
        if not 0 < args.test_fraction < 1:
            die("--test-fraction must be between 0 and 1")
        strata = None if args.no_stratify else args.pops["population"].tolist()
        train, test = random_split(individuals, args.test_fraction, args.seed, strata, args.pairs)
        info.update(name=f"random{int(round(100 * args.test_fraction))}_seed{args.seed}",
                    source="random", test_fraction=args.test_fraction, seed=args.seed,
                    stratified=strata is not None)
    if not train or not test:
        die(f"empty split: {len(train)} train / {len(test)} test individuals found in the haplograph")
    write_individual_list(splits / "train_individuals.txt", train)
    write_individual_list(splits / "test_individuals.txt", test)
    side = {**{x: "train" for x in train}, **{x: "test" for x in test}}
    pairs = args.pairs.assign(across_split=[{side.get(a), side.get(b)} == {"train", "test"}
                                            for a, b in zip(args.pairs["individual_a"], args.pairs["individual_b"])])
    pairs.to_csv(splits / "related_pairs.tsv", sep="\t", index=False)
    args.pops.to_csv(splits / "populations.tsv", sep="\t", index=False)
    sup = args.pops.set_index("individual")["superpopulation"]
    info.update(superpopulations={k: sup.reindex(v).value_counts().to_dict() for k, v in (("train", train), ("test", test))},
                related_pairs=len(pairs), related_pairs_across_split=int(pairs["across_split"].sum()))
    if info["related_pairs_across_split"]:
        log(f"  warning: {info['related_pairs_across_split']} related pairs straddle the split (see related_pairs.tsv)")
    info.update(train=len(train), test=len(test),
                haplograph_individuals_in_neither_set=len(present - set(train) - set(test)),
                train_set_id=set_id(train), test_set_id=set_id(test),
                train_file="train_individuals.txt", test_file="test_individuals.txt")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", help="comma-separated chromosomes, e.g. chr21,chr22 or 21,22 (default: all, chr1-chr22)")
    ap.add_argument("--workdir", default="haplograph", help="root folder for raw/ and splits/")
    ap.add_argument("--edges", choices=sorted(EDGE_FILE) + ["both"], default="all",
                    help="all = edges.csv.gz (needed for a train-only graph); lift5 = small pre-filtered file")
    ap.add_argument("--split", choices=["bh26", "random"], default="bh26")
    ap.add_argument("--splits-url", default=BH26_SPLITS, help="folder with half_A.txt / half_B.txt (bh26)")
    ap.add_argument("--test-fraction", type=float, default=0.5, help="random split: share of individuals in test")
    ap.add_argument("--seed", type=int, default=0, help="random split seed")
    ap.add_argument("--no-stratify", action="store_true", help="random split: do not stratify by population")
    ap.add_argument("--no-verify", action="store_true", help="skip the MD5 checks of downloaded files")
    ap.add_argument("--haplotypes", action="store_true",
                    help="also download each haplotype's cluster per haploblock (phased data for steps 2-6)")
    ap.add_argument("--hashes-url", default=HASHES_URL, help="folder with <chrom>/<chrom>_individual_hashes_*.tsv")
    ap.add_argument("--sequences", action="store_true",
                    help="also download the representative sequence of every cluster (step 3 evo2_seq; GBs per chromosome)")
    ap.add_argument("--sequences-url", default=SEQUENCES_URL, help="folder with <chrom>_rep_seq.tar.gz / .md5")
    ap.add_argument("--pedigree-url", default=PEDIGREE_URL, help="1000 Genomes pedigree (related pairs)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads of the per-haploblock haplotype files")
    ap.add_argument("--test", action="store_true",
                    help="self-test: write a small synthetic haplograph (no network) and split it")
    args = ap.parse_args(argv)

    if args.jobs < 1:
        die("--jobs must be >= 1")
    chroms = parse_chroms(args.chroms) or ALL_CHROMS
    if args.test:
        args.workdir = str(test_workdir("download"))
        args.split, args.haplotypes, args.sequences, chroms = "random", True, True, ["chr21", "chr22"]
        make_synthetic_haplograph(Path(args.workdir) / "raw", chroms)
        log(f"[test] synthetic haplograph written to {args.workdir}/raw")
    else:
        unknown = [c for c in chroms if c not in ALL_CHROMS]
        if unknown:
            die(f"unknown chromosome(s) {unknown}; the haplograph has chr1-chr22 only")
        checksums = None if args.no_verify else load_checksums(args.workdir)
        log(f"downloading {len(chroms)} chromosome(s) into {args.workdir}/raw ({args.edges} edges)")
        for chrom in chroms:
            fetch_chromosome(args.workdir, chrom, ["nodes.csv.gz"] + (list(EDGE_FILE.values()) if args.edges == "both" else [EDGE_FILE[args.edges]]), checksums)

    individuals = None
    for chrom in chroms:                       # every chromosome must list the same individuals
        inds = haplograph_individuals(Path(args.workdir) / "raw" / chrom / "nodes.csv.gz")
        if individuals is None:
            individuals = inds
        elif set(inds) != set(individuals):
            die(f"{chrom}: individual columns differ from {chroms[0]}")
        if args.haplotypes:
            fetch_haplotypes(args.workdir, chrom, args.hashes_url, args.jobs, not args.no_verify)
        if args.sequences:
            for ext in ("md5", "tar.gz"):              # members are checked against the .md5 in step 3
                download(f"{args.sequences_url}/{chrom}_rep_seq.{ext}", Path(args.workdir) / "raw" / chrom / f"rep_seq.{ext}")

    args.pops = populations(args.workdir, individuals)
    args.pairs = related_pairs(args.workdir, individuals, args.pedigree_url)
    info = make_split(args, individuals)
    info["chromosomes"] = chroms
    info["edges"] = args.edges
    with open(Path(args.workdir) / "splits" / "split.json", "w") as f:
        json.dump(info, f, indent=2)
    log(f"split {info['name']}: {info['train']:,} train / {info['test']:,} test individuals "
        f"({info['haplograph_individuals_in_neither_set']:,} haplograph individuals in neither set)")
    log(f"  -> {os.path.join(args.workdir, 'splits', 'train_individuals.txt')}, test_individuals.txt")
    log("done.")


if __name__ == "__main__":
    main()
