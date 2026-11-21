"""
Step 0 - download the 1000 Genomes haplograph and make the train / test split of individuals.

Downloads, genome-wide (chr1-chr22, the default) or for selected chromosomes (--chroms), the files
the pipeline needs from https://data.haploblocks.org/haplograph/1000G/<chrom>/, checks them against
the dataset's checksums.md5, and writes the TRAINING and TEST sets of individuals used by the rest
of the pipeline: steps 1-4 learn from the training individuals only, and step 5 validates the
fine-tuned Evo 2 on the test individuals.

SPLIT (--split)
  bh26    (default) the published BH_26 split, stratified by population:
          https://data.haploblocks.org/BH_26/splits/  half_A.txt = train (1,252), half_B.txt = test (1,251)
  random  a new random split; --test-fraction (default 0.5) of the individuals go to the test set,
          stratified by 1000G population (from phenotypes_real.csv) unless --no-stratify.
  Only individuals present in the haplograph are written; IDs of the split that are not in the
  haplograph (about 45 for BH_26) are counted in split.json.

OUTPUT  (in --workdir, default haplograph/)
  raw/<chrom>/nodes.csv.gz        one row per haploblock cluster: id, high_dim_edge (the haploblock
                                  region), then one 0/1 column per individual (2,548 in 1000G)
  raw/<chrom>/edges.csv.gz        source, target, weight (co-occurring cluster pairs)   (--edges all)
  raw/<chrom>/edges_lift_above_threshold.csv.gz   same + lift, only lift >= 5         (--edges lift5)
  splits/train_individuals.txt    TRAINING set: one sample ID per line
  splits/test_individuals.txt     TEST set: one sample ID per line
  splits/split.json               split name, source, sizes, chromosomes, set ids
  (bh26 also keeps the original splits/half_A.txt and splits/half_B.txt)

USAGE
  python download_haplograph.py                               # whole genome, BH_26 split
  python download_haplograph.py --chroms chr21,chr22          # two chromosomes only
  python download_haplograph.py --chroms 21 --edges lift5     # small pre-filtered edge file
  python download_haplograph.py --split random --test-fraction 0.3 --seed 1
  python download_haplograph.py --test                        # synthetic data, no network

Sizes: edges.csv.gz is ~80 MB (chr21) to ~1 GB (large chromosomes) per chromosome; nodes.csv.gz is
~7-100 MB. Whole genome: plan for ~15 GB of disk.

REQUIRES  numpy, pandas (stdlib only for the download itself)
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from haplo_common import (ALL_CHROMS, BASE_URL, BH26_SPLITS, die, download, log, make_synthetic_haplograph,
                          parse_chroms, read_individual_list, set_id, test_workdir, write_individual_list)

EDGE_FILES = {"all": ["edges.csv.gz"], "lift5": ["edges_lift_above_threshold.csv.gz"],
              "both": ["edges.csv.gz", "edges_lift_above_threshold.csv.gz"]}


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


def population_of(workdir, individuals):
    """1000G population per individual (for stratifying a random split); None if unavailable."""
    import pandas as pd
    path = download(f"{BASE_URL}/phenotypes_real.csv", Path(workdir) / "raw" / "phenotypes_real.csv")
    ph = pd.read_csv(path, dtype=str)
    need = {"individual_id", "phenotype", "value"}
    if not need <= set(ph.columns):
        log(f"  warning: {path} lacks columns {sorted(need)}; split is not stratified")
        return None
    pop = ph[ph["phenotype"].str.lower() == "population"].set_index("individual_id")["value"]
    return [pop.get(i, "unknown") for i in individuals]


def random_split(individuals, test_fraction, seed, strata=None):
    rng = np.random.default_rng(seed)
    strata = strata if strata is not None else ["all"] * len(individuals)
    train, test = [], []
    for s in sorted(set(strata)):
        members = [x for x, g in zip(individuals, strata) if g == s]
        rng.shuffle(members)
        n_test = int(round(test_fraction * len(members)))
        test += members[:n_test]
        train += members[n_test:]
    return sorted(train), sorted(test)


def make_split(args, individuals):
    splits = Path(args.workdir) / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    present = set(individuals)
    info = {"split": args.split, "n_haplograph_individuals": len(individuals)}
    if args.split == "bh26":
        a = read_individual_list(download(f"{args.splits_url}/half_A.txt", splits / "half_A.txt"))
        b = read_individual_list(download(f"{args.splits_url}/half_B.txt", splits / "half_B.txt"))
        overlap = set(a) & set(b)
        if overlap:
            die(f"the BH_26 halves overlap ({len(overlap)} IDs, e.g. {sorted(overlap)[:3]})")
        train, test = [x for x in a if x in present], [x for x in b if x in present]
        info.update(name="half_A", source=args.splits_url, train_in_file=len(a), test_in_file=len(b))
    else:
        if not 0 < args.test_fraction < 1:
            die("--test-fraction must be between 0 and 1")
        strata = None if args.no_stratify else population_of(args.workdir, individuals)
        train, test = random_split(individuals, args.test_fraction, args.seed, strata)
        info.update(name=f"random{int(round(100 * args.test_fraction))}_seed{args.seed}",
                    source="random", test_fraction=args.test_fraction, seed=args.seed,
                    stratified=strata is not None)
    if not train or not test:
        die(f"empty split: {len(train)} train / {len(test)} test individuals found in the haplograph")
    write_individual_list(splits / "train_individuals.txt", train)
    write_individual_list(splits / "test_individuals.txt", test)
    info.update(train=len(train), test=len(test),
                haplograph_individuals_in_neither_set=len(present - set(train) - set(test)),
                train_set_id=set_id(train), test_set_id=set_id(test),
                train_file="train_individuals.txt", test_file="test_individuals.txt")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", help="comma-separated chromosomes, e.g. chr21,chr22 or 21,22 (default: all, chr1-chr22)")
    ap.add_argument("--workdir", default="haplograph", help="root folder for raw/ and splits/")
    ap.add_argument("--edges", choices=sorted(EDGE_FILES), default="all",
                    help="all = edges.csv.gz (needed for a train-only graph); lift5 = small pre-filtered file")
    ap.add_argument("--split", choices=["bh26", "random"], default="bh26")
    ap.add_argument("--splits-url", default=BH26_SPLITS, help="folder with half_A.txt / half_B.txt (bh26)")
    ap.add_argument("--test-fraction", type=float, default=0.5, help="random split: share of individuals in test")
    ap.add_argument("--seed", type=int, default=0, help="random split seed")
    ap.add_argument("--no-stratify", action="store_true", help="random split: do not stratify by population")
    ap.add_argument("--no-verify", action="store_true", help="skip the MD5 check against checksums.md5")
    ap.add_argument("--test", action="store_true",
                    help="self-test: write a small synthetic haplograph (no network) and split it")
    args = ap.parse_args(argv)

    chroms = parse_chroms(args.chroms) or ALL_CHROMS
    if args.test:
        args.workdir = str(test_workdir("download"))
        args.split, args.no_stratify, chroms = "random", True, ["chr21", "chr22"]
        make_synthetic_haplograph(Path(args.workdir) / "raw", chroms)
        log(f"[test] synthetic haplograph written to {args.workdir}/raw")
    else:
        unknown = [c for c in chroms if c not in ALL_CHROMS]
        if unknown:
            die(f"unknown chromosome(s) {unknown}; the haplograph has chr1-chr22 only")
        checksums = None if args.no_verify else load_checksums(args.workdir)
        log(f"downloading {len(chroms)} chromosome(s) into {args.workdir}/raw ({args.edges} edges)")
        for chrom in chroms:
            fetch_chromosome(args.workdir, chrom, ["nodes.csv.gz"] + EDGE_FILES[args.edges], checksums)

    individuals = None
    for chrom in chroms:                       # every chromosome must list the same individuals
        inds = haplograph_individuals(Path(args.workdir) / "raw" / chrom / "nodes.csv.gz")
        if individuals is None:
            individuals = inds
        elif set(inds) != set(individuals):
            die(f"{chrom}: individual columns differ from {chroms[0]}")

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
