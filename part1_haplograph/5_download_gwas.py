"""
Step 5 - download GWAS hits for one trait, for the gwas_enrichment task of step 6.

Downloads the NHGRI-EBI GWAS Catalog (all associations, ontology-annotated, about 75 MB zipped,
cached in <workdir>/gwas/catalog/), keeps the associations of one trait (--trait regular expression on the trait
names, --trait-id ontology IDs, or both: the union) and writes one row per variant (lowest p-value).
Catalog positions are GRCh38, the same build as the haplograph regions. The catalog lists reported
hits only (p < 1e-5), so step 6 compares hits per haploblock with the haploblock's length.

OUTPUT (in --workdir, default haplograph/)
  gwas/<label>.tsv     CHR, POS, SNP, P, PUBMEDID   (one row per chromosome position)
  gwas/<label>.json    source, catalog file sha1, filter, row counts

USAGE
  python 5_download_gwas.py --trait "^body height$" --label height     # trait name (regular expression)
  python 5_download_gwas.py --trait-id OBA_VT0001253 --label height_id   # by ontology ID (+ multi-trait rows)
  python 5_download_gwas.py --test                                    # synthetic catalog, no network
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --gwas haplograph/gwas/height.tsv
"""
import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from haplo_common import die, download, is_url, log, test_workdir

CATALOG_URL = ("https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
               "gwas-catalog-associations_ontology-annotated-full.zip")
COLS = ["PUBMEDID", "DISEASE/TRAIT", "CHR_ID", "CHR_POS", "SNPS", "P-VALUE", "MAPPED_TRAIT", "MAPPED_TRAIT_URI"]
TEST_HOTSPOT = ("chr21", 1_100_000, 1_116_000)      # default planted window of the --test catalog


def sha1(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def make_test_catalog(path, hotspot=TEST_HOTSPOT, seed=0):
    """Catalog-format zip on the synthetic haplograph's coordinates (chr21 from 1 Mb, chr22 from
    2 Mb): 15 background hits per chromosome, 10 evenly spaced hits in `hotspot` (chrom, start,
    end), and 40 hits per chromosome of another trait."""
    rng = np.random.default_rng(seed)
    rows = []
    for chrom, lo in (("21", 1_000_000), ("22", 2_000_000)):
        rows += [(chrom, p, "synthetic trait") for p in rng.integers(lo, lo + 600_000, 15)]
        rows += [(chrom, p, "other trait") for p in rng.integers(lo, lo + 600_000, 40)]
    c, a, b = hotspot
    rows += [(c[3:], int(p), "synthetic trait") for p in np.linspace(a, b, 10)]
    df = pd.DataFrame(rows, columns=["CHR_ID", "CHR_POS", "MAPPED_TRAIT"])
    df["DISEASE/TRAIT"] = df["MAPPED_TRAIT"]
    df["MAPPED_TRAIT_URI"] = np.where(df["MAPPED_TRAIT"] == "synthetic trait",
                                      "http://www.ebi.ac.uk/efo/EFO_0000001", "http://www.ebi.ac.uk/efo/EFO_0000002")
    df["SNPS"] = [f"rs{i + 1}" for i in range(len(df))]
    df["P-VALUE"] = [f"{x:.1E}" for x in 10.0 ** -rng.uniform(8, 20, len(df))]
    df["PUBMEDID"] = "1"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("associations.tsv", df[COLS].to_csv(sep="\t", index=False))


def read_catalog(path, trait_ids, trait):
    """Rows of the catalog zip matching either filter (union), with numeric CHR/POS/P; rows on
    several loci or not placed on GRCh38 are dropped. Returns (rows, matching rows, studies)."""
    keep = []
    try:
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if not n.endswith("/")]
            if len(members) != 1:
                die(f"{path}: expected one file in the zip, found {members[:5]}")
            with z.open(members[0]) as fh:
                for df in pd.read_csv(fh, sep="\t", usecols=COLS, dtype=str, chunksize=200_000):
                    m = pd.Series(False, index=df.index)
                    if trait_ids:
                        uri_ids = df["MAPPED_TRAIT_URI"].fillna("").str.findall(r"([^/,\s]+)\s*(?:,|$)")
                        m |= uri_ids.apply(lambda ids: bool(trait_ids.intersection(ids)))
                    if trait:
                        m |= (df["MAPPED_TRAIT"].fillna("").str.contains(trait, case=False)
                              | df["DISEASE/TRAIT"].fillna("").str.contains(trait, case=False))
                    keep.append(df[m])
    except (zipfile.BadZipFile, ValueError) as e:
        die(f"{path} is not a GWAS Catalog associations zip ({e}); check the catalog downloads page")
    df = pd.concat(keep, ignore_index=True)
    n = len(df)
    out = pd.DataFrame({"CHR": "chr" + df["CHR_ID"].str.strip().replace({"23": "X", "24": "Y"}),
                        "POS": pd.to_numeric(df["CHR_POS"], errors="coerce"),
                        "SNP": df["SNPS"], "P": pd.to_numeric(df["P-VALUE"], errors="coerce"),
                        "PUBMEDID": df["PUBMEDID"]})
    out = out[out["CHR"].str.fullmatch(r"chr(\d+|X|Y)") & out["POS"].notna() & out["P"].between(0, 1)]
    return out.assign(POS=out["POS"].astype(np.int64)), n, int(df["PUBMEDID"].nunique())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", help="regular expression on MAPPED_TRAIT / DISEASE/TRAIT (case-insensitive)")
    ap.add_argument("--trait-id", help="comma-separated trait ontology IDs (last part of MAPPED_TRAIT_URI, e.g. "
                    "OBA_VT0001253 for body height); also matches rows mapped to several traits")
    ap.add_argument("--label", help="output name (default: from --trait-id / --trait)")
    ap.add_argument("--workdir", default="haplograph", help="root folder; output in <workdir>/gwas/")
    ap.add_argument("--catalog", default=CATALOG_URL, help="catalog zip: URL or local file")
    ap.add_argument("--force", action="store_true", help="download the catalog again")
    ap.add_argument("--test", action="store_true", help="self-test: synthetic catalog, no network")
    args = ap.parse_args(argv)

    if args.test:
        args.workdir = str(test_workdir("5_download_gwas"))
        args.catalog = str(Path(args.workdir) / "catalog.zip")
        args.trait_id, args.trait, args.label = "EFO_0000001", None, "synthetic"
        make_test_catalog(args.catalog)
    if not (args.trait_id or args.trait):
        die('give --trait and/or --trait-id (e.g. --trait "^body height$")')
    if args.trait:
        try:
            re.compile(args.trait)
        except re.error as e:
            die(f"--trait {args.trait!r} is not a valid regular expression ({e})")
    trait_ids = {e.strip().replace(":", "_") for e in (args.trait_id or "").split(",") if e.strip()}
    label = args.label or re.sub(r"[^A-Za-z0-9._-]+", "_", args.trait_id or args.trait).strip("._")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", label):
        die(f"--label {label!r}: use letters, digits, '.', '_' and '-' only (it is a file name in {args.workdir}/gwas/)")
    out_dir = Path(args.workdir) / "gwas"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = args.catalog
    if is_url(path):
        if not path.startswith("https://"):
            die("--catalog URL must use https")
        path = Path(out_dir) / "catalog" / Path(path).name
        if args.force and path.exists():
            path.unlink()
        path = download(args.catalog, path, hint="check the GWAS Catalog downloads page for the current file name")
    elif not Path(path).exists():
        die(f"catalog file not found: {path}")

    df, n_rows, n_studies = read_catalog(path, trait_ids, args.trait)
    if df.empty:
        die(f"no catalog associations match --trait {args.trait!r} / --trait-id {args.trait_id!r}")
    n_placed = len(df)
    df = df.sort_values(["P", "SNP"], kind="stable").drop_duplicates(["CHR", "POS"]).sort_values(["CHR", "POS"])
    tsv = out_dir / f"{label}.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    info = {"source": args.catalog, "catalog_sha1": sha1(path),
            "catalog_file_date": time.strftime("%Y-%m-%d", time.localtime(Path(path).stat().st_mtime)),
            "genome_build": "GRCh38", "trait": args.trait, "trait_ids": sorted(trait_ids),
            "matching_associations": n_rows, "dropped_multi_locus_or_unplaced": n_rows - n_placed,
            "variants": len(df), "studies": n_studies}
    with open(out_dir / f"{label}.json", "w") as f:
        json.dump(info, f, indent=2)
    log(f"{n_rows:,} associations from {n_studies:,} studies -> {len(df):,} variants -> {tsv}")
    if args.test:                                   # the planted hotspot must survive the filter
        c, a, b = TEST_HOTSPOT
        if ((df["CHR"] == c) & df["POS"].between(a, b)).sum() < 10 or (df["P"] > 1e-5).any():
            die("self-test: the synthetic catalog was not filtered as expected")
        log("self-test checks passed")
    return str(tsv)


if __name__ == "__main__":
    main()
