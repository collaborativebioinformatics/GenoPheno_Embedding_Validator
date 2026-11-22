"""
GWAS step: download_gwas.py (catalog filtering) and gwas_enrichment in step 5.
Run:  python -m pytest part1_haplograph/tests -q
"""
import json
import zipfile

import pandas as pd

from conftest import run_validate, script

download_gwas = script("download_gwas")
COLS = download_gwas.COLS


def write_catalog(path, rows):
    df = pd.DataFrame(rows, columns=["CHR_ID", "CHR_POS", "SNPS", "P-VALUE", "MAPPED_TRAIT", "MAPPED_TRAIT_URI"])
    df["DISEASE/TRAIT"], df["PUBMEDID"] = df["MAPPED_TRAIT"], ["2"] + ["1"] * (len(df) - 1)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a.tsv", df[COLS].to_csv(sep="\t", index=False))


def test_catalog_filter_dedup_and_cleaning(tmp_path):
    """Trait regex and ontology IDs select rows (the ID also matches multi-trait rows); one row per
    position (lowest p); multi-locus, unplaced and MT rows are dropped; 23 becomes chrX; studies
    are counted before deduplication."""
    uri = "http://purl.obolibrary.org/obo/OBA_VT0001253"
    write_catalog(tmp_path / "c.zip", [
        ("21", "100", "rs1", "1E-9", "body height", uri),
        ("21", "100", "rs1", "1E-400", "body height", uri),           # same variant, p below float range
        ("21", "200", "rs2", "2E-6", "body height", uri),
        ("21;21", "300;400", "rs3;rs4", "1E-8", "body height", uri),  # multi-locus row
        ("", "", "rs5", "1E-8", "body height", uri),                  # not placed on GRCh38
        ("23", "500", "rs6", "1E-10", "body height", uri),
        ("MT", "50", "rs9", "1E-10", "body height", uri),
        ("21", "700", "rs8", "1E-9", "body height, body mass index", uri + ", http://www.ebi.ac.uk/efo/EFO_0004340"),
        ("21", "600", "rs7", "1E-10", "body mass index", "http://www.ebi.ac.uk/efo/EFO_0004340"),
    ])
    base = [("chr21", 100), ("chr21", 200), ("chrX", 500)]
    for flags, expected in ((["--trait", "^body height$"], base),
                            (["--trait-id", "OBA:VT0001253"], base[:2] + [("chr21", 700), ("chrX", 500)])):
        tsv = download_gwas.main(["--workdir", str(tmp_path), "--catalog", str(tmp_path / "c.zip"),
                                  "--label", "h"] + flags)
        df = pd.read_csv(tsv, sep="\t")
        assert list(zip(df["CHR"], df["POS"])) == expected
        assert df.loc[df["POS"] == 100, "P"].item() == 0
        assert json.loads((tmp_path / "gwas" / "h.json").read_text())["studies"] == 2


def test_hit_list_is_not_mistaken_for_summary_statistics(pipeline, tmp_path):
    """A catalog hit list whose rows are mostly suggestive (5e-8 <= p < 1e-5) must use the
    region-length background, not the 'all tested SNPs' one."""
    gwas = tmp_path / "hits.tsv"
    rows = ["CHR\tPOS\tSNP\tP"] + [f"chr21\t{1_000_000 + 7_000 * i}\trs{i}\t{1e-9 if i < 10 else 1e-6}"
                                   for i in range(60)]
    gwas.write_text("\n".join(rows) + "\n")
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"],
                             "--task", f"gwas_enrichment gwas={gwas} permutations=20"], tmp_path / "v")
    assert tasks["gwas_enrichment"]["summary"]["file_type"] == "hits only"


def test_planted_hotspot_is_found(pipeline, tmp_path):
    """download_gwas --test plants 25 hits in one 1 kb window; it must be the top enriched haploblock."""
    tsv = download_gwas.main(["--test"])
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"],
                             "--task", f"gwas_enrichment gwas={tsv} permutations=20"], tmp_path / "v")
    s = tasks["gwas_enrichment"]["summary"]
    chrom, a, b = download_gwas.TEST_HOTSPOT
    region = s["top_region"]
    start, end = map(int, region.rsplit("_", 1)[1].split("-"))
    assert region.startswith(chrom + "_") and start <= b and end >= a and s["regions_enriched_q<0.05"] >= 1


def test_missing_gwas_file_names_the_download_step(pipeline, tmp_path):
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"],
                             "--task", "gwas_enrichment gwas=no/such/file.tsv"], tmp_path / "v")
    assert "download_gwas.py" in tasks["gwas_enrichment"].get("error", "")
