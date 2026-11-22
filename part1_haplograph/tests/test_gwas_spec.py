"""
Spec-test regressions for the GWAS step (5_download_gwas.py and gwas_enrichment in step 6).
Each test guards a confirmed issue; tests for unfixed issues fail until the fix lands.
Run:  python -m pytest part1_haplograph/tests -q
"""
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from conftest import run_validate, script

download_gwas = script("5_download_gwas")


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.zip"
    download_gwas.make_test_catalog(path)
    return path


def run_download(tmp_path, catalog, *flags):
    return download_gwas.main(["--workdir", str(tmp_path / "wd"), "--catalog", str(catalog)] + list(flags))


def test_label_cannot_write_outside_the_gwas_folder(tmp_path, catalog):
    """--label is used as a file name; '../' must not place outputs outside <workdir>/gwas/."""
    try:
        tsv = run_download(tmp_path, catalog, "--trait-id", "EFO_0000001", "--label", "../../escaped")
    except SystemExit:
        tsv = None
    assert not (tmp_path / "escaped.tsv").exists() and not (tmp_path / "escaped.json").exists()
    if tsv is not None:
        assert (tmp_path / "wd" / "gwas").resolve() in Path(tsv).resolve().parents


@pytest.mark.parametrize("flags", [["--trait", "("], ["--trait", "synthetic", "--label", "a/b"]])
def test_bad_trait_regex_or_label_stops_with_a_message(tmp_path, catalog, flags):
    """An invalid --trait regular expression or a label with '/' must stop with a readable
    'error: ...' (die), not a re.error / OSError traceback."""
    with pytest.raises(SystemExit, match="error"):
        run_download(tmp_path, catalog, *flags)


def test_unreadable_catalog_stops_with_a_message(tmp_path):
    """A file that is not a zip, or a zip without the catalog columns (format change), must stop
    with a readable 'error: ...' instead of BadZipFile / pandas ValueError tracebacks."""
    bad = tmp_path / "notzip.zip"
    bad.write_text("not a zip")
    nocols = tmp_path / "nocols.zip"
    with zipfile.ZipFile(nocols, "w") as z:
        z.writestr("a.tsv", "A\tB\n1\t2\n")
    for path in (bad, nocols):
        with pytest.raises(SystemExit, match="error"):
            run_download(tmp_path, path, "--trait", "x")


def _chr21_regions(graph_dir):
    ids = pd.read_csv(graph_dir / "nodes.csv.gz", usecols=["id"])["id"]
    reg = ids.str.extract(r"^(chr\w+)_(\d+)-(\d+)_").drop_duplicates()
    reg.columns = ["chrom", "start", "end"]
    reg = reg[reg["chrom"] == "chr21"].astype({"start": int, "end": int}).sort_values("start")
    return reg.reset_index(drop=True)


def test_window_keeps_hits_in_their_own_haploblock(pipeline, tmp_path):
    """With window=W > 0, a hit inside haploblock i but within W bp of haploblock i+1 was assigned
    to i+1 only (assign_to_regions takes the last region whose extended start <= pos), so hits
    moved to the neighbouring haploblock. A hit must still count for the haploblock that contains it."""
    reg = _chr21_regions(pipeline["G"])
    r = reg.iloc[2]
    name = f"chr21_{r.start}-{r.end}"
    rows = ["CHR\tPOS\tSNP\tP"] + [f"chr21\t{r.end - 50 - i}\trs{i}\t1e-9" for i in range(5)]
    gwas = tmp_path / "edge_hits.tsv"
    gwas.write_text("\n".join(rows) + "\n")
    run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"],
                  "--task", f"gwas_enrichment gwas={gwas} window=1000 mode=hits permutations=5"], tmp_path / "v")
    out = pd.read_csv(tmp_path / "v" / "gwas_enrichment_edge_hits_regions.csv.gz")
    assert out.loc[out["region"] == name, "n_hits"].item() == 5


def test_no_testable_snps_does_not_report_a_top_region(pipeline, tmp_path):
    """If every SNP falls outside the haploblock span (e.g. wrong genome build), the task must not
    succeed with an arbitrary 'top_region' (it reported the first haploblock, all p = 1)."""
    gwas = tmp_path / "outside.tsv"
    gwas.write_text("CHR\tPOS\tSNP\tP\nchr21\t90000000\trs1\t1e-9\nchr21\t90000500\trs2\t1e-9\n")
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"],
                             "--task", f"gwas_enrichment gwas={gwas} permutations=5"], tmp_path / "v")
    t = tasks["gwas_enrichment"]
    assert t["status"] != "ok" or t["summary"]["top_region"] is None
