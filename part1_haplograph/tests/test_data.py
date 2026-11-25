"""
Steps 1-2: phased haplotypes, split, populations and related pairs, graph and haplotype outputs.
Run:  python -m pytest part1_haplograph/tests -q
"""
import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from conftest import script, synthetic_workdir
from haplo_common import make_synthetic_haplograph, read_graph_individuals, write_individual_list


def test_haplotypes_reproduce_carriers_and_junctions_match_counts(pipeline):
    """Each individual's two haplotype rows reproduce its carriers exactly; the junction counts of
    every haploblock sum to that haploblock's training-haplotype cluster counts."""
    G = pipeline["G"]
    H = np.load(G / "haplotypes.npz")["node"]
    C = sp.load_npz(G / "carriers.npz").toarray().astype(bool)
    U = np.zeros_like(C)
    for i in range(C.shape[1]):
        U[H[2 * i], i] = U[H[2 * i + 1], i] = True
    assert (U == C).all()
    inds = read_graph_individuals(G)
    train = set(pipeline["train"])
    rows = np.array([r for i, x in enumerate(inds) if x in train for r in (2 * i, 2 * i + 1)])
    j = np.load(G / "junctions.npz")
    for k in np.unique(j["block"]):
        m = j["block"] == k
        sums = pd.Series(j["count"][m]).groupby(j["a"][m]).sum().to_dict()
        vals, cnt = np.unique(H[rows, k], return_counts=True)
        assert sums == dict(zip(vals.tolist(), cnt.tolist()))


def test_haplotype_lift_pairs_come_from_training_haplotypes_only(pipeline):
    """--lift haplotype (the default with haplotype data) never takes candidate pairs from a file made
    from all individuals; every edge joins clusters seen together on >= --min-pair-count training haplotypes."""
    G = pipeline["G"]
    info = json.loads((G / "graph_info.json").read_text())
    assert info["lift"] == "haplotype" and info["pair_source"] == "train_haplotypes"
    H = np.load(G / "haplotypes.npz")["node"]
    inds, train = read_graph_individuals(G), set(pipeline["train"])
    rows = [r for i, x in enumerate(inds) if x in train for r in (2 * i, 2 * i + 1)]
    A = sp.triu(sp.load_npz(G / "graph.npz")).tocoo()
    together = {(a, b) for h in H[rows] for a in h for b in h}
    assert all((a, b) in together for a, b in zip(A.row, A.col))


def test_mismatching_haplotypes_stop_the_build(tmp_path):
    """A haplotype table that does not reproduce nodes.csv.gz must be a hard error, not a warning."""
    synthetic_workdir(tmp_path, ("chr21",), n_ind=20)
    f = tmp_path / "raw" / "chr21" / "haplotypes.csv.gz"
    t = pd.read_csv(f, index_col=0)
    col = t.columns[3]
    t.iloc[0, 3] = t[col].max() if t.iloc[0, 3] != t[col].max() else t[col].min()
    t.to_csv(f, compression="gzip")
    with pytest.raises(SystemExit, match="do not reproduce the carriers"):
        script("2_build_graph").main(["--workdir", str(tmp_path), "--force"])


def test_step1_writes_populations_and_keeps_families_together(tmp_path):
    """--split random keeps related individuals (1000 Genomes pedigree) on one side and records the
    superpopulation counts per side; populations are never a model input."""
    dl = script("1_download_haplograph")
    wd = tmp_path / "wd"
    inds = make_synthetic_haplograph(wd / "raw", ("chr21",), n_ind=20, seed=0)
    dl.main(["--workdir", str(wd), "--chroms", "chr21", "--no-verify", "--split", "random", "--haplotypes"])
    pairs = pd.read_csv(wd / "splits" / "related_pairs.tsv", sep="\t")
    assert len(pairs) == 4 and not pairs["across_split"].any()
    info = json.loads((wd / "splits" / "split.json").read_text())
    assert sum(info["superpopulations"]["train"].values()) + sum(info["superpopulations"]["test"].values()) == len(inds)
    assert (wd / "raw" / "chr21" / "haplotypes.csv.gz").exists() and not (wd / "raw" / "chr21" / "haplotype_hashes").exists()


def test_step1_new_splits_url_is_used(tmp_path):
    """split.json's 'source' must describe the files actually used."""
    inds = make_synthetic_haplograph(tmp_path / "wd" / "raw", ("chr21",), n_ind=20, seed=0)
    for name, (a, b) in {"A": (inds[:10], inds[10:]), "B": (inds[10:], inds[:10])}.items():
        d = tmp_path / name
        d.mkdir()
        write_individual_list(d / "half_A.txt", a)
        write_individual_list(d / "half_B.txt", b)
    dl = script("1_download_haplograph")
    common = ["--workdir", str(tmp_path / "wd"), "--chroms", "chr21", "--no-verify"]
    dl.main(common + ["--splits-url", (tmp_path / "A").as_uri()])
    dl.main(common + ["--splits-url", (tmp_path / "B").as_uri()])
    train = (tmp_path / "wd" / "splits" / "train_individuals.txt").read_text().split()
    assert sorted(train) == sorted(inds[10:])


def test_step2_cache_follows_raw_dir(tmp_path):
    """A graph built with --raw-dir B must contain B's clusters, not those cached from raw/."""
    make_synthetic_haplograph(tmp_path / "raw", ("chr21",), n_ind=30, n_regions=12, seed=0)
    make_synthetic_haplograph(tmp_path / "rawB", ("chr21",), n_ind=30, n_regions=4, seed=1)
    b2 = script("2_build_graph")
    b2.main(["--workdir", str(tmp_path), "--no-split", "--lift", "individual"])
    b2.main(["--workdir", str(tmp_path), "--raw-dir", str(tmp_path / "rawB"), "--no-split", "--lift", "individual",
             "--out", str(tmp_path / "graphB")])
    built = pd.read_csv(tmp_path / "graphB" / "nodes.csv.gz")
    expected = pd.read_csv(tmp_path / "rawB" / "chr21" / "nodes.csv.gz", usecols=["id"])
    assert sorted(built["id"]) == sorted(expected["id"])


def test_step2_missing_chromosome_hint_keeps_the_workdir(pipeline, tmp_path):
    """The 'download it first' hint must carry the same --workdir, or the suggested command downloads elsewhere."""
    wd = tmp_path / "custom_workdir"
    shutil.copytree(pipeline["wd"] / "splits", wd / "splits")
    shutil.copytree(pipeline["wd"] / "raw" / "chr21", wd / "raw" / "chr21")
    with pytest.raises(SystemExit) as e:
        script("2_build_graph").main(["--workdir", str(wd), "--chroms", "chr22"])
    msg = str(e.value)
    assert "1_download_haplograph.py" in msg and "--workdir" in msg and str(wd) in msg


def test_synthetic_hash_files_decode_like_the_server(tmp_path):
    """The per-haploblock files carry the cluster index in the last 20 bits of HASH; node N = index + 1."""
    make_synthetic_haplograph(tmp_path / "raw", ("chr21",), n_ind=10, seed=0)
    f = sorted((tmp_path / "raw" / "chr21" / "haplotype_hashes").iterdir())[0]
    df = pd.read_csv(f, sep="\t", dtype=str)
    region = "chr21_" + f.name.split("_individual_hashes_")[1][:-4]
    ids = set(pd.read_csv(tmp_path / "raw" / "chr21" / "nodes.csv.gz", usecols=["id"])["id"])
    assert {f"{region}_cluster{int(h[-20:], 2) + 1}" for h in df["HASH"]} <= ids
    with gzip.open(tmp_path / "raw" / "chr21" / "nodes.csv.gz", "rt") as fh:
        assert fh.readline().startswith("id,high_dim_edge,")
