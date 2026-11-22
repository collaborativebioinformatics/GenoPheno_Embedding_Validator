"""
Regression tests for confirmed part 1 issues (spec vs code, leakage, security).

All issues below are fixed; each test guards one of them. (While an issue is open, its test is
marked xfail(strict=True), and the marker is removed when the fix lands.)
Run:  python -m pytest part1_haplograph/tests -q
"""
import json

import numpy as np
import pytest

from conftest import STEP3_FAST, run_validate, script
from haplo_common import make_synthetic_haplograph, set_id, write_individual_list


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class _WritesMarker:
    """Unpickling this object calls open(path, "w"), i.e. runs code chosen by the file's author."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_model_load_does_not_unpickle_arbitrary_objects(tmp_path):
    """A malicious model file must not execute code when step 6 reads it."""
    import torch
    marker = tmp_path / "PWNED"
    evil = tmp_path / "evil.pt"
    torch.save({"mode": "embeddings", "x": _WritesMarker(str(marker))}, evil)
    v = script("6_validate_evo2")
    try:
        v.Model(str(evil)).meta
    except (SystemExit, Exception):
        pass                                  # refusing the file is the correct behaviour
    assert not marker.exists(), "loading the model file executed code from it"


def test_step4_checkpoints_load_with_weights_only(pipeline, tmp_path):
    """Guards the fix above: 4a and 4b checkpoints must stay loadable with weights_only=True,
    so step 6 never needs the unsafe full unpickler."""
    import torch
    torch.load(pipeline["model"], map_location="cpu", weights_only=True)
    m4b = script("4b_finetune_evo2_hash")
    tsv = tmp_path / "hash_height.tsv"
    m4b.write_test_tsv(str(tsv), n=20, bits=16)
    out = tmp_path / "hash.pt"
    m4b.main(["--hash-pheno", str(tsv), "--mock-evo2", "--device", "cpu", "--mock-hidden", "32",
              "--mock-layers", "2", "--train-blocks", "1", "--batch", "4", "--hash-window", "16",
              "--eval-every", "1000", "--out", str(out)])
    torch.load(out, map_location="cpu", weights_only=True)


def test_nemotron_scrub_removes_relative_and_windows_paths():
    """DOCUMENTATION section 6: file paths are removed before anything is sent to Nemotron."""
    v = script("6_validate_evo2")
    for path, secret in [("patient_cohort_X/model.pt", "patient_cohort_X"),
                         ("haplograph/secret_study/evo2_next_gnn.pt", "secret_study"),
                         (r"C:\Users\alice\study\model.pt", "alice")]:
        assert secret not in v._scrub(path), f"{path!r} -> {v._scrub(path)!r}"


def test_nemotron_key_not_sent_over_plain_http(tmp_path, monkeypatch):
    """The NVIDIA API key must not travel unencrypted."""
    import argparse
    import urllib.request
    v = script("6_validate_evo2")
    calls = []
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-test-key")

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise OSError("offline test")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    args = argparse.Namespace(nemotron_model="m", nemotron_url="http://example.invalid/v1")
    v.nemotron_summary(args, [{"name": "t", "status": "ok", "summary": {"x": 1}}], str(tmp_path / "n.md"))
    assert not calls, "the API key was sent to an http:// endpoint"


# ---------------------------------------------------------------------------
# Train/test leakage: each step must use the graph's training set
# ---------------------------------------------------------------------------

def test_step3_refuses_training_set_other_than_graphs(graph_copy):
    """README: 'Each step checks that it uses the same training set as the step before and stops if not.'"""
    with pytest.raises(SystemExit):
        script("3_node_embeddings").main(["--graph-dir", str(graph_copy), "--force",
                                          "--train-individuals", str(graph_copy / "test_individuals.txt")]
                                         + STEP3_FAST)


def test_step4_refuses_training_set_other_than_graphs(graph_copy, pipeline):
    """Step 4 must not fine-tune on test individuals even if the embeddings were (wrongly) made from them."""
    info_p = graph_copy / "node_emb_gnn.json"
    info = json.loads(info_p.read_text())
    info["train_set_id"] = set_id(pipeline["test"])          # embeddings 'made from' the test set
    info_p.write_text(json.dumps(info))
    from conftest import STEP4_FAST
    with pytest.raises(SystemExit):
        script("4a_finetune_evo2_embeddings").main(
            ["--graph-dir", str(graph_copy), "--out", str(graph_copy / "leak.pt"),
             "--train-individuals", str(graph_copy / "test_individuals.txt")] + STEP4_FAST)


def test_embedding_cooccurrence_not_held_out_when_embeddings_saw_test_set(graph_copy, pipeline, tmp_path):
    """If the embeddings were trained on test individuals, the AUC must not be reported as held-out."""
    info_p = graph_copy / "node_emb_gnn.json"
    info = json.loads(info_p.read_text())
    info["train_set_id"] = set_id(pipeline["test"])
    info_p.write_text(json.dumps(info))
    _, tasks = run_validate(["--model", graph_copy / "evo2_mock_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "embedding_cooccurrence min_lift=1.5 min_support=3 min_expected=1 "
                                       "pairs=3000 bootstrap=0"], tmp_path / "v")
    t = tasks["embedding_cooccurrence"]
    assert t["status"] == "ok" and t["summary"].get("status") != "skipped"
    assert t["summary"]["held_out"] == "no"


# ---------------------------------------------------------------------------
# Step 6 must refuse (or at least report) a model from another graph build / other embeddings
# ---------------------------------------------------------------------------

def _stale_flagged(task):
    text = " ".join(task.get("notes", [])) + " " + task.get("error", "")
    return task["status"] != "ok" or "graph" in text.lower() and ("build" in text.lower() or "graph_id" in text)


def test_step5_flags_model_from_other_graph_build(graph_copy, tmp_path):
    """DOCUMENTATION section 5: each step stores the graph build id and the next step checks it."""
    p = graph_copy / "graph_info.json"
    info = json.loads(p.read_text())
    info["min_lift"] = 3.0                                  # what a --force rebuild with other settings does
    p.write_text(json.dumps(info))
    _, tasks = run_validate(["--model", graph_copy / "evo2_mock_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "test_set_performance max_examples=10 train_examples=5 bootstrap=0 "
                                       "device=cpu"], tmp_path / "v")
    assert _stale_flagged(tasks["test_set_performance"])


def test_step5_reports_changed_embeddings(graph_copy, tmp_path):
    """Evaluating a model on embeddings it was not trained with must fail or be recorded in the report."""
    Z = np.load(graph_copy / "node_emb_gnn.npy")
    np.save(graph_copy / "node_emb_gnn.npy", Z[::-1].copy())
    _, tasks = run_validate(["--model", graph_copy / "evo2_mock_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "test_set_performance max_examples=10 train_examples=5 bootstrap=0 "
                                       "device=cpu"], tmp_path / "v")
    t = tasks["test_set_performance"]
    assert t["status"] != "ok" or any("node_emb" in n or "embedding" in n.lower() and "changed" in n
                                      for n in t.get("notes", []))


# ---------------------------------------------------------------------------
# Spec vs code: step 6 task selection
# ---------------------------------------------------------------------------

def test_gwas_shortcut_keeps_default_tasks(pipeline, tmp_path):
    """README step-6 example `--splits ... --gwas height_hg38.tsv` must still run the default checks."""
    gwas = tmp_path / "gwas.tsv"
    rows = ["CHR\tPOS\tRSID\tP"] + [f"chr21\t{1_000_000 + 10_000 * i}\trs{i}\t{1e-9 if i < 5 else 0.5}"
                                     for i in range(40)]
    gwas.write_text("\n".join(rows) + "\n")
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"], "--gwas", gwas],
                            tmp_path / "v")
    assert "gwas_enrichment" in tasks
    assert {"model_info", "test_set_performance", "cluster_consistency"} <= set(tasks), sorted(tasks)


# ---------------------------------------------------------------------------
# Caches that ignore their inputs
# ---------------------------------------------------------------------------

def test_step1_cache_follows_raw_dir(tmp_path):
    """A graph built with --raw-dir B must contain B's clusters, not those cached from raw/."""
    make_synthetic_haplograph(tmp_path / "raw", ("chr21",), n_ind=30, n_regions=12, seed=0)
    make_synthetic_haplograph(tmp_path / "rawB", ("chr21",), n_ind=30, n_regions=4, seed=1)
    b1 = script("2_build_graph")
    b1.main(["--workdir", str(tmp_path), "--no-split"])
    b1.main(["--workdir", str(tmp_path), "--raw-dir", str(tmp_path / "rawB"), "--no-split",
             "--out", str(tmp_path / "graphB")])
    import pandas as pd
    built = pd.read_csv(tmp_path / "graphB" / "nodes.csv.gz")
    expected = pd.read_csv(tmp_path / "rawB" / "chr21" / "nodes.csv.gz", usecols=["id"])
    assert sorted(built["id"]) == sorted(expected["id"])


def test_step3_recomputes_when_settings_change(graph_copy):
    """Rerunning step 3 with another --dim must not leave embeddings of the old size."""
    args = [a for a in STEP3_FAST]
    args[args.index("--dim") + 1] = "12"
    script("3_node_embeddings").main(["--graph-dir", str(graph_copy)] + args)
    assert np.load(graph_copy / "node_emb_gnn.npy").shape[1] == 12


def test_step0_new_splits_url_is_used(tmp_path):
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


# ---------------------------------------------------------------------------
# Rare-cluster merging (--min-support)
# ---------------------------------------------------------------------------

def test_min_support_merging(pipeline):
    """Merged clusters: fewer tokens, valid local indices, mean embeddings, no repeated tokens."""
    from evo2_model import GenomeData
    G = str(pipeline["G"])
    every = list(range(len(pipeline["train"]) + len(pipeline["test"])))
    base, merged = GenomeData(G, "gnn", None, 0, every), GenomeData(G, "gnn", None, 20, every)
    assert merged.rare.any()
    for c in base.chroms:
        cands = merged.candidates(c)
        assert len(cands) < len(base.candidates(c))
        for i in range(base.by_ind.shape[0]):
            nodes, local = merged.tokens(c, i)
            assert ((local >= 0) & (local < len(cands))).all() and (cands[local] == nodes).all()
            assert not (nodes[1:] == nodes[:-1]).any()
    rep = merged.canon[merged.rare]
    for r in np.unique(rep):
        members = np.where(merged.rare & (merged.canon == r))[0]
        assert np.allclose(merged.Z[r], base.Z[members].mean(0), atol=1e-5)
