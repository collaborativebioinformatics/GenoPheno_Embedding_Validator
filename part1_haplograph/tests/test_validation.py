"""
Step 6: model loading safety, leakage checks, baselines and the model-querying tasks.
Run:  python -m pytest part1_haplograph/tests -q
"""
import csv
import json
import re

import numpy as np
import pytest
import torch

from conftest import PART1, SMALL, STEP4_FAST, run_validate, script
from haplo_baselines import BlockBaselines
from haplo_common import set_id


# ---- security ----------------------------------------------------------------

class _WritesMarker:
    """Unpickling this object calls open(path, "w"), i.e. runs code chosen by the file's author."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_model_load_does_not_unpickle_arbitrary_objects(tmp_path):
    marker = tmp_path / "PWNED"
    evil = tmp_path / "evil.pt"
    torch.save({"mode": "embeddings", "x": _WritesMarker(str(marker))}, evil)
    try:
        script("6_validate_evo2").Model(str(evil)).meta
    except (SystemExit, Exception):
        pass                                  # refusing the file is the correct behaviour
    assert not marker.exists(), "loading the model file executed code from it"


def test_nemotron_scrub_removes_relative_and_windows_paths():
    v = script("6_validate_evo2")
    for path, secret in [("patient_cohort_X/model.pt", "patient_cohort_X"),
                         ("haplograph/secret_study/evo2_next_gnn.pt", "secret_study"),
                         (r"C:\Users\alice\study\model.pt", "alice")]:
        assert secret not in v._scrub(path), f"{path!r} -> {v._scrub(path)!r}"


def test_nemotron_key_not_sent_over_plain_http(tmp_path, monkeypatch):
    import argparse
    import urllib.request
    v = script("6_validate_evo2")
    calls = []
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-test-key")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: calls.append(req))
    args = argparse.Namespace(nemotron_model="m", nemotron_url="http://example.invalid/v1")
    v.nemotron_summary(args, [{"name": "t", "status": "ok", "summary": {"x": 1}}], str(tmp_path / "n.md"))
    assert not calls, "the API key was sent to an http:// endpoint"


# ---- leakage and stale inputs ------------------------------------------------------

def test_embedding_cooccurrence_not_held_out_when_embeddings_saw_test_set(graph_copy, pipeline, tmp_path):
    info_p = graph_copy / "node_emb_gnn.json"
    info = json.loads(info_p.read_text())
    info["train_set_id"] = set_id(pipeline["test"])
    info_p.write_text(json.dumps(info))
    _, tasks = run_validate(["--model", graph_copy / "small_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "embedding_cooccurrence min_lift=1.5 min_support=3 min_expected=1 "
                                       "pairs=3000 bootstrap=20"], tmp_path / "v")
    s = tasks["embedding_cooccurrence"]["summary"]
    assert s["held_out"] == "no" and s["labels"] == "haplotype"
    assert "auc_minus_carrier_svd_gnn" in s and "auc_minus_carrier_svd_95ci_gnn" in s


def test_step6_refuses_a_model_from_another_graph_build(graph_copy, tmp_path):
    p = graph_copy / "graph_info.json"
    info = json.loads(p.read_text())
    info["min_lift"] = 3.0                                  # what a --force rebuild with other settings does
    p.write_text(json.dumps(info))
    _, tasks = run_validate(["--model", graph_copy / "small_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "test_set_performance max_examples=10 train_examples=5 bootstrap=0 device=cpu"],
                            tmp_path / "v")
    t = tasks["test_set_performance"]
    assert t["status"] != "ok" and "graph" in t["error"]


def test_step6_refuses_changed_embeddings(graph_copy, tmp_path):
    Z = np.load(graph_copy / "node_emb_gnn.npy")
    np.save(graph_copy / "node_emb_gnn.npy", Z[::-1].copy())
    _, tasks = run_validate(["--model", graph_copy / "small_next_gnn.pt", "--graph-dir", graph_copy,
                             "--task", "test_set_performance max_examples=10 train_examples=5 bootstrap=0 device=cpu"],
                            tmp_path / "v")
    t = tasks["test_set_performance"]
    assert t["status"] != "ok" and "node_emb" in t["error"]


def test_splits_none_train_set_is_models_training_set(pipeline, tmp_path):
    meta = torch.load(pipeline["model"], weights_only=True)
    run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"], "--splits", "none",
                  "--task", "model_info"], tmp_path / "val")
    with open(tmp_path / "val" / "split.csv") as f:
        train = {r["individual"] for r in csv.DictReader(f) if r["set"] == "train"}
    assert not train & set(pipeline["test"]) and train == set(meta["train_individuals"])


def test_step6_reproduces_the_stored_validation_accuracy(graph_copy, tmp_path):
    """Steps 4a and 6 build the same tokens: with --splits none, step 6 on the model's own validation
    individuals reproduces the best validation accuracy stored in the checkpoint."""
    out = tmp_path / "m.pt"
    script("4a_finetune_evo2_embeddings").main(["--graph-dir", str(graph_copy), "--min-support", "3",
                                                "--val-examples", "1000", "--out", str(out)] + STEP4_FAST)
    meta = torch.load(out, weights_only=True)
    _, tasks = run_validate(["--model", out, "--graph-dir", graph_copy, "--splits", "none", "--task",
                             "test_set_performance max_examples=0 train_examples=0 bootstrap=0 device=cpu"], tmp_path / "v")
    s = tasks["test_set_performance"]["summary"]
    assert s["test_top1_accuracy"] == pytest.approx(meta["metrics"]["best"]["accuracy"], abs=1e-4)


# ---- command line --------------------------------------------------------------------

def test_default_tasks_with_gwas_shortcut(pipeline, tmp_path):
    gwas = tmp_path / "gwas.tsv"
    rows = ["CHR\tPOS\tRSID\tP"] + [f"chr21\t{1_000_000 + 10_000 * i}\trs{i}\t{1e-9 if i < 5 else 0.5}"
                                     for i in range(40)]
    gwas.write_text("\n".join(rows) + "\n")
    _, tasks = run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"], "--gwas", gwas],
                            tmp_path / "v")
    assert {"model_info", "test_set_performance", "head_ceiling", "embedding_cooccurrence",
            "cluster_consistency", "gwas_enrichment"} <= set(tasks), sorted(tasks)


def test_step6_exits_nonzero_when_a_task_fails(graph_copy, tmp_path):
    with pytest.raises(SystemExit) as e:
        script("6_validate_evo2").main(["--model", str(graph_copy / "small_next_gnn.pt"), "--graph-dir", str(graph_copy),
                                        "--task", f"gwas_enrichment gwas={tmp_path / 'missing.tsv'}",
                                        "--out-dir", str(tmp_path / "v")])
    assert e.value.code not in (0, None)


def test_step6_missing_model_points_to_step_4a(graph_copy, tmp_path):
    with pytest.raises(SystemExit) as e:
        script("6_validate_evo2").main(["--model", str(graph_copy / "nope.pt"), "--graph-dir", str(graph_copy),
                                        "--out-dir", str(tmp_path / "v")])
    assert "4a_finetune_evo2_embeddings.py" in str(e.value)


def test_documented_validation_tasks_match_the_file():
    text = (PART1.parent / "DOCUMENTATION.md").read_text()
    block = re.search(r"one per line \(\[`validation_tasks\.txt`\][^\n]*\n+```\n(.*?)```", text, re.S)
    assert block, "excerpt of validation_tasks.txt not found in DOCUMENTATION.md"
    documented = [ln.strip() for ln in block.group(1).splitlines() if ln.strip()]
    actual = [ln.strip() for ln in (PART1 / "validation_tasks.txt").read_text().splitlines()
              if ln.strip() and not ln.lstrip().startswith("#")]
    assert documented == actual


# ---- baselines -----------------------------------------------------------------------

def markov_haplotypes(n, B=12, K=3, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.dirichlet(np.ones(K) * 0.2, size=(B, K))
    Y = np.zeros((n, B), int)
    Y[:, 0] = rng.integers(0, K, n)
    for b in range(1, B):
        for i in range(n):
            Y[i, b] = rng.choice(K, p=T[b, Y[i, b - 1]])
    return Y, np.full(B, K)


def test_baselines_learn_a_markov_chain():
    """On a first-order Markov chain, bigram, trigram and the HMM predict the next block far better than
    the frequency, and their probabilities sum to 1."""
    Y, K = markov_haplotypes(600)
    bl = BlockBaselines(Y[:400], K)
    ll = {}
    for b in (1, 5):
        for n in ("frequency", "bigram", "trigram"):
            assert np.allclose(bl.next_dist(Y[400:], b, n).sum(1), 1)
    for n in ("frequency", "bigram", "trigram", "hmm"):
        top, pt = bl.predict_next(Y[400:], n)
        assert top.shape == pt.shape == (200, Y.shape[1] - 1) and (pt > 0).all() and (pt <= 1).all()
        ll[n] = np.log(pt).mean()
    assert min(ll["bigram"], ll["trigram"], ll["hmm"]) > ll["frequency"] + 0.1, ll


def test_hmm_imputation_uses_both_flanks():
    Y, K = markov_haplotypes(600, seed=1)
    bl = BlockBaselines(Y[:400], K)
    mask = np.zeros((200, Y.shape[1]), bool)
    mask[:, 6] = True
    post = bl.hmm(Y[400:], causal=False, mask=mask)[6]
    assert np.allclose(post.sum(1), 1)
    one = mask.copy()                                  # rows masked elsewhere do not change a row's posterior
    one[1:, 6], one[1:, 3] = False, True
    assert np.allclose(bl.hmm(Y[400:], causal=False, mask=one)[6][0], post[0], atol=1e-5)
    assert (post.argmax(1) == Y[400:, 6]).mean() > (bl.freq[6].argmax() == Y[400:, 6]).mean()


# ---- model-querying tasks ---------------------------------------------------------------

def test_masked_model_imputation_and_model_comparison(graph_copy, tmp_path):
    m4 = script("4a_finetune_evo2_embeddings")
    masked, frozen = tmp_path / "masked.pt", tmp_path / "frozen.pt"
    m4.main(["--graph-dir", str(graph_copy), "--objective", "masked", "--out", str(masked)] + STEP4_FAST)
    m4.main(["--graph-dir", str(graph_copy), "--train-blocks", "0", "--out", str(frozen)] + STEP4_FAST)
    _, tasks = run_validate(["--model", graph_copy / "small_next_gnn.pt", "--model", frozen, "--model", masked,
                             "--graph-dir", graph_copy, "--task", f"masked_imputation model={masked} spans=1 "
                             "per_individual=2 bootstrap=20 device=cpu", "--task", "model_comparison bootstrap=20 device=cpu"],
                            tmp_path / "v")
    m = tasks["masked_imputation"]["summary"]
    assert m["model_scored"] == "yes" and 0 <= m["span1_model_concordance"] <= 1
    pairs = open(tmp_path / "v" / "model_comparison_pairs.csv").read()
    assert pairs.count("yes") == 1 and pairs.count("no (") == 2          # next vs masked are not compared


def test_phenotype_prediction_null_trait_has_no_signal(graph_copy, tmp_path):
    out = tmp_path / "p.pt"
    script("4b_finetune_evo2_phenotype").main(["--graph-dir", str(graph_copy), "--method", "gnn", "--simulate", "null",
                                               *SMALL, "--epochs", "1", "--batch", "8", "--log-every", "1000",
                                               "--out", str(out)])
    _, tasks = run_validate(["--model", out, "--graph-dir", graph_copy, "--task",
                             "phenotype_prediction bootstrap=200 device=cpu"], tmp_path / "v")
    s = tasks["phenotype_prediction"]["summary"]
    assert s["ridge_r2_95ci"][0] < 0.05 and s["causal_haploblocks"] == 0


def test_ridge_recovers_planted_additive_blocks(graph_copy):
    """With strong additive effects, ridge on cluster dosages ranks the causal haploblocks first."""
    val = script("6_validate_evo2")
    import pandas as pd
    import scipy.sparse as sp
    m4b = script("4b_finetune_evo2_phenotype")
    m4b.simulate(str(graph_copy), "additive", 0.95, 2, 3)
    key = json.loads((graph_copy / "phenotypes" / "additive_answer.json").read_text())
    h = np.load(graph_copy / "haplotypes.npz")
    H = h["node"]
    X = sp.csr_matrix((np.ones(H.size), (np.repeat(np.arange(len(H)) // 2, H.shape[1]), H.ravel())))
    inds = open(graph_copy / "individuals.txt").read().split()
    y = pd.concat([pd.read_csv(graph_copy / "phenotypes" / f"additive_{p}.tsv", sep="\t") for p in ("train", "test")])
    y = y.set_index("individual")["additive"].reindex(inds).to_numpy()
    _, beta = val.ridge_fit_predict(X, y, X)
    nodes = pd.read_csv(graph_copy / "nodes.csv.gz")
    imp = pd.Series(np.abs(beta[:len(nodes)])).groupby(nodes["region"].to_numpy()[:len(beta)]).sum()
    causal = {b for c in key["causal"] for b in c["haploblocks"]}
    assert causal <= set(imp.sort_values(ascending=False).index[:2 * len(causal)])
