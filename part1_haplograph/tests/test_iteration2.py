"""
Regression tests from the second audit (iteration 2: --min-support merging, docstring shortening).

All issues below are fixed; each test guards one of them.
Run:  python -m pytest part1_haplograph/tests -q
"""
import csv

import numpy as np
import pytest
import torch

from conftest import STEP4_FAST, run_validate, script


def _carrier_support(G, names):
    """Per-node number of carriers among the graph individuals `names`."""
    import scipy.sparse as sp
    from haplo_common import read_graph_individuals
    pos = {x: i for i, x in enumerate(read_graph_individuals(G))}
    C = sp.load_npz(f"{G}/carriers.npz").tocsc()
    return np.asarray(C[:, [pos[x] for x in names]].sum(axis=1)).ravel()


def _step4_default_min_support(G):
    """--min-support default as parsed by step 4 (train() is replaced, nothing is trained)."""
    m4 = script("4a_finetune_evo2_embeddings")
    seen = []
    orig = m4.train
    m4.train = seen.append
    try:
        m4.main(["--method", "gnn", "--graph-dir", str(G), "--mock-evo2"])
    finally:
        m4.train = orig
    return seen[0].min_support


# ---------------------------------------------------------------------------
# Index consistency between steps 4 and 5 (guard; passes)
# ---------------------------------------------------------------------------

def test_step5_reproduces_stored_val_accuracy_with_min_support(graph_copy, tmp_path):
    """Steps 4 and 5 must build the same merged vocabulary (tokens, local indices, mean
    embeddings): with --splits none, step 5 on the model's own validation individuals must
    reproduce the validation accuracy stored in the checkpoint."""
    out = tmp_path / "k3.pt"
    script("4a_finetune_evo2_embeddings").main(
        ["--graph-dir", str(graph_copy), "--min-support", "3", "--val-examples", "1000",
         "--epochs", "3", "--head-lr", "1e-2", "--out", str(out)] + STEP4_FAST)
    meta = torch.load(out, weights_only=True)
    assert meta["config"]["min_support"] == 3
    _, tasks = run_validate(["--model", out, "--graph-dir", graph_copy, "--splits", "none",
                             "--task", "test_set_performance max_examples=0 train_examples=5 bootstrap=0"],
                            tmp_path / "val")
    s = tasks["test_set_performance"]["summary"]
    assert s["test_targets_merged_rare_share"] > 0
    assert s["test_top1_accuracy"] == pytest.approx(meta["metrics"]["after"]["accuracy"], abs=1e-4)


# ---------------------------------------------------------------------------
# Rare-cluster merging (--min-support)
# ---------------------------------------------------------------------------

def test_default_min_support_merged_tokens_occur_in_training(pipeline):
    """The point of the 'rare cluster of this region' token is that the model can learn to predict
    it. With the step-4 default, every merged token with at least one member carried by a training
    individual must occur in the training sequences (at --min-support 1 none does). A region whose
    rare clusters were all unseen in training cannot have a learnable token at any threshold."""
    from evo2_model import GenomeData
    from haplo_common import read_graph_individuals
    G, meta = str(pipeline["G"]), torch.load(pipeline["model"], weights_only=True)
    pos = {x: i for i, x in enumerate(read_graph_individuals(G))}
    tr = [pos[x] for x in meta["train_individuals"]]
    support = _carrier_support(G, meta["train_individuals"])
    data = GenomeData(G, "gnn", None, _step4_default_min_support(G), tr)
    learnable = {int(data.canon[n]) for n in np.where(data.rare & (support > 0))[0]}
    assert learnable, "no merged token has a training carrier on the synthetic graph"
    seen = set()
    for c in data.chroms:
        for i in tr:
            nodes, _ = data.tokens(c, i)
            seen.update(nodes[data.rare[nodes]].tolist())
    assert learnable <= seen, f"{len(learnable - seen)} of {len(learnable)} learnable merged tokens never occur"


def test_min_support_membership_uses_step4_training_individuals_only(pipeline):
    """DOCUMENTATION: '1 merges only clusters never seen in training', 'membership from the training
    carriers only'. Step 4 trains on train_individuals and holds out val_individuals, so membership
    must be decided by the carriers among train_individuals alone."""
    from evo2_model import GenomeData
    G, meta = str(pipeline["G"]), torch.load(pipeline["model"], weights_only=True)
    from haplo_common import read_graph_individuals
    pos = {x: i for i, x in enumerate(read_graph_individuals(G))}
    tr = [pos[x] for x in meta["train_individuals"]]
    s_tr = _carrier_support(G, meta["train_individuals"])
    for k in (1, 2, 3, 5):
        assert np.array_equal(GenomeData(G, "gnn", None, k, tr).rare, s_tr < k), f"--min-support {k}"


# ---------------------------------------------------------------------------
# --splits none
# ---------------------------------------------------------------------------

def test_splits_none_train_set_is_models_training_set(pipeline, tmp_path):
    """The bigram baseline is documented as 'counted on the training individuals'. With
    --splits none (test = the model's validation individuals) the train side must be the model's
    own train_individuals, not the graph's held-out test half."""
    meta = torch.load(pipeline["model"], weights_only=True)
    run_validate(["--model", pipeline["model"], "--graph-dir", pipeline["G"], "--splits", "none",
                  "--task", "model_info"], tmp_path / "val")
    with open(tmp_path / "val" / "split.csv") as f:
        train = {r["individual"] for r in csv.DictReader(f) if r["set"] == "train"}
    assert not train & set(pipeline["test"]), "graph test-half individuals are in step 5's train set"
    assert train == set(meta["train_individuals"])
