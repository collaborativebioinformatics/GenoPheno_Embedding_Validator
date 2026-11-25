"""
Step 3: carrier_svd, the XGBoost cis-partner target, evo2_seq, and training-set checks.
Run:  python -m pytest part1_haplograph/tests -q
"""
import json

import numpy as np
import pytest

from conftest import STEP3_FAST, script


def test_carrier_svd_is_the_gnn_input(pipeline):
    """carrier_svd = the first --dim columns of the cached carrier features the GNN reads (largest component first)."""
    G = pipeline["G"]
    Z = np.load(G / "node_emb_carrier_svd.npy")
    feats = [f for f in G.iterdir() if f.name.startswith("features_svd")]
    assert feats and np.allclose(Z, np.load(feats[0])[:, :Z.shape[1]])
    var = np.load(feats[0]).var(0)
    assert np.allclose(var, 1, atol=1e-3)                    # standardized columns


def test_xgboost_target_is_the_cis_partner_rank(graph_copy):
    """The class of a node is the frequency rank, in the next haploblock, of its most frequent partner on
    the same training haplotype (junctions.npz); -1 when there is none."""
    import pandas as pd
    m3 = script("3_node_embeddings")
    nodes = pd.read_csv(graph_copy / "nodes.csv.gz")
    args = type("A", (), {"graph_dir": str(graph_copy)})()
    y, source = m3.cis_partner_class(args, nodes, 4)
    assert "junctions" in source and y.min() >= -1 and y.max() <= 3
    j = np.load(graph_copy / "junctions.npz")
    a = int(j["a"][0])
    partners = j["b"][j["a"] == a][np.argmax(j["count"][j["a"] == a])]
    region = nodes.loc[partners, "region"]
    same = nodes[nodes["region"] == region].sort_values("support_train", ascending=False, kind="stable")
    assert y[a] == min(list(same["node"]).index(partners), 3)


def test_evo2_seq_embeds_every_cluster_and_caches_per_backbone(graph_copy, pipeline):
    """evo2_seq gives every cluster an embedding from its representative sequence; a different backbone
    setting must not reuse the cached outputs."""
    G = graph_copy
    raw = pipeline["wd"] / "raw"
    common = ["--graph-dir", str(G), "--method", "evo2_seq", "--raw-dir", str(raw), "--backbone", "small",
              "--small-layers", "1", "--seq-window", "64", "--dim", "4", "--device", "cpu"]
    m3 = script("3_node_embeddings")
    m3.main(common + ["--small-hidden", "16"])
    info = json.loads((G / "node_emb_evo2_seq.json").read_text())["summary"]
    assert info["clusters_with_sequence"] == info["clusters"]
    m3.main(common + ["--small-hidden", "24", "--force"])
    assert len(list((G / "evo2_seq_cache").iterdir())) == 2


def test_step3_refuses_training_set_other_than_graphs(graph_copy):
    """Each step checks that it uses the same training set as the step before and stops if not."""
    with pytest.raises(SystemExit):
        script("3_node_embeddings").main(["--graph-dir", str(graph_copy), "--force",
                                          "--train-individuals", str(graph_copy / "test_individuals.txt")]
                                         + STEP3_FAST)


def test_step3_recomputes_when_settings_change(graph_copy):
    """Rerunning step 3 with another --dim must not leave embeddings of the old size."""
    args = list(STEP3_FAST)
    args[args.index("--dim") + 1] = "12"
    script("3_node_embeddings").main(["--graph-dir", str(graph_copy)] + args)
    assert np.load(graph_copy / "node_emb_gnn.npy").shape[1] == 12
