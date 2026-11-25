"""
Step 4a / 4b model: haplotype tokens, softmax and heads, masked objective, controls, training loop.
Run:  python -m pytest part1_haplograph/tests -q
"""
import argparse
import json

import numpy as np
import pytest
import torch

from conftest import SMALL, STEP4_FAST, script
from evo2_model import GenomeData, HaploModel, SmallRunner, backbone_fingerprint, fit, randomize
from haplo_common import read_graph_individuals, set_id


def data_of(G, units="haplotype", min_support=0, pos_freqs=4):
    n = len(read_graph_individuals(G))
    return GenomeData(str(G), "gnn", None, min_support, list(range(n)), pos_freqs, units)


def test_haplotype_units_have_one_token_per_haploblock_in_the_same_order(pipeline):
    d = data_of(pipeline["G"])
    for c in d.chroms:
        n_blocks = len(d.block_cols[c])
        regions = None
        for ex in range(d.H.shape[0]):
            t = d.tokens(c, ex)
            assert len(t) == n_blocks
            r = d.region_code[t]
            assert regions is None or (r == regions).all()
            regions = r


def test_block_softmax_equals_full_softmax_restricted_to_the_block(pipeline):
    d = data_of(pipeline["G"])
    rnr = SmallRunner(16, 1, None, "cpu", 0)
    block = HaploModel(rnr, d, "scaled-cosine", "block")
    full = HaploModel(rnr, d, "scaled-cosine", "all")
    full.load({k: v for k, v in block.state().items()})
    c = d.chroms[0]
    with torch.no_grad():
        lb, tb, cb, nodes, _ = block.forward(c, 0)
        lf, tf, cf, _, _ = full.forward(c, 0)
    for p in range(len(nodes)):
        in_block = d.region_code[cf[p].numpy()] == d.region_code[nodes[p]]
        restricted = torch.log_softmax(lf[p][torch.from_numpy(in_block)], 0)
        target = int(np.where(cf[p].numpy()[in_block] == nodes[p])[0][0])
        assert abs(float(restricted[target]) - float(torch.log_softmax(lb[p], 0)[tb[p]])) < 1e-5


def test_head_floors(pipeline):
    """A learned table has no fixed floor; a cosine head at the largest scale gets close to 0 on distinct keys."""
    val = script("6_validate_evo2")
    d = data_of(pipeline["G"])
    rnr = SmallRunner(16, 1, None, "cpu", 0)
    targets = np.concatenate([d.tokens(c, 0)[1:] for c in d.chroms])
    assert val.head_floor(HaploModel(rnr, d, "table", "block"), d, targets) is None
    m = HaploModel(rnr, d, "scaled-cosine", "block")
    with torch.no_grad():
        m.extra["log_scale"].fill_(float(np.log(100.0)))
    assert val.head_floor(m, d, targets) < 0.1


def test_masked_objective_reads_both_sides_and_not_the_masked_token(pipeline):
    d = data_of(pipeline["G"])
    rnr = SmallRunner(16, 2, None, "cpu", 0)
    m = HaploModel(rnr, d, "scaled-cosine", "block", "masked")
    m.train(False)
    c = d.chroms[0]
    orig = d.H.copy()
    cols = d.block_cols[c]

    def pred(pos):
        with torch.no_grad():
            return m.forward(c, 0, mask=[pos])[0][0].clone()
    base = pred(5)
    alt = lambda b: next(x for x in np.unique(d.H[:, cols[b]]) if x != d.H[0, cols[b]])
    d.H[0, cols[5]] = alt(5)                                         # the masked token itself
    assert torch.allclose(pred(5), base)
    d.H[0, cols[5]] = orig[0, cols[5]]
    d.H[0, cols[8]] = alt(8)                                         # a token to the right
    assert not torch.allclose(pred(5), base)
    d.H[:] = orig


def test_init_random_replaces_every_weight_and_keeps_the_parameter_count():
    rnr = SmallRunner(16, 2, None, "cpu", 0)
    before = {k: v.clone() for k, v in rnr.net.state_dict().items()}
    randomize(rnr.net, seed=1)
    after = rnr.net.state_dict()
    assert before.keys() == after.keys()
    assert all(not torch.equal(before[k], after[k]) for k in before if before[k].dim() > 1)    # every weight matrix


def test_resume_continues_where_training_stopped(tmp_path):
    """Two epochs in one run == one epoch, then --resume to two epochs."""
    def run(epochs, resume):
        torch.manual_seed(0)
        lin = torch.nn.Linear(3, 1)
        X, y = torch.randn(12, 3, generator=torch.Generator().manual_seed(1)), torch.randn(12)
        args = argparse.Namespace(lr=1e-2, head_lr=1e-2, batch=4, epochs=epochs, seed=0, log_every=1000,
                                  eval_every=1000, resume=resume)
        loss = lambda j: (lin(X[j]) - y[j]).pow(2).mean()
        fit(args, [], list(lin.parameters()), 12, loss, lambda: {"loss": float(sum(loss(j) for j in range(12)))},
            str, lambda m: None, ("lin", list(lin.parameters())), resume_path=str(tmp_path / f"r{resume}.resume"))
        return lin.weight.detach().clone()
    straight = run(2, False)
    run(1, True)
    (tmp_path / "rTrue.resume").rename(tmp_path / "keep")
    (tmp_path / "keep").rename(tmp_path / "rTrue.resume")
    resumed = run(2, True)
    assert torch.allclose(straight, resumed, atol=1e-6)


def test_smoke_reports_memory_time_and_trainable_parameters():
    rep = script("4a_finetune_evo2_embeddings").smoke(argparse.Namespace(
        backbone="small", bionemo=False, init="pretrained", small_hidden=16, small_layers=1, device="cpu", seed=0,
        train_blocks=None, max_len=32))
    assert rep["trainable_params"] > 0 and rep["seconds_per_step_at_max_len"] >= 0


def test_min_support_counts_training_haplotypes_and_merged_tokens_occur_in_training(pipeline):
    G = pipeline["G"]
    inds = read_graph_individuals(G)
    tr = [i for i, x in enumerate(inds) if x in set(pipeline["train"])]
    d = GenomeData(str(G), "gnn", None, 3, tr, 4)
    rows = np.ravel([[2 * i, 2 * i + 1] for i in tr])
    assert np.array_equal(d.rare, np.bincount(d.H[rows].ravel(), minlength=len(d.rare)) < 3)
    learnable = {int(d.canon[n]) for n in np.where(d.rare & (d.support > 0))[0]}
    seen = {int(t) for c in d.chroms for r in rows for t in d.tokens(c, r) if d.rare[t]}
    assert learnable <= seen


def test_position_and_block_length_are_appended(pipeline):
    """--pos-freqs k adds 2k standardized sin/cos columns plus one column of haploblock log length."""
    G = pipeline["G"]
    base, enc = data_of(G, pos_freqs=0), data_of(G, pos_freqs=8)
    assert enc.Z.shape[1] == base.Z.shape[1] + 16
    assert abs(float(enc.Z[:, -16:].std(0).mean()) - 1) < 0.05
    emb = np.load(G / "node_emb_gnn.npy").shape[1]
    assert base.Z.shape[1] == emb + 1


def test_step4_refuses_embeddings_from_another_training_set(graph_copy, pipeline):
    info_p = graph_copy / "node_emb_gnn.json"
    info = json.loads(info_p.read_text())
    info["train_set_id"] = set_id(pipeline["test"])
    info_p.write_text(json.dumps(info))
    with pytest.raises(SystemExit):
        script("4a_finetune_evo2_embeddings").main(["--graph-dir", str(graph_copy), "--out", str(graph_copy / "x.pt")]
                                                   + STEP4_FAST)


def test_checkpoints_load_with_weights_only(pipeline, tmp_path, graph_copy):
    """4a and 4b checkpoints stay loadable with weights_only=True, so step 6 never unpickles code."""
    torch.load(pipeline["model"], map_location="cpu", weights_only=True)
    m4b = script("4b_finetune_evo2_phenotype")
    m4b.main(["--graph-dir", str(graph_copy), "--method", "gnn", "--simulate", "additive", "--causal-blocks", "2",
              *SMALL, "--epochs", "1", "--batch", "8", "--log-every", "1000", "--out", str(tmp_path / "p.pt")])
    torch.load(tmp_path / "p.pt", map_location="cpu", weights_only=True)
    tsv = tmp_path / "h.tsv"
    m4b.write_test_hashes(str(tsv), n=20, bits=16)
    m4b.main(["--input", "hash", "--hash-pheno", str(tsv), "--hash-bits", "8", *SMALL, "--epochs", "1", "--batch", "4",
              "--log-every", "1000", "--out", str(tmp_path / "h.pt")])
    torch.load(tmp_path / "h.pt", map_location="cpu", weights_only=True)


def test_4b_refuses_test_individuals_in_its_phenotype_file(graph_copy, tmp_path):
    f = tmp_path / "p.tsv"
    f.write_text("individual\ty\n" + "".join(f"{x}\t{i}\n" for i, x in
                                              enumerate((graph_copy / "test_individuals.txt").read_text().split())))
    with pytest.raises(SystemExit, match="test individuals"):
        script("4b_finetune_evo2_phenotype").main(["--graph-dir", str(graph_copy), "--method", "gnn",
                                                   "--phenotype", str(f), *SMALL])


def test_cis_simulation_ignores_trans_configurations(graph_copy):
    """The cis trait counts haplotypes carrying BOTH clusters; the same clusters on different haplotypes
    contribute nothing."""
    m4b = script("4b_finetune_evo2_phenotype")
    m4b.simulate(str(graph_copy), "cis", 0.999, 1, 0)
    key = json.loads((graph_copy / "phenotypes" / "cis_answer.json").read_text())["causal"][0]
    h = np.load(graph_copy / "haplotypes.npz")
    H, blocks = h["node"], list(h["blocks"])
    b0, b1 = blocks.index(key["haploblocks"][0]), blocks.index(key["haploblocks"][1])
    a, c = key["clusters"]
    cis = ((H[:, b0] == a) & (H[:, b1] == c)).reshape(-1, 2).sum(1)
    import pandas as pd
    inds = read_graph_individuals(graph_copy)
    y = pd.concat([pd.read_csv(graph_copy / "phenotypes" / f"cis_{p}.tsv", sep="\t") for p in ("train", "test")])
    y = y.set_index("individual")["cis"].reindex(inds).to_numpy()
    assert np.corrcoef(cis, y)[0, 1] ** 2 > 0.99


def test_backbone_fingerprint_follows_weights_and_ignores_extra_state():
    class TE(torch.nn.Linear):                         # like TransformerEngine: FP8 metadata in the state_dict
        def get_extra_state(self):
            return torch.randn(4)
    torch.manual_seed(0)
    a = TE(4, 4)
    f = backbone_fingerprint(a)
    assert backbone_fingerprint(a) == f
    with torch.no_grad():
        a.weight[0, 0] += 1
    assert backbone_fingerprint(a) != f
