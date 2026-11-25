"""
Regression tests for spec-test findings on the haplotype-level pipeline (steps 4a, 4b, 6).
Run:  python -m pytest part1_haplograph/tests/test_spec_haplotypes.py -q
"""
import pytest

import evo2_model
from conftest import SMALL, STEP4_FAST, run_validate, script


def test_masked_imputation_maps_softmax_all_probabilities_to_the_target_block(graph_copy, tmp_path, monkeypatch):
    """masked_imputation turns the model's candidate probabilities into a per-haploblock vector with
    data.block_pos[cand]. With --softmax all (allowed for --objective masked) the candidates are every
    token of the chromosome, so tokens of other haploblocks overwrite the target haploblock's slots.
    An oracle that puts all mass on the true cluster must give concordance 1 for either softmax."""
    m4 = script("4a_finetune_evo2_embeddings")
    out = tmp_path / "masked_all.pt"
    m4.main(["--graph-dir", str(graph_copy), "--objective", "masked", "--softmax", "all", "--out", str(out)]
            + STEP4_FAST)
    orig = evo2_model.HaploModel.score

    def oracle(self, chrom, ex, mask=None, probs=False):
        sc = orig(self, chrom, ex, mask=mask, probs=probs)
        if probs:
            sc["p"] = (sc["cand"] == sc["target"][:, None]).astype(float)
        return sc
    monkeypatch.setattr(evo2_model.HaploModel, "score", oracle)
    _, tasks = run_validate(["--model", out, "--graph-dir", graph_copy, "--task",
                             "masked_imputation spans=1 per_individual=2 bootstrap=0 device=cpu"], tmp_path / "v")
    s = tasks["masked_imputation"]["summary"]
    assert s["model_scored"] == "yes"
    assert s["span1_model_concordance"] == 1.0


def test_4b_empty_phenotype_file_stops_with_a_clear_message(graph_copy, tmp_path):
    """An empty --phenotype (or --hash-pheno) file raised pandas.errors.EmptyDataError with a traceback
    instead of the script's own 'expected a tab-separated file with 2 columns' error."""
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    with pytest.raises(SystemExit):
        script("4b_finetune_evo2_phenotype").main(["--graph-dir", str(graph_copy), "--method", "gnn",
                                                   "--phenotype", str(empty), *SMALL, "--epochs", "1"])

