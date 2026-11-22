"""
Iteration 3: positive control for the step-4 learning setup (H12 in IMPROVEMENT_PLAN.md).
Run:  python -m pytest part1_haplograph/tests -q
"""
from conftest import STEP4_FAST, run_validate, script


def test_step4_learns_at_least_the_bigram_signal(graph_copy, tmp_path):
    """With the genomic-position encoding (step-4 default), the mock backbone must reach at least
    0.8 x the bigram's top-1 on its validation individuals (half of the training set, so the
    comparison rests on hundreds of predictions): the learning setup can learn."""
    out = tmp_path / "control.pt"
    script("4a_finetune_evo2_embeddings").main(
        ["--graph-dir", str(graph_copy), "--epochs", "60", "--val-frac", "0.5", "--out", str(out)] + STEP4_FAST)
    _, tasks = run_validate(["--model", out, "--graph-dir", graph_copy, "--splits", "none", "--task",
                             "test_set_performance max_examples=0 train_examples=5 bootstrap=0 device=cpu"],
                            tmp_path / "v")
    s = tasks["test_set_performance"]["summary"]
    assert s["test_predictions"] >= 200
    assert s["test_top1_accuracy"] >= 0.8 * s["bigram_baseline_top1_accuracy"], s


def test_position_encoding_is_appended(pipeline):
    """--pos-freqs k adds 2k standardized columns (sin, cos) to every embedding."""
    from evo2_model import GenomeData
    G = str(pipeline["G"])
    base, enc = GenomeData(G, "gnn", None, 0, [], 0), GenomeData(G, "gnn", None, 0, [], 8)
    assert enc.Z.shape[1] == base.Z.shape[1] + 16
    assert abs(float(enc.Z[:, -16:].std(0).mean()) - 1) < 0.05
