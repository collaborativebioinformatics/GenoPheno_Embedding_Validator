"""
Positive controls: on synthetic haplotypes drawn from a known block-HMM, the learning setup of step 4a
must get close to what the generating model achieves.
Run:  python -m pytest part1_haplograph/tests -q
"""
import numpy as np

from conftest import SMALL, SPLIT_NAME, run_validate, script, synthetic_workdir
from haplo_common import read_graph_individuals, synthetic_next_block_ce

CTRL = ["--method", "gnn", *SMALL[:-4], "--small-hidden", "64", "--small-layers", "2", "--batch", "8",
        "--log-every", "1000", "--eval-every", "10000", "--val-frac", "0.2", "--head", "table"]


def control_graph(tmp_path):
    wd = tmp_path / "ctrl"
    synthetic_workdir(wd, ("chr21",), n_ind=300, n_regions=30, seed=3)
    script("2_build_graph").main(["--workdir", str(wd), "--min-lift", "1.2"])
    G = wd / f"chr21_train_{SPLIT_NAME}"
    script("3_node_embeddings").main(["--graph-dir", str(G), "--method", "gnn", "--device", "cpu", "--dim", "8",
                                      "--svd-dim", "8", "--gnn-epochs", "30", "--gnn-hidden", "16"])
    return wd, G


def test_small_backbone_approaches_the_true_conditional_entropy(tmp_path):
    """Next-haploblock cross-entropy of the small backbone on test haplotypes is within 10% of the
    generating model's (the population of each haplotype is hidden, as for the model)."""
    wd, G = control_graph(tmp_path)
    out = G / "ctrl.pt"
    script("4a_finetune_evo2_embeddings").main(["--graph-dir", str(G), "--epochs", "8", "--min-support", "0",
                                                "--out", str(out)] + CTRL)
    _, tasks = run_validate(["--model", out, "--graph-dir", G, "--task",
                             "test_set_performance max_examples=0 train_examples=0 bootstrap=0 device=cpu"], tmp_path / "v")
    s = tasks["test_set_performance"]["summary"]
    inds = read_graph_individuals(G)
    test = (G / "test_individuals.txt").read_text().split()
    rows = np.ravel([[2 * inds.index(x), 2 * inds.index(x) + 1] for x in test])
    true_ce = synthetic_next_block_ce(wd / "raw", "chr21", rows)
    assert s["test_loss"] <= 1.1 * true_ce, (s["test_loss"], true_ce)
    assert s["test_top1_accuracy"] >= 0.8 * s["bigram_top1_accuracy"], s


def test_masked_imputation_reaches_most_of_the_hmm(tmp_path):
    """A masked-objective small backbone imputes single masked haploblocks with at least 0.9 x the
    concordance of the Li-Stephens HMM fitted on the training haplotypes (it needs many epochs on
    240 training individuals: 0.88 x after 20, 0.95 x after 80)."""
    wd, G = control_graph(tmp_path)
    out = G / "masked.pt"
    script("4a_finetune_evo2_embeddings").main(["--graph-dir", str(G), "--epochs", "80", "--min-support", "0",
                                                "--objective", "masked", "--mask-frac", "0.3", "--out", str(out)] + CTRL)
    _, tasks = run_validate(["--model", out, "--graph-dir", G, "--task",
                             "masked_imputation spans=1 per_haplotype=5 bootstrap=0 device=cpu"], tmp_path / "v")
    s = tasks["masked_imputation"]["summary"]
    assert s["span1_model_concordance"] >= 0.9 * s["span1_hmm_concordance"], s
