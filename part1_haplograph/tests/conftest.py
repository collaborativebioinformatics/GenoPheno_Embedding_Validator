"""
Shared fixtures for the part 1 regression tests (CPU only, synthetic data, no network).

`pipeline` builds a tiny synthetic haplograph once per test session, in a pytest temp folder
(never the shared <system temp>/haplograph_selftest used by the scripts' --test flags):
  step 1 equivalent (synthetic raw/ + splits/), step 2, step 3 (gnn only), step 4 (--mock-evo2).
Tests that change files must work on a copy (see `graph_copy`).
"""
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

PART1 = Path(__file__).resolve().parents[1]
if str(PART1) not in sys.path:
    sys.path.insert(0, str(PART1))

from haplo_common import make_synthetic_haplograph, set_id, write_individual_list  # noqa: E402

SPLIT_NAME = "halfX"


def script(name):
    """Import a pipeline script by file name (they start with digits)."""
    return importlib.import_module(name)


STEP3_FAST = ["--method", "gnn", "--device", "cpu", "--dim", "8", "--svd-dim", "8",
              "--gnn-epochs", "10", "--gnn-hidden", "16"]
STEP4_FAST = ["--method", "gnn", "--mock-evo2", "--device", "cpu", "--mock-hidden", "32",
              "--mock-layers", "2", "--train-blocks", "1", "--max-len", "64", "--batch", "4",
              "--eval-every", "10000", "--log-every", "1000"]


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    wd = tmp_path_factory.mktemp("haplo")
    inds = make_synthetic_haplograph(wd / "raw", ("chr21", "chr22"), n_ind=40, seed=0)
    train, test = inds[::2], inds[1::2]
    splits = wd / "splits"
    splits.mkdir()
    write_individual_list(splits / "train_individuals.txt", train)
    write_individual_list(splits / "test_individuals.txt", test)
    (splits / "split.json").write_text(json.dumps({
        "name": SPLIT_NAME, "train_file": "train_individuals.txt", "test_file": "test_individuals.txt",
        "train_set_id": set_id(train), "test_set_id": set_id(test)}))
    script("2_build_graph").main(["--workdir", str(wd), "--min-lift", "1.2"])
    G = wd / f"chr21_chr22_train_{SPLIT_NAME}"
    script("3_node_embeddings").main(["--graph-dir", str(G)] + STEP3_FAST)
    script("4a_finetune_evo2_embeddings").main(["--graph-dir", str(G)] + STEP4_FAST)
    model = G / "evo2_mock_next_gnn.pt"
    assert model.exists()
    return {"wd": wd, "G": G, "model": model, "train": train, "test": test}


@pytest.fixture
def graph_copy(pipeline, tmp_path):
    """A private copy of the pipeline's graph folder (with embeddings and the mock model)."""
    dst = tmp_path / "graph"
    shutil.copytree(pipeline["G"], dst, ignore=shutil.ignore_patterns("validation_*"))
    return dst


def run_validate(args, out_dir):
    """Run step 6 in-process; return summary.json as a dict and the task records by name."""
    try:
        script("6_validate_evo2").main(list(map(str, args)) + ["--out-dir", str(out_dir)])
    except SystemExit:                       # a failed task: the summary is still written
        if not (Path(out_dir) / "summary.json").exists():
            raise
    summary = json.loads((Path(out_dir) / "summary.json").read_text())
    return summary, {t["name"]: t for t in summary["tasks"]}
