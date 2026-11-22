"""
Regression tests from the spec check of the renumbered part 1 pipeline (steps 1-6, setup.py):
README.md part 1 and DOCUMENTATION.md sections 1, 3, 4, 10, 11 against the scripts' behaviour.
Each test guards one confirmed (and fixed) issue.
CPU only, synthetic data, no network.  Run:  python -m pytest part1_haplograph/tests -q
"""
import re
import shutil
from pathlib import Path

import pytest

from conftest import PART1, script

DOC = PART1.parent / "DOCUMENTATION.md"


def test_step6_exits_nonzero_when_a_task_fails(graph_copy, tmp_path):
    """README: validation_tasks.txt needs step 5's haplograph/gwas/height.tsv, so step 5 must run first.
    If it did not, gwas_enrichment fails, yet step 6 exits 0, so a chained run (`set -e`, CI)
    reports success. A failed task should give a non-zero exit code."""
    missing = tmp_path / "gwas" / "height.tsv"                      # step 5 not run
    with pytest.raises(SystemExit) as e:
        script("6_validate_evo2").main([
            "--model", str(graph_copy / "evo2_mock_next_gnn.pt"), "--graph-dir", str(graph_copy),
            "--task", f"gwas_enrichment gwas={missing}", "--out-dir", str(tmp_path / "v")])
    assert e.value.code not in (0, None)


def test_step2_missing_chromosome_hint_keeps_the_workdir(pipeline, tmp_path):
    """2_build_graph.py --workdir W --chroms chr22, with chr22 not downloaded, tells the user to run
    `python 1_download_haplograph.py --chroms chr22`, which writes to ./haplograph, not W; rerunning
    step 2 then fails again. The hint must carry the same --workdir."""
    wd = tmp_path / "custom_workdir"
    shutil.copytree(pipeline["wd"] / "splits", wd / "splits")
    shutil.copytree(pipeline["wd"] / "raw" / "chr21", wd / "raw" / "chr21")
    with pytest.raises(SystemExit) as e:
        script("2_build_graph").main(["--workdir", str(wd), "--chroms", "chr22"])
    msg = str(e.value)
    assert "1_download_haplograph.py" in msg and "--workdir" in msg and str(wd) in msg


def test_step6_missing_model_points_to_step_4a(graph_copy, tmp_path):
    """Every other missing-input message names the step to run first (e.g. 'run 2_build_graph.py
    first'); a missing --model only says 'not found'. It should point to 4a_finetune_evo2_embeddings.py."""
    with pytest.raises(SystemExit) as e:
        script("6_validate_evo2").main(["--model", str(graph_copy / "evo2_next_gnn.pt"),
                                        "--graph-dir", str(graph_copy), "--out-dir", str(tmp_path / "v")])
    assert "4a_finetune_evo2_embeddings.py" in str(e.value)


def test_documented_validation_tasks_match_the_file():
    """DOCUMENTATION.md step 6 prints validation_tasks.txt (the file README and section 11 tell users
    to pass); the excerpt has clump=25000 and no min_carriers while the file has clump=auto and
    min_carriers=1, so users copying the excerpt run different settings."""
    text = DOC.read_text()
    block = re.search(r"one per line \(\[`validation_tasks\.txt`\][^\n]*\n+```\n(.*?)```", text, re.S)
    assert block, "excerpt of validation_tasks.txt not found in DOCUMENTATION.md"
    documented = [ln.strip() for ln in block.group(1).splitlines() if ln.strip()]
    actual = [ln.strip() for ln in (PART1 / "validation_tasks.txt").read_text().splitlines()
              if ln.strip() and not ln.lstrip().startswith("#")]
    assert documented == actual


def test_setup_venv_failure_hint_matches_the_cause(tmp_path, monkeypatch):
    """`setup.py --venv PATH` on an unwritable PATH (or on macOS) says 'sudo apt install python3-venv',
    which is wrong there; the hint should name the real cause (the path cannot be created)."""
    setup = script("setup")
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    args = setup.argparse.Namespace(dry_run=False, verbose=False)
    with pytest.raises(setup.SetupError) as e:
        setup.make_venv(setup.sys.executable, str(blocker / "env"), args)
    assert "apt install" not in str(e.value)
