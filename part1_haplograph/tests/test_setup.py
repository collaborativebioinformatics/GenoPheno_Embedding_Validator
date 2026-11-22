"""
setup.py: the Evo 2 install plan (dry run, Linux GPU simulated) and readable failures.
Run:  python -m pytest part1_haplograph/tests -q
"""
import pytest

from conftest import script

setup = script("setup")


def test_evo2_dry_run_plans_every_install_step(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name)
    setup.main(["--evo2", "--dry-run", "--skip-cuda-toolkit", "--venv", str(tmp_path / "env"),
                "--evo2-src", str(tmp_path / "src"), "--evo2-ref", "v0.3.0"])
    out = capsys.readouterr().out
    for expected in ("-m venv", "https://download.pytorch.org/whl/cu128", "flash-attn==2.8.0.post2",
                     "--no-build-isolation", "git clone", "checkout --quiet v0.3.0", "requirements.txt",
                     "evo2.test.test_evo2_generation"):
        assert expected in out, expected


def test_failing_command_shows_its_output_and_a_hint(capsys):
    args = setup.argparse.Namespace(dry_run=False, verbose=False)
    with pytest.raises(setup.SetupError) as e:
        setup.run([setup.sys.executable, "-c", "print('boom'); raise SystemExit(3)"], "a failing step", args,
                  hint="try X")
    msg = str(e.value)
    assert "a failing step failed (exit code 3)" in msg and "boom" in msg and "hint: try X" in msg


def test_evo2_refuses_non_linux_with_a_hint(monkeypatch, capsys):
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit):
        setup.main(["--evo2", "--dry-run"])
    assert "--mock-evo2" in capsys.readouterr().err
