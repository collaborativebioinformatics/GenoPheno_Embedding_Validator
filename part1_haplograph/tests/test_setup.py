"""
setup.py: the Evo 2 install plan (dry run, Linux GPU simulated) and readable failures.
Run:  python -m pytest part1_haplograph/tests -q
"""
import pytest

from conftest import script

setup = script("setup")


def test_evo2_dry_run_plans_every_install_step(monkeypatch, capsys, tmp_path):
    fake_linux_gpu(monkeypatch)
    setup.main(["--evo2", "--dry-run", "--skip-cuda-toolkit", "--venv", str(tmp_path / "env"),
                "--evo2-src", str(tmp_path / "src"), "--evo2-ref", "v0.3.0"])
    out = capsys.readouterr().out
    for expected in ("-m venv", "https://download.pytorch.org/whl/cu128", "flash-attn==2.8.0.post2",
                     "--no-build-isolation", "git clone", "checkout --quiet --detach v0.3.0", "requirements.txt",
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
    assert "--backbone small" in capsys.readouterr().err


def fake_linux_gpu(monkeypatch, cap="12.0", cuda="12.8"):
    """Simulate a Linux machine whose nvidia-smi reports `cap` and a driver for CUDA `cuda`."""
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name)
    real = setup.run

    def run(cmd, what, args, hint="", env=None, probe=False):
        if probe and cmd[0] == "nvidia-smi":
            return cap + "\n" if "--query-gpu=compute_cap" in cmd else f"| CUDA Version: {cuda} |"
        return real(cmd, what, args, hint, env, probe)
    monkeypatch.setattr(setup, "run", run)


def test_dry_run_reads_the_gpu_and_forces_a_flash_attn_build_on_sm120(monkeypatch, capsys, tmp_path):
    fake_linux_gpu(monkeypatch, cap="12.0")
    setup.main(["--evo2", "--dry-run", "--skip-cuda-toolkit", "--venv", str(tmp_path / "e")])
    out = capsys.readouterr().out
    assert "compute capability 12.0" in out and "--no-binary flash-attn" in out and "compiling from source" in out


def test_old_driver_is_refused(monkeypatch, capsys):
    fake_linux_gpu(monkeypatch, cuda="12.4")
    with pytest.raises(SystemExit):
        setup.main(["--evo2", "--dry-run", "--skip-cuda-toolkit"])
    assert "CUDA 12.4" in capsys.readouterr().err


def test_unexpected_compute_capability_is_a_clear_error(monkeypatch, capsys):
    fake_linux_gpu(monkeypatch, cap="[N/A]")
    with pytest.raises(SystemExit):
        setup.main(["--evo2", "--dry-run", "--skip-cuda-toolkit"])
    assert "--gpu rtx" in capsys.readouterr().err


def test_verify_catches_an_import_that_crashes_python(tmp_path, monkeypatch):
    (tmp_path / "crashing_mod.py").write_text("import os\nos.abort()\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    args = setup.argparse.Namespace(dry_run=False, verbose=False)
    with pytest.raises(setup.SetupError, match="cannot be imported"):
        setup.verify(setup.sys.executable, ["json", "crashing_mod"], args)


def test_os_release_parsing_and_missing_file(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('ID=ubuntu\nVERSION_ID="24.04"\n')
    assert setup.os_release(f) == {"ID": "ubuntu", "VERSION_ID": "24.04"}
    with pytest.raises(setup.SetupError, match="--skip-cuda-toolkit"):
        setup.os_release(tmp_path / "missing")


def test_environment_variables_set_the_defaults(monkeypatch, capsys, tmp_path):
    fake_linux_gpu(monkeypatch, cap="8.9")
    monkeypatch.setenv("EVO2_REF", "v9.9")
    monkeypatch.setenv("SKIP_CUDA_TOOLKIT", "1")
    monkeypatch.setenv("MAX_JOBS", "7")
    setup.main(["--evo2", "--dry-run", "--venv", str(tmp_path / "e"), "--evo2-src", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert "skipped (--skip-cuda-toolkit)" in out and "v9.9" in out and "--no-binary" not in out
    monkeypatch.setenv("MAX_JOBS", "many")
    with pytest.raises(SystemExit):
        setup.main(["--evo2", "--dry-run"])


def test_cpu_dry_run_plans_a_venv_and_the_requirements(capsys, tmp_path):
    setup.main(["--dry-run", "--venv", str(tmp_path / "env")])
    out = capsys.readouterr().out
    assert "-m venv" in out and "requirements.txt" in out
