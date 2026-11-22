"""
Install the dependencies of the part 1 pipeline (haplograph -> embeddings -> Evo 2).

Two modes:
  CPU core (any OS): numpy, pandas, scipy, torch, xgboost from requirements.txt, into the current
      Python or a new venv. Enough for every step with --mock-evo2 and for every --test.
  Evo 2 on a GPU (--evo2; Linux, NVIDIA GPU with compute capability >= 8.0, driver for CUDA 12.8):
      CUDA 12.8 toolkit (Ubuntu, apt + sudo), CUDA_HOME in ~/.bashrc, a venv with torch 2.7.1 (cu128),
      flash-attn 2.8.0.post2 (compiled from source on Blackwell), the evo2 package (light install,
      no Transformer Engine, from git at --evo2-ref), requirements.txt, then Evo 2's generation test.

Each step prints one line; the output of a failing command is shown with a hint (--verbose shows
all output). Settings also read the environment variables of the former install_evo2.sh
(PYTHON, VENV, EVO2_SRC, EVO2_REF, MAX_JOBS, SKIP_CUDA_TOOLKIT=1, SKIP_EVO2_TEST=1).

USAGE
  python setup.py                                # CPU core into the current Python
  python setup.py --venv ~/haplograph-env        # CPU core into a new venv
  python setup.py --evo2                         # Evo 2 on a GPU, venv ~/evo2-env, GPU auto-detected
  python setup.py --evo2 --gpu rtx               # force RTX PRO 6000 Blackwell (sm_120); l40s = sm_89
  python setup.py --evo2 --evo2-ref v0.3.0 --skip-evo2-test
  python setup.py --dry-run [--evo2]             # print the commands without running them
RAPIDS and BioNeMo need their own environments: see DOCUMENTATION.md section 10.
"""
import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUDA_HOME = "/usr/local/cuda-12.8"
TORCH = ["torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"]
FLASH_ATTN = "flash-attn==2.8.0.post2"
EVO2_GIT = "https://github.com/arcinstitute/evo2"
BUILD_TOOLS = ["pip", "setuptools", "wheel", "ninja", "packaging", "psutil"]
CORE = ["numpy", "pandas", "scipy", "torch", "xgboost"]


class SetupError(Exception):
    pass


def log(msg):
    print(f"[setup] {msg}", flush=True)


def run(cmd, what, args, hint="", env=None, check=True):
    """Run a command quietly; on failure show its last output lines and a hint."""
    log(f"{what} ...")
    if args.dry_run:
        print("        $ " + " ".join(shlex.quote(str(c)) for c in cmd))
        return ""
    try:
        r = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})}, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        raise SetupError(f"{what}: command not found: {cmd[0]}" + (f"\n  hint: {hint}" if hint else ""))
    if args.verbose and r.stdout:
        print(r.stdout.rstrip())
    if check and r.returncode:
        tail = "\n".join((r.stdout or "").strip().splitlines()[-25:])
        raise SetupError(f"{what} failed (exit code {r.returncode}): {' '.join(map(str, cmd[:6]))} ...\n"
                         + (f"--- last output ---\n{tail}\n-------------------\n" if tail else "")
                         + (f"  hint: {hint}" if hint else "  rerun with --verbose for the full output"))
    return r.stdout or ""


def pip(py, pkgs, what, args, extra=(), env=None, hint=""):
    run([py, "-m", "pip", "install", "--disable-pip-version-check", *(["-q"] if not args.verbose else []),
         *pkgs, *extra], f"installing {what}", args, env=env,
        hint=hint or "check the network connection and that the Python version is supported (3.10-3.12)")


def make_venv(python, venv, args):
    py = Path(venv).expanduser() / "bin" / "python"
    if py.exists():
        log(f"using the existing venv {venv}")
    else:
        run([python, "-m", "venv", Path(venv).expanduser()], f"creating the venv {venv} with {python}", args,
            hint="on Debian/Ubuntu: sudo apt install python3-venv (or python3.12-venv)")
    return str(py)


def verify(py, modules, args):
    """Import each module and log its version."""
    if args.dry_run:
        return
    code = ("import importlib, sys\nfor m in sys.argv[1:]:\n    try:\n        v = getattr(importlib.import_module(m), "
            "'__version__', 'ok')\n        print(f'{m} {v}')\n    except Exception as e:\n"
            "        print(f'{m} FAILED {type(e).__name__}: {e}')")
    out = subprocess.run([py, "-c", code, *modules], text=True, capture_output=True).stdout
    failed = [line for line in out.splitlines() if " FAILED " in line]
    for line in out.splitlines():
        log(("  x " if " FAILED " in line else "  ok ") + line)
    if failed:
        hint = ("xgboost needs the OpenMP runtime on macOS: brew install libomp"
                if any(line.startswith("xgboost") for line in failed) and sys.platform == "darwin" else
                "rerun with --verbose and check the messages above")
        raise SetupError(f"{len(failed)} package(s) installed but cannot be imported\n  hint: {hint}")


def install_cpu(args):
    py = make_venv(args.python, args.venv, args) if args.venv else sys.executable
    log(f"CPU core into {py}")
    if args.venv:
        pip(py, ["--upgrade", "pip"], "pip (upgrade)", args)
    pip(py, ["-r", HERE / "requirements.txt"], "numpy, pandas, scipy, torch, xgboost (requirements.txt)", args)
    verify(py, CORE, args)
    return py


def gpu_arch(args):
    """Checks for the Evo 2 install; returns the compute capability to build for, e.g. '12.0'."""
    if platform.system() != "Linux":
        raise SetupError("Evo 2 needs Linux with an NVIDIA GPU.\n  hint: on this system use `python setup.py` "
                         "(CPU core) and run step 4a with --mock-evo2")
    if not shutil.which("nvidia-smi"):
        raise SetupError("nvidia-smi not found\n  hint: install the NVIDIA driver first (CUDA 12.8 capable, >= 570)")
    if not shutil.which(args.python):
        raise SetupError(f"{args.python} not found\n  hint: install Python 3.12, or pass --python python3.11")
    smi = run(["nvidia-smi"], "reading the NVIDIA driver version", args)
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    if m and tuple(map(int, m[1].split("."))) < (12, 8):
        raise SetupError(f"the driver supports CUDA {m[1]}, Evo 2's torch build needs 12.8\n"
                         "  hint: update the NVIDIA driver to >= 570")
    cap = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], "detecting the GPU", args)
    arch = {"auto": (cap.strip().splitlines() or ["8.0"])[0].strip(), "rtx": "12.0", "l40s": "8.9"}[args.gpu]
    if int(float(arch)) < 8:
        raise SetupError(f"GPU compute capability {arch} is too old: flash-attn needs 8.0 (Ampere) or newer")
    log(f"GPU compute capability {arch} (sm_{arch.replace('.', '')})" + (f", CUDA {m[1]} driver" if m else ""))
    return arch


def install_cuda_toolkit(args):
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if args.skip_cuda_toolkit or nvcc.exists():
        log("CUDA 12.8 toolkit: " + ("skipped (--skip-cuda-toolkit)" if args.skip_cuda_toolkit else f"found {nvcc}"))
    else:
        osr = dict(re.findall(r'^(\w+)="?([^"\n]*)"?', Path("/etc/os-release").read_text(), re.M))
        if osr.get("ID") != "ubuntu":
            raise SetupError("the automatic CUDA toolkit install supports Ubuntu only\n  hint: install CUDA 12.8 "
                             "yourself and rerun with --skip-cuda-toolkit")
        cpu = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(platform.machine())
        if not cpu:
            raise SetupError(f"unsupported CPU architecture {platform.machine()}")
        repo = f"ubuntu{osr.get('VERSION_ID', '').replace('.', '')}"
        deb = "/tmp/cuda-keyring_1.1-1_all.deb"
        url = f"https://developer.download.nvidia.com/compute/cuda/repos/{repo}/{cpu}/cuda-keyring_1.1-1_all.deb"
        hint = "needs sudo rights and network access to developer.download.nvidia.com"
        run(["wget", "-q", "-O", deb, url], "downloading the CUDA apt keyring", args, hint=hint)
        run(["sudo", "dpkg", "-i", deb], "adding the CUDA apt repository", args, hint=hint)
        run(["sudo", "apt", "update"], "updating apt", args, hint=hint)
        run(["sudo", "apt", "install", "-y", "cuda-toolkit-12-8", "build-essential"],
            "installing the CUDA 12.8 toolkit and build-essential (apt)", args, hint=hint)
    bashrc = Path.home() / ".bashrc"
    if f"CUDA_HOME={CUDA_HOME}" not in (bashrc.read_text() if bashrc.exists() else ""):
        log(f"adding CUDA_HOME={CUDA_HOME} to {bashrc}")
        if not args.dry_run:
            with open(bashrc, "a") as f:
                f.write(f"export CUDA_HOME={CUDA_HOME}\nexport PATH=$CUDA_HOME/bin:$PATH\n"
                        "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n")
    env = {"CUDA_HOME": CUDA_HOME, "PATH": f"{CUDA_HOME}/bin:{os.environ.get('PATH', '')}",
           "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    out = run([f"{CUDA_HOME}/bin/nvcc", "--version"], "checking nvcc", args, env=env,
              hint=f"the CUDA 12.8 toolkit must be in {CUDA_HOME}")
    if not args.dry_run and "release 12.8" not in out:
        raise SetupError(f"{CUDA_HOME}/bin/nvcc is not CUDA 12.8:\n{out.strip()}")
    return env


def install_evo2(args):
    arch = gpu_arch(args)
    env = install_cuda_toolkit(args)
    py = make_venv(args.python, args.venv or "~/evo2-env", args)
    pip(py, ["--upgrade", *BUILD_TOOLS], "build tools (" + ", ".join(BUILD_TOOLS) + ")", args)
    pip(py, TORCH[:1], "torch 2.7.1 for CUDA 12.8", args, extra=TORCH[1:])
    blackwell = int(float(arch)) >= 12          # prebuilt flash-attn wheels often lack sm_120
    pip(py, [FLASH_ATTN], f"{FLASH_ATTN} ({'compiling from source, can take 30+ min' if blackwell else 'wheel or source'})",
        args, extra=["--no-build-isolation"] + (["--no-binary", "flash-attn"] if blackwell else []),
        env={**env, "TORCH_CUDA_ARCH_LIST": arch, "MAX_JOBS": str(args.max_jobs)},
        hint="compiling flash-attn needs the CUDA 12.8 toolkit, build-essential and a lot of RAM: lower --max-jobs")
    src = Path(args.evo2_src).expanduser()
    if not (src / ".git").exists():
        run(["git", "clone", "-q", EVO2_GIT, src], f"cloning evo2 into {src}", args, hint="needs git and network access")
    run(["git", "-C", src, "fetch", "--tags", "--quiet"], "fetching evo2 tags", args)
    run(["git", "-C", src, "checkout", "--quiet", args.evo2_ref], f"checking out evo2 {args.evo2_ref}", args,
        hint="--evo2-ref must be a tag, branch or commit of the evo2 repository")
    pip(py, ["-e", src], f"evo2 ({args.evo2_ref}, light install, no Transformer Engine)", args, env=env)
    pip(py, ["-r", HERE / "requirements.txt"], "pipeline packages (requirements.txt)", args)
    verify(py, CORE + ["flash_attn", "evo2"], args)
    if args.skip_evo2_test:
        log("Evo 2 generation test: skipped")
    else:
        run([py, "-m", "evo2.test.test_evo2_generation", "--model_name", "evo2_7b_base"],
            "Evo 2 generation test (downloads the 7B weights, ~15 GB)", args, env=env,
            hint="needs ~15 GB of disk for the weights and a GPU with enough memory (80 GB recommended)")
    return py


def main(argv=None):
    e = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evo2", action="store_true", help="install Evo 2 on a Linux GPU machine (default: CPU core)")
    ap.add_argument("--venv", default=e("VENV"), help="create / use this venv (default: current Python; ~/evo2-env with --evo2)")
    ap.add_argument("--python", default=e("PYTHON"),
                    help="Python that creates the venv (default: python3.12 with --evo2, else this Python)")
    ap.add_argument("--gpu", choices=["auto", "rtx", "l40s"], default="auto",
                    help="auto-detect, or force RTX PRO 6000 Blackwell (sm_120) / L40S (sm_89)")
    ap.add_argument("--evo2-src", default=e("EVO2_SRC", "~/evo2-src"), help="where the evo2 repository is cloned")
    ap.add_argument("--evo2-ref", default=e("EVO2_REF", "main"), help="evo2 tag / commit (a release tag is reproducible)")
    ap.add_argument("--max-jobs", type=int, default=int(e("MAX_JOBS", "4")), help="parallel flash-attn compile jobs")
    ap.add_argument("--skip-cuda-toolkit", action="store_true", default=e("SKIP_CUDA_TOOLKIT") == "1")
    ap.add_argument("--skip-evo2-test", action="store_true", default=e("SKIP_EVO2_TEST") == "1",
                    help="skip the final generation test (it downloads the 7B weights, ~15 GB)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--verbose", action="store_true", help="show the full output of every command")
    args = ap.parse_args(argv)
    args.python = args.python or ("python3.12" if args.evo2 else sys.executable)
    try:
        py = install_evo2(args) if args.evo2 else install_cpu(args)
    except SetupError as err:
        print(f"[setup] ERROR: {err}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit("[setup] interrupted")
    log("done." + (f" Activate with: source {Path(py).parent}/activate." if py != sys.executable else "")
        + " Check the pipeline with: python 6_validate_evo2.py --test")


if __name__ == "__main__":
    main()
