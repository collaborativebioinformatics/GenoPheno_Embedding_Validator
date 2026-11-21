#!/usr/bin/env bash
# Evo 2 install (no conda) for the part 1 pipeline, on Ubuntu with an NVIDIA GPU.
# Light install: evo2_7b models, no Transformer Engine. Tested settings: RTX PRO 6000 Blackwell
# (sm_120) and L40S (sm_89); any GPU with compute capability >= 8.0 (Ampere or newer) should work.
#
# Usage:
#   bash install_evo2.sh            # auto-detects the GPU
#   bash install_evo2.sh rtx        # force RTX PRO 6000 / Blackwell settings (sm_120)
#   bash install_evo2.sh l40s       # force L40S settings (sm_89)
#
# Environment variables (optional):
#   PYTHON=python3.12        Python used for the venv (3.11 or 3.12)
#   VENV=~/evo2-env          where the venv is created
#   EVO2_SRC=~/evo2-src      where the evo2 repository is cloned
#   MAX_JOBS=4               parallel compile jobs for flash-attn (more = faster, more RAM)
#   SKIP_CUDA_TOOLKIT=1      do not apt-install the CUDA 12.8 toolkit (e.g. it is already installed)
#   SKIP_EVO2_TEST=1         skip the final generation test (it downloads the 7B weights, ~15 GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-$HOME/evo2-env}"
EVO2_SRC="${EVO2_SRC:-$HOME/evo2-src}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Checks, detect GPU and choose settings
# ---------------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || fail "Evo 2 needs Linux with an NVIDIA GPU. On other systems, test the pipeline with --mock-evo2 (see DOCUMENTATION.md)."
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: install the NVIDIA driver first (CUDA 12.8 capable, driver >= 570)."
command -v "$PYTHON" >/dev/null || fail "$PYTHON not found: install Python 3.12 (or set PYTHON=python3.11)."

GPU_CHOICE="${1:-auto}"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
case "$GPU_CHOICE" in
  auto) ARCH="$CAP" ;;
  rtx)  ARCH="12.0" ;;
  l40s) ARCH="8.9" ;;
  *) fail "Unknown option '$GPU_CHOICE'. Use 'rtx', 'l40s', or nothing to auto-detect." ;;
esac
MAJOR="${ARCH%%.*}"
[ "$MAJOR" -ge 8 ] 2>/dev/null || fail "GPU compute capability '$ARCH' is too old: flash-attn needs 8.0 (Ampere) or newer."
if [ "$MAJOR" -ge 12 ]; then
  # Blackwell (sm_120): prebuilt flash-attn wheels often lack it, so always compile from source.
  FLASH_ATTN_EXTRA="--no-binary flash-attn"
else
  # Ampere / Ada / Hopper: prebuilt wheels usually work; pip only compiles if none matches.
  FLASH_ATTN_EXTRA=""
fi

echo ">>> Installing for compute capability $ARCH (sm_${ARCH/./})"
nvidia-smi | grep "CUDA Version"   # must be 12.8 or higher

# ---------------------------------------------------------------------------
# 1. CUDA Toolkit 12.8 (system-wide)
# ---------------------------------------------------------------------------
if [ "${SKIP_CUDA_TOOLKIT:-0}" != "1" ] && [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  . /etc/os-release
  [ "${ID:-}" = "ubuntu" ] || fail "automatic CUDA toolkit install supports Ubuntu only; install CUDA 12.8 yourself and rerun with SKIP_CUDA_TOOLKIT=1"
  UBUNTU_REPO="ubuntu$(echo "$VERSION_ID" | tr -d .)"           # e.g. ubuntu2204 or ubuntu2404
  case "$(uname -m)" in
    x86_64)  CUDA_ARCH="x86_64" ;;
    aarch64) CUDA_ARCH="sbsa" ;;
    *) fail "unsupported CPU architecture $(uname -m)" ;;
  esac
  TMP_DEB="$(mktemp -d)/cuda-keyring_1.1-1_all.deb"
  wget -q -O "$TMP_DEB" "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_REPO}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$TMP_DEB"
  sudo apt update
  sudo apt install -y cuda-toolkit-12-8 build-essential
fi

# ---------------------------------------------------------------------------
# 2. CUDA_HOME (only added to ~/.bashrc once)
# ---------------------------------------------------------------------------
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc 2>/dev/null; then
  echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
  echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version   # should report 12.8

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel ninja packaging psutil

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 4. Flash Attention (compiled from source on Blackwell, see step 0)
# ---------------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="$ARCH"
# shellcheck disable=SC2086
MAX_JOBS="${MAX_JOBS:-4}" pip install flash-attn==2.8.0.post2 --no-build-isolation $FLASH_ATTN_EXTRA

# ---------------------------------------------------------------------------
# 5. Evo 2 light install (no Transformer Engine)
# ---------------------------------------------------------------------------
[ -d "$EVO2_SRC" ] || git clone https://github.com/arcinstitute/evo2 "$EVO2_SRC"
pip install -e "$EVO2_SRC"

# ---------------------------------------------------------------------------
# 6. Pipeline packages (part 1)
# ---------------------------------------------------------------------------
pip install -r "$SCRIPT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
python -c "import torch, flash_attn; print('torch', torch.__version__, '| capability', torch.cuda.get_device_capability(), '| flash-attn', flash_attn.__version__)"
if [ "${SKIP_EVO2_TEST:-0}" != "1" ]; then
  python -m evo2.test.test_evo2_generation --model_name evo2_7b
fi

echo ">>> Done. Activate later with: source $VENV/bin/activate"
