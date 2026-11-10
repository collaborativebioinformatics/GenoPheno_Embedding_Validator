#!/usr/bin/env bash
# Evo 2 install (no conda) for NVIDIA RTX PRO 6000 Blackwell or L40S
# Both GPUs use the light install: evo2_7b models, no Transformer Engine.
#
# Usage:
#   bash install_evo2.sh            # auto-detects the GPU
#   bash install_evo2.sh rtx        # force RTX PRO 6000 settings
#   bash install_evo2.sh l40s       # force L40S settings

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Detect GPU and choose settings
# ---------------------------------------------------------------------------
GPU_CHOICE="${1:-auto}"

if [ "$GPU_CHOICE" = "auto" ]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
  case "$CAP" in
    12.0) GPU_CHOICE="rtx" ;;
    8.9)  GPU_CHOICE="l40s" ;;
    *) echo "Unrecognized compute capability '$CAP'. Run with 'rtx' or 'l40s'."; exit 1 ;;
  esac
fi

case "$GPU_CHOICE" in
  rtx)
    # RTX PRO 6000 Blackwell: sm_120. Needs CUDA 12.8+.
    # Prebuilt flash-attn wheels often lack sm_120, so always compile from source.
    ARCH="12.0"
    FLASH_ATTN_EXTRA="--no-binary flash-attn"
    ;;
  l40s)
    # L40S: Ada Lovelace, sm_89. Prebuilt flash-attn wheels usually work,
    # so pip tries one first and only compiles if none matches.
    ARCH="8.9"
    FLASH_ATTN_EXTRA=""
    ;;
  *) echo "Unknown option '$GPU_CHOICE'. Use 'rtx' or 'l40s'."; exit 1 ;;
esac

echo ">>> Installing for: $GPU_CHOICE (sm_${ARCH/./})"
nvidia-smi | grep "CUDA Version"   # must be 12.8 or higher

# ---------------------------------------------------------------------------
# 1. CUDA Toolkit 12.8 (system-wide, same for both GPUs)
# ---------------------------------------------------------------------------
UBUNTU_REPO="ubuntu$(lsb_release -rs | tr -d .)"   # e.g. ubuntu2204 or ubuntu2404
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8 build-essential

# ---------------------------------------------------------------------------
# 2. CUDA_HOME (same for both GPUs; only added to ~/.bashrc once)
# ---------------------------------------------------------------------------
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
  echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
  echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version   # should report 12.8

# ---------------------------------------------------------------------------
# 3. Python environment (same for both GPUs)
# ---------------------------------------------------------------------------
python3.12 -m venv ~/evo2-env
source ~/evo2-env/bin/activate
pip install --upgrade pip setuptools wheel ninja packaging psutil

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 4. Flash Attention (differs by GPU: see settings in step 0)
# ---------------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="$ARCH"
MAX_JOBS="${MAX_JOBS:-4}" pip install flash-attn==2.8.0.post2 --no-build-isolation $FLASH_ATTN_EXTRA

# ---------------------------------------------------------------------------
# 5. Evo 2 light install (same for both GPUs; no Transformer Engine)
# ---------------------------------------------------------------------------
[ -d evo2 ] || git clone https://github.com/arcinstitute/evo2
pip install -e ./evo2

# ---------------------------------------------------------------------------
# 6. Other packages
# ---------------------------------------------------------------------------
pip install numpy pandas scipy xgboost

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
python -c "import torch, flash_attn; print('torch', torch.__version__, '| capability', torch.cuda.get_device_capability(), '| flash-attn', flash_attn.__version__)"
python -m evo2.test.test_evo2_generation --model_name evo2_7b

echo ">>> Done. Activate later with: source ~/evo2-env/bin/activate"