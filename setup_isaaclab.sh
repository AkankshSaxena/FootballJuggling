#!/bin/bash
# Ubuntu 22.04 / Brev setup for Unitree RL Lab on Isaac Sim 5.1.
# Usage: bash setup_isaaclab.sh [--skip-driver] [--skip-sim]

set -e
LOGFILE="$HOME/isaaclab_setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

SKIP_DRIVER=false
SKIP_SIM=false
for arg in "$@"; do
  case "$arg" in
    --skip-driver) SKIP_DRIVER=true ;;
    --skip-sim) SKIP_SIM=true ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

DRIVER_VERSION=580
ISAAC_SIM_VERSION=5.1.0
ISAAC_SIM_URL="https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-${ISAAC_SIM_VERSION}-linux-x86_64.zip"
ISAAC_SIM_DIR="$HOME/isaacsim"
WORK_DIR="$HOME/isaaclab"
ISAACLAB_DIR="$WORK_DIR/IsaacLab"
ISAACLAB_VERSION=v2.3.2
UNITREE_RL_LAB_DIR="${UNITREE_RL_LAB_DIR:-$WORK_DIR/unitree_rl_lab}"
UNITREE_MODEL_DIR="${UNITREE_MODEL_DIR:-$WORK_DIR/unitree_model}"
G1_USD="G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
CONDA_ENV=isaaclab
PYTHON_VERSION=3.11
TORCH_VERSION=2.7.0+cu128
TORCHVISION_VERSION=0.22.0+cu128
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TENSORDICT_VERSION=0.8.3

step() { printf '\n== %s ==\n' "$1"; }
die() { printf '%s\n' "$1" >&2; exit 1; }

step "NVIDIA driver and system packages"
sudo apt-get update

if ! $SKIP_DRIVER; then
  current_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  if [[ "${current_driver%%.*}" != "$DRIVER_VERSION" ]]; then
    sudo apt-get -y -t jammy-updates install --no-install-recommends "nvidia-driver-$DRIVER_VERSION"
    echo "Driver $DRIVER_VERSION installed. Run sudo reboot, then rerun this script."
    exit 0
  fi
  echo "NVIDIA driver $current_driver is active."
fi

sudo apt-get install -y \
  git git-lfs wget curl unzip tmux \
  build-essential cmake \
  libgl1 libegl1 libglu1-mesa \
  libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
  libvulkan1 mesa-vulkan-drivers vulkan-tools \
  xfce4 xfce4-goodies

# Let Vulkan discover the installed NVIDIA manifest. The old setup exported
# this one fixed path even when it did not exist.
if [[ "${VK_ICD_FILENAMES:-}" == /usr/share/vulkan/icd.d/nvidia_icd.json &&
      ! -f "$VK_ICD_FILENAMES" ]]; then
  unset VK_ICD_FILENAMES
fi

step "Miniconda and Python $PYTHON_VERSION"
if [[ ! -x "$HOME/miniconda3/bin/conda" ]]; then
  installer=$(mktemp --suffix=.sh)
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
  bash "$installer" -b -p "$HOME/miniconda3"
  rm "$installer"
fi

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda init bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | grep -q "^$CONDA_ENV "; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
fi
conda activate "$CONDA_ENV"
active_python=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "$active_python" == "$PYTHON_VERSION" ]] ||
  die "Conda env $CONDA_ENV uses Python $active_python. This setup needs $PYTHON_VERSION."

step "Isaac Sim $ISAAC_SIM_VERSION"
# Unitree RL Lab uses Isaac Lab, which still needs the Isaac Sim runtime.
if [[ ! -f "$ISAAC_SIM_DIR/isaac-sim.sh" || ! -f "$ISAAC_SIM_DIR/setup_conda_env.sh" ]]; then
  if $SKIP_SIM; then
    die "--skip-sim requires an Isaac Sim install at $ISAAC_SIM_DIR."
  fi
  mkdir -p "$ISAAC_SIM_DIR"
  archive="$ISAAC_SIM_DIR/isaac-sim.zip"
  wget -c --show-progress "$ISAAC_SIM_URL" -O "$archive"
  unzip -oq "$archive" -d "$ISAAC_SIM_DIR"
  (cd "$ISAAC_SIM_DIR" && ./post_install.sh)
  rm "$archive"
fi

step "Isaac Lab $ISAACLAB_VERSION and Unitree RL Lab"
mkdir -p "$WORK_DIR"
if [[ -e "$ISAACLAB_DIR/.git" ]]; then
  if ! git -C "$ISAACLAB_DIR" rev-parse --verify --quiet "refs/tags/$ISAACLAB_VERSION" >/dev/null; then
    git -C "$ISAACLAB_DIR" fetch origin "refs/tags/$ISAACLAB_VERSION:refs/tags/$ISAACLAB_VERSION"
  fi
  git -C "$ISAACLAB_DIR" checkout --quiet "$ISAACLAB_VERSION"
else
  git clone --branch "$ISAACLAB_VERSION" --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

if [[ ! -e "$UNITREE_RL_LAB_DIR/.git" ]]; then
  git clone https://github.com/unitreerobotics/unitree_rl_lab.git "$UNITREE_RL_LAB_DIR"
fi
printf 'Isaac Lab commit: %s\n' "$(git -C "$ISAACLAB_DIR" rev-parse --short HEAD)"
printf 'Unitree RL Lab commit: %s\n' "$(git -C "$UNITREE_RL_LAB_DIR" rev-parse --short HEAD)"

# Isaac Lab v2.3.2 ships the published-checkpoint helper in isaaclab_rl.
sed -i 's/^from isaaclab\.utils\.pretrained_checkpoint import /from isaaclab_rl.utils.pretrained_checkpoint import /' \
  "$UNITREE_RL_LAB_DIR/scripts/rsl_rl/play.py"

cd "$ISAACLAB_DIR"
if [[ -L _isaac_sim || ! -e _isaac_sim ]]; then
  ln -sfnT "$ISAAC_SIM_DIR" _isaac_sim
elif [[ ! _isaac_sim -ef "$ISAAC_SIM_DIR" ]]; then
  die "$ISAACLAB_DIR/_isaac_sim already exists and points to a different installation."
fi
source "$ISAAC_SIM_DIR/setup_conda_env.sh"

step "Install Isaac Lab and Unitree RL Lab in $CONDA_ENV"
# flatdict needs pkg_resources at build time. Install the core with the
# pinned build tools before asking the launcher to install the extensions.
python -m pip install 'setuptools<81' wheel
python -m pip install --ignore-installed toml
python -m pip install --index-url "$TORCH_INDEX" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
python -m pip install --no-build-isolation -e source/isaaclab
# The launcher also reinstalls the core extension through a child pip process.
# Carry the no-build-isolation option into that process as well.
PIP_NO_BUILD_ISOLATION=0 ./isaaclab.sh --install rsl_rl

# Install the Unitree package in editable mode. Use the active environment
# directly so setup works without relying on the upstream autocomplete hook.
python -m pip install -e "$UNITREE_RL_LAB_DIR/source/unitree_rl_lab" argcomplete wandb
python -m pip install --index-url "$TORCH_INDEX" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
# Apply the known working pin after all other package installs.
python -m pip install --no-deps "tensordict==$TENSORDICT_VERSION"

step "Configure Unitree G1 29-DOF USD assets"
unitree_config="$UNITREE_RL_LAB_DIR/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
if [[ ! -s "$UNITREE_MODEL_DIR/$G1_USD" ]] ||
   head -c 64 "$UNITREE_MODEL_DIR/$G1_USD" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  if [[ ! -e "$UNITREE_MODEL_DIR/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
      https://huggingface.co/datasets/unitreerobotics/unitree_model "$UNITREE_MODEL_DIR"
  fi
  git -C "$UNITREE_MODEL_DIR" lfs pull
fi
if [[ ! -s "$UNITREE_MODEL_DIR/$G1_USD" ]] ||
   head -c 64 "$UNITREE_MODEL_DIR/$G1_USD" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  die "G1 model is missing or still an LFS pointer at $UNITREE_MODEL_DIR/$G1_USD."
fi

# Set the upstream placeholder only. Keep any custom Unitree configuration
# already present in a checkout supplied by the user.
python - "$unitree_config" "$UNITREE_MODEL_DIR" <<'PYTHON'
from pathlib import Path
import re
import sys

config = Path(sys.argv[1])
model_dir = str(Path(sys.argv[2]).resolve())
source = config.read_text(encoding="utf-8")
pattern = r'^(UNITREE_MODEL_DIR\s*=\s*)["\x27]path/to/unitree_model["\x27]'
updated = re.sub(pattern, lambda match: match[1] + repr(model_dir), source, count=1, flags=re.MULTILINE)
if updated != source:
    config.write_text(updated, encoding="utf-8")
PYTHON

step "Verify Python stack and run a headless Unitree G1 smoke test"
python - "$TORCH_VERSION" "$TENSORDICT_VERSION" <<'PYTHON'
import sys
import tensordict
import torch

if torch.__version__ != sys.argv[1] or tensordict.__version__ != sys.argv[2]:
    raise SystemExit(f"Unexpected package versions: torch={torch.__version__}, tensordict={tensordict.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU. Check nvidia-smi and reboot if the driver changed.")
PYTHON

cd "$UNITREE_RL_LAB_DIR"
if ! timeout --kill-after=30s 600 python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity --headless --num_envs 16 --max_iterations 2; then
  die "Unitree G1 smoke test failed. Check $LOGFILE."
fi

step "Setup complete"
printf 'Isaac Sim       %s (%s)\n' "$ISAAC_SIM_DIR" "$ISAAC_SIM_VERSION"
printf 'Isaac Lab       %s (%s)\n' "$ISAACLAB_DIR" "$ISAACLAB_VERSION"
printf 'Unitree RL Lab  %s\n' "$UNITREE_RL_LAB_DIR"
printf 'Unitree tasks   %s/source/unitree_rl_lab/unitree_rl_lab/tasks\n' "$UNITREE_RL_LAB_DIR"
printf 'Conda env       %s\n' "$CONDA_ENV"
printf 'Log file        %s\n' "$LOGFILE"
printf '\nTo load Isaac Sim in a new shell after conda activate %s:\n' "$CONDA_ENV"
printf 'source %q\n' "$ISAAC_SIM_DIR/setup_conda_env.sh"
