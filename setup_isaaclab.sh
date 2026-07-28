#!/bin/bash
# =============================================================================
# Isaac Lab Setup Script — A6000 / Brev
# Idempotent: safe to re-run. Skips steps already done.
# Usage: bash setup_isaaclab.sh [--skip-driver] [--skip-sim]
# =============================================================================

set -e
LOGFILE="$HOME/isaaclab_setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_DRIVER=false
SKIP_GUI=false
SKIP_SIM=false
for arg in "$@"; do
  case $arg in
    --skip-driver) SKIP_DRIVER=true ;;
    --skip-sim)    SKIP_SIM=true ;;  # skip Isaac Sim download (already done)
  esac
done

# ── Config — edit these if needed ─────────────────────────────────────────────
DRIVER_VERSION="550"
ISAAC_SIM_VERSION="5.1.0"
ISAAC_SIM_URL="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-${ISAAC_SIM_VERSION}-linux-x86_64.zip"
ISAAC_SIM_DIR="$HOME/isaacsim"
ISAACLAB_DIR="$HOME/isaaclab/IsaacLab"
# PINNED Isaac Lab version. Do NOT track `main` — it moved to the 3.0 line
# (Python 3.12 / torch 2.10 / Isaac Sim 6.0) and will not install against this
# 3.11 env + Isaac Sim 5.1. v2.3.2 is the last release on the 2.3 (Isaac Sim
# 4.5/5.0/5.1) line. Bump this ONLY together with ISAAC_SIM_VERSION + PYTHON_VERSION.
ISAACLAB_VERSION="v2.3.2"
CONDA_ENV="isaaclab"
PYTHON_VERSION="3.11"
WORK_DIR="$HOME/isaaclab"

# ── Helpers ───────────────────────────────────────────────────────────────────
green()  { echo -e "\e[32m[✓] $1\e[0m"; }
yellow() { echo -e "\e[33m[~] $1\e[0m"; }
red()    { echo -e "\e[31m[✗] $1\e[0m"; }
step()   { echo -e "\n\e[1m\e[36m══ $1 ══\e[0m"; }

# =============================================================================
# STEP 1: NVIDIA Driver
# =============================================================================
step "STEP 1: NVIDIA Driver"

if $SKIP_DRIVER; then
  yellow "Skipping driver install (--skip-driver)"
else
  CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo "none")
  if [[ "$CURRENT_DRIVER" == "$DRIVER_VERSION" ]]; then
    green "Driver $DRIVER_VERSION already installed"
  else
    yellow "Current driver: $CURRENT_DRIVER — installing $DRIVER_VERSION"
    sudo apt purge -y 'nvidia*' 2>/dev/null || true
    sudo apt autoremove -y
    sudo apt install -y nvidia-driver-${DRIVER_VERSION}
    red "Driver installed. REBOOT REQUIRED → run: sudo reboot"
    red "After reboot, re-run this script with --skip-driver"
    exit 0
  fi
fi

# =============================================================================
# STEP 2: System Dependencies
# =============================================================================
step "STEP 2: System Dependencies"

sudo apt update   # no upgrade — it causes issues on cloud VMs

sudo apt install -y \
  git wget curl unzip tmux \
  build-essential cmake \
  python3.11 python3.11-venv python3-pip \
  libgl1 libegl1 libglu1-mesa \
  libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
  libvulkan1 mesa-vulkan-drivers vulkan-tools \
  xfce4 xfce4-goodies

green "System deps installed"

# ── Vulkan check ──────────────────────────────────────────────────────────────
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  yellow "nvidia_icd.json missing — installing nvidia-utils"
  sudo apt install -y nvidia-utils-${DRIVER_VERSION} 2>/dev/null || \
  sudo apt install -y nvidia-utils-565 2>/dev/null || true
fi

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# Make it persistent
grep -qxF "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" ~/.bashrc || \
  echo "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" >> ~/.bashrc

green "Vulkan configured"

# =============================================================================
# STEP 3: Miniconda
# =============================================================================
step "STEP 3: Miniconda"

if [ -d "$HOME/miniconda3" ]; then
  green "Miniconda already installed"
else
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $HOME/miniconda3
  rm /tmp/miniconda.sh
  green "Miniconda installed"
fi

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accept ToS silently
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Init conda in bashrc if not already
grep -qxF ". $HOME/miniconda3/etc/profile.d/conda.sh" ~/.bashrc || \
  conda init bash

# =============================================================================
# STEP 4: Conda Environment
# =============================================================================
step "STEP 4: Conda Environment ($CONDA_ENV)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

if conda env list | grep -q "^$CONDA_ENV "; then
  green "Conda env '$CONDA_ENV' already exists"
else
  conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
  green "Conda env '$CONDA_ENV' created"
fi

conda activate $CONDA_ENV
conda install pip -y

# Guard: v2.3.2 requires Python 3.11. If a stale/wrong env is active, fail loudly
# now instead of hitting the confusing "requires >=3.12" error mid-install.
ACTIVE_PY=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$ACTIVE_PY" != "$PYTHON_VERSION" ]]; then
  red "Active Python is $ACTIVE_PY but $ISAACLAB_VERSION needs $PYTHON_VERSION."
  red "Recreate the env:  conda env remove -n $CONDA_ENV && re-run this script."
  exit 1
fi
green "Python $ACTIVE_PY OK for $ISAACLAB_VERSION"

# =============================================================================
# STEP 5: Isaac Sim Download + Install
# =============================================================================
step "STEP 5: Isaac Sim"

mkdir -p "$HOME/isaacsim"

if $SKIP_SIM; then
  yellow "Skipping Isaac Sim download (--skip-sim)"
elif [ -f "$ISAAC_SIM_DIR/isaac-sim.sh" ]; then
  green "Isaac Sim already installed at $ISAAC_SIM_DIR"
else
  yellow "Downloading Isaac Sim $ISAAC_SIM_VERSION (~10GB — run this inside tmux)"
  cd "$HOME/isaacsim"
  wget -q --show-progress "$ISAAC_SIM_URL" -O isaac-sim.zip
  unzip -q isaac-sim.zip
  rm isaac-sim.zip

  cd "$ISAAC_SIM_DIR"
  ./post_install.sh
  green "Isaac Sim installed"
fi

# =============================================================================
# STEP 6: Clone / Checkout Isaac Lab (PINNED)
# =============================================================================
step "STEP 6: Isaac Lab ($ISAACLAB_VERSION)"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ -d "$ISAACLAB_DIR/.git" ]; then
  green "Isaac Lab repo already cloned — checking out $ISAACLAB_VERSION"
  cd "$ISAACLAB_DIR"
  # Was `git pull` here (tracked main -> drifted to 3.0 line and broke install).
  # Now: fetch tags and hard-pin to the tag. No tracking of main.
  git fetch --tags --quiet origin
  git checkout --quiet "$ISAACLAB_VERSION"
else
  # Shallow clone straight at the pinned tag.
  git clone --branch "$ISAACLAB_VERSION" --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  green "Isaac Lab cloned at $ISAACLAB_VERSION"
fi

# Log exactly what we're on (git hygiene — record the commit).
cd "$ISAACLAB_DIR"
green "Isaac Lab checked out: $(git describe --tags --always) @ $(git rev-parse --short HEAD)"

# =============================================================================
# STEP 7: Link Isaac Sim → Isaac Lab + Install
# =============================================================================
step "STEP 7: Link and Install"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

cd "$ISAACLAB_DIR"

# Symlink
if [ ! -L "_isaac_sim" ]; then
  ln -s "$ISAAC_SIM_DIR" _isaac_sim
  green "Symlinked _isaac_sim → $ISAAC_SIM_DIR"
else
  green "Symlink already exists"
fi

# Source conda env for Isaac Sim
if [ -f "$ISAAC_SIM_DIR/setup_conda_env.sh" ]; then
  source "$ISAAC_SIM_DIR/setup_conda_env.sh"
fi

# Install Isaac Lab
./isaaclab.sh --install

green "Isaac Lab installed"

# =============================================================================
# STEP 8: Extra Python Packages
# =============================================================================
step "STEP 8: Extra Packages (wandb, etc.)"

pip install -q wandb

green "Extra packages installed"

# =============================================================================
# STEP 9: Smoke Test
# =============================================================================
step "STEP 9: Smoke Test"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate $CONDA_ENV

if [ -f "$ISAAC_SIM_DIR/setup_conda_env.sh" ]; then
  source "$ISAAC_SIM_DIR/setup_conda_env.sh"
fi

export DISPLAY=
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

yellow "Running Cartpole test (headless, ~2 min)..."
cd "$ISAACLAB_DIR"
timeout 120 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 50 && green "Cartpole test passed ✓" || red "Cartpole test failed — check $LOGFILE"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║            SETUP COMPLETE                            ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Isaac Sim : $ISAAC_SIM_DIR ($ISAAC_SIM_VERSION)"
echo "║  Isaac Lab : $ISAACLAB_DIR ($ISAACLAB_VERSION)"
echo "║  Conda env : $CONDA_ENV (Python $PYTHON_VERSION)"
echo "║  Log file  : $LOGFILE"
echo "╚══════════════════════════════════════════════════════╝"