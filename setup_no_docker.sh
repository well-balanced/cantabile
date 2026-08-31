#!/usr/bin/env bash
# Set up a worker without Docker — for a Kubernetes pod, or any machine where
# containers are not available.
#
#   bash setup_no_docker.sh
#   . .venv/bin/activate && cd rl
#   WANDB_API_KEY=... HF_TOKEN=... python worker.py --gpu 0
#
# Idempotent: every step checks whether it has already been done, so re-running after
# a failure picks up where it stopped rather than redoing the 7 GB pip install.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
MENAGERIE_COMMIT=1afc8be64233dcfe943b2fe0c505ec1e87a0a13e   # pinned by env/scripts/install_deps.sh

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. system libraries -----------------------------------------------------
# EGL is what lets MuJoCo render without a display. Without it the run does not
# error — it renders black frames and completes, which is worse. PyAudio is the one
# dependency in the tree with no wheel, so it needs headers and a compiler.
say "system libraries"
PKGS="libegl1 libgles2 libglib2.0-0 libosmesa6 libsndfile1 fluidsynth build-essential portaudio19-dev"
if [[ $EUID -eq 0 ]]; then APT=""; elif command -v sudo >/dev/null; then APT="sudo"; else APT=""; fi
if command -v apt-get >/dev/null; then
  if [[ $EUID -ne 0 && -z "${APT}" ]]; then
    warn "not root and no sudo — skipping. Install these yourself if anything below fails:"
    warn "  $PKGS"
  else
    $APT apt-get update -qq
    $APT apt-get install -y --no-install-recommends $PKGS
  fi
else
  warn "no apt-get; install the equivalents of: $PKGS"
fi

# --- 2. python environment ---------------------------------------------------
say "python environment"
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1090
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip

if ! python -c "import jax" 2>/dev/null; then
  pip install -q "jax[cuda12]==0.6.2"
fi
if ! python -c "import flax, wandb, tyro" 2>/dev/null; then
  pip install -q -r rl/requirements.txt
fi
if ! python -c "import mujoco, dm_control, note_seq, huggingface_hub, torch" 2>/dev/null; then
  # torch is only used to write .pt state_dicts at export time — training is JAX — so
  # the CPU wheel is deliberate. The +cpu local version outranks plain PyPI's, so this
  # picks the CPU build and pulls no nvidia-* packages.
  pip install -q mujoco dm_control note_seq pretty_midi soundfile huggingface_hub \
      torch --extra-index-url https://download.pytorch.org/whl/cpu
fi
python -c "import robopianist" 2>/dev/null || pip install -q -e ./env

# --- 3. assets git does not carry --------------------------------------------
# Three directories are gitignored but required at runtime. Two can be fetched from
# their upstream sources; the third cannot legally be redistributed.
say "assets"

SF_DIR=env/robopianist/soundfonts
if [[ ! -f "$SF_DIR/TimGM6mb.sf2" ]]; then
  echo "  fetching soundfonts"
  mkdir -p "$SF_DIR"
  wget -qO "$SF_DIR/TimGM6mb.sf2" \
    https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw
else
  echo "  soundfonts ok"
fi

HAND_DIR=env/robopianist/models/hands/third_party/shadow_hand
if [[ ! -d "$HAND_DIR" ]]; then
  echo "  fetching shadow hand model"
  # Sparse + blobless clone of just the one directory. A full menagerie checkout is
  # 688 MB and nothing under env/robopianist/ references any of the rest of it.
  tmp=$(mktemp -d)
  git clone -q --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git "$tmp/m"
  git -C "$tmp/m" sparse-checkout set shadow_hand
  git -C "$tmp/m" checkout -q "$MENAGERIE_COMMIT"
  mkdir -p "$HAND_DIR"
  cp -r "$tmp/m/shadow_hand/." "$HAND_DIR/"
  rm -rf "$tmp"
else
  echo "  shadow hand ok"
fi

PIG_DIR=env/robopianist/music/data/pig_single_finger
PIG_N=$(ls "$PIG_DIR"/*.proto 2>/dev/null | wc -l)
if [[ "$PIG_N" -lt 100 ]]; then
  warn "PIG corpus missing ($PIG_DIR: $PIG_N files, expected 150)"
  cat <<'EOF'
  These are derived from the PIG dataset and keep its original licence, so they are
  not redistributed with this repo or any public image. Two ways to get them:

    copy from a machine that already has them (1.6 MB):
      rsync -a <host>:<path>/cantabile/env/robopianist/music/data/pig_single_finger/ \
            env/robopianist/music/data/pig_single_finger/

    or regenerate from the dataset, downloaded from its official site:
      robopianist preprocess --dataset-dir <PianoFingeringDataset_v1.2> \
                             --save-dir env/robopianist/music/data/pig_single_finger

  Without them the environment cannot load the evaluation songs. preflight catches
  this before any cell is claimed, so nothing is wasted — but no work gets done.
EOF
else
  echo "  PIG corpus ok ($PIG_N songs)"
fi

# --- 4. verify ---------------------------------------------------------------
say "verify"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PY'
import sys
import jax
d = jax.devices()[0]
print(f"  jax      : {d.platform} ({d})")
if d.platform != "gpu":
    print("  !! JAX sees no GPU — training would run on CPU")
try:
    from robopianist import suite
    e = suite.load(environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0", seed=0)
    e.reset()
    print("  env      : loads and renders")
except Exception as ex:
    print(f"  !! env failed: {type(ex).__name__}: {ex}")
    sys.exit(1)
PY

cat <<EOF

Setup complete. To start workers on this machine:

  . $VENV/bin/activate
  export WANDB_API_KEY=... HF_TOKEN=...
  export WANDB_ENTITY=cantabile WANDB_PROJECT=cantabile
  export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd rl
  for i in 1 2 3 4; do python worker.py --gpu 0 & done

XLA_PYTHON_CLIENT_PREALLOCATE=false is what makes several workers fit on one GPU —
without it the first JAX process takes about three quarters of the card. A run uses
roughly 800 MiB, so memory is never the limit; compute is.

--gpu indexes what this machine can see. Inside a pod with one GPU allocated that is
always 0, whichever physical card it is. Check with nvidia-smi.

Queue: https://huggingface.co/datasets/well-balanced/cantabile-runs
EOF
