#!/usr/bin/env bash
# Set up a worker without Docker and without root — for an unprivileged Kubernetes
# pod, or any host where you cannot install system packages.
#
#   bash setup_no_docker.sh
#   . .venv/bin/activate && cd rl
#   WANDB_API_KEY=... HF_TOKEN=... python worker.py --gpu 0 --no-eval-video
#
# Nothing here needs apt, sudo, or a compiler. That is possible because two things
# were pulled off the required path:
#
#   PyAudio      backs only MidiFile.play() — playback through a local speaker — and
#                ships no Linux wheel. It is now a lazy import and an optional extra,
#                so the package installs with no compiler and no portaudio headers.
#
#   OpenGL       is needed only to render eval videos. The musical metrics come from
#                MidiEvaluationWrapper, which touches no GL context at all, so
#                `--no-eval-video` with MUJOCO_GL=disable trains and measures with no
#                EGL, no OSMesa and no soundfont.
#
# Idempotent: each step checks whether it is already done, so a re-run after a failure
# does not repeat the multi-gigabyte pip install.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# jaxlib 0.6.2 publishes wheels for cp310-cp313 only. On anything outside that, pip
# reports "no matching distribution" for jaxlib rather than naming the interpreter,
# which reads like a broken index. Check it here, where the fix can be stated.
PYV=$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")
case "$PYV" in
  3.10|3.11|3.12|3.13) ;;
  *) die "python $PYV is not supported by jaxlib 0.6.2 (needs 3.10-3.13).
  Point PYTHON at a supported interpreter — with conda, no root needed:
    conda create -y -n cantabile python=3.11
    PYTHON=\"\$(conda info --base)/envs/cantabile/bin/python\" bash setup_no_docker.sh" ;;
esac

# --- 1. python environment ---------------------------------------------------
say "python environment"
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1090
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip

python -c "import jax" 2>/dev/null || pip install -q "jax[cuda12]==0.6.2"
python -c "import flax, wandb, tyro" 2>/dev/null || pip install -q -r rl/requirements.txt
if ! python -c "import mujoco, dm_control, note_seq, huggingface_hub, torch" 2>/dev/null; then
  # torch is the CPU wheel on purpose: it only writes .pt state_dicts at export time,
  # training is JAX. The +cpu local version outranks plain PyPI's, so this selects the
  # CPU build and pulls no nvidia-* packages.
  pip install -q mujoco dm_control note_seq pretty_midi soundfile huggingface_hub \
      torch --extra-index-url https://download.pytorch.org/whl/cpu
fi
python -c "import robopianist" 2>/dev/null || pip install -q -e ./env

# --- 2. the hand model -------------------------------------------------------
# Gitignored, so a fresh clone does not have it, and MuJoCo cannot build the scene
# without it. Fetched sparsely: a full mujoco_menagerie checkout is 688 MB and nothing
# under env/robopianist/ references any of the rest.
say "hand model"
HAND_DIR=env/robopianist/models/hands/third_party/shadow_hand
MENAGERIE_COMMIT=1afc8be64233dcfe943b2fe0c505ec1e87a0a13e   # pinned by env/scripts/install_deps.sh
if [[ -f "$HAND_DIR/right_hand.xml" ]]; then
  echo "  ok"
else
  tmp=$(mktemp -d)
  git clone -q --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git "$tmp/m"
  git -C "$tmp/m" sparse-checkout set shadow_hand
  git -C "$tmp/m" checkout -q "$MENAGERIE_COMMIT"
  mkdir -p "$HAND_DIR"
  cp -r "$tmp/m/shadow_hand/." "$HAND_DIR/"
  rm -rf "$tmp"
  echo "  fetched ($(ls "$HAND_DIR" | wc -l) files)"
fi

# --- 3. the MIDI corpus ------------------------------------------------------
# The only asset that cannot be fetched. Soundfonts are no longer needed at all:
# they exist for eval-video audio, which --no-eval-video does not produce.
say "song data"
PIG_DIR=env/robopianist/music/data/pig_single_finger
PIG_N=$(ls "$PIG_DIR"/*.proto 2>/dev/null | wc -l)
if [[ "$PIG_N" -lt 100 ]]; then
  warn "PIG corpus missing ($PIG_DIR: $PIG_N files, expected 150)"
  cat <<'EOF'
  These are derived from the PIG dataset and keep its original licence, so they are
  not redistributed with this repo. Two ways to get them:

    copy from a machine that has them (1.6 MB):
      rsync -a <host>:<path>/cantabile/env/robopianist/music/data/pig_single_finger/ \
            env/robopianist/music/data/pig_single_finger/

    or regenerate from the dataset, downloaded from its official site:
      robopianist preprocess --dataset-dir <PianoFingeringDataset_v1.2> \
                             --save-dir env/robopianist/music/data/pig_single_finger

  Setup continues without them — everything else is still worth having — but no
  evaluation song can load, and preflight will refuse to claim work.
EOF
else
  echo "  ok ($PIG_N songs)"
fi

# --- 4. verify ---------------------------------------------------------------
# Load an environment for real. "pip returned zero" and "this works" are different
# claims, and the ways this fails are quiet ones.
say "verify"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PY'
import sys
import jax
d = jax.devices()[0]
print(f"  jax  : {d.platform} ({d})")
if d.platform != "gpu":
    print("  !! JAX sees no GPU — training would run on CPU. Check nvidia-smi and that")
    print("     the pod actually has a GPU allocated.")
try:
    from robopianist import suite
    suite.load(environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0", seed=0).reset()
    print("  env  : loads")
except Exception as ex:
    print(f"  !! env failed: {type(ex).__name__}: {ex}")
    sys.exit(1)
PY

cat <<EOF

Setup complete. To start workers:

  . $VENV/bin/activate
  export WANDB_API_KEY=... HF_TOKEN=...
  export WANDB_ENTITY=cantabile WANDB_PROJECT=cantabile
  export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd rl
  for i in 1 2 3 4; do python worker.py --gpu 0 --no-eval-video & done

--no-eval-video keeps every metric and drops only the rollout video. Leave it off and
the run needs a working EGL context, which an unprivileged pod usually cannot provide.

XLA_PYTHON_CLIENT_PREALLOCATE=false is what lets several workers share one GPU —
without it the first JAX process takes about three quarters of the card. A run uses
roughly 800 MiB, so memory is not the limit; compute is, and four per GPU already
reaches about 50% utilisation.

--gpu indexes what this machine can see. In a pod with one GPU allocated that is 0,
whichever physical card it is. Check with nvidia-smi.

Queue: https://huggingface.co/datasets/well-balanced/cantabile-runs
EOF
