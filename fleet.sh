#!/usr/bin/env bash
# Start training workers on this machine. Intended to be the only thing you run on a
# new box:
#
#   curl -fsSL https://raw.githubusercontent.com/well-balanced/cantabile/eval51/fleet.sh -o fleet.sh
#   WANDB_API_KEY=... HF_TOKEN=... GPUS=0,2 bash fleet.sh
#
# GPUS is required and never inferred. These machines are shared, so a launcher that
# detected "all GPUs" would quietly take slots someone else is using. State them.
#
# Everything the training needs is in the image. The one thing that cannot be shipped
# in an image is the host's NVIDIA plumbing, so this checks for it and tells you the
# exact command if it is missing, rather than failing later with a black render or a
# CPU-only JAX.
set -euo pipefail

IMAGE="${IMAGE:-wellbalanced/cantabile:v1}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"   # default when a GPU is given without ":n"
GPUS="${GPUS:-}"                      # REQUIRED. "0,2" or, per-GPU, "0:4,2:2"
SEED="${SEED:-43}"
METHODS="${METHODS:-base}"
SCRATCH="${SCRATCH:-$PWD/cantabile-scratch}"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- credentials, before anything slow happens -------------------------------
[[ -f .env ]] && { set -a; . ./.env; set +a; }
[[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is not set (export it, or put it in ./.env)
  Without it the worker would fall back to wandb offline mode: the run trains to
  completion and its metrics are written inside a container that is then removed."
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set (needs write access to the runs repo)
  Without it the failure appears only at upload, ~15 hours in, with the checkpoints
  about to be deleted along with the container."

# --- which GPUs, and how many slots on each -----------------------------------
# Never inferred. `nvidia-smi` would happily list GPUs that a colleague is mid-run on,
# and four workers landing on one of those is the kind of mistake that is only noticed
# hours later.
[[ -n "$GPUS" ]] || die "GPUS is not set. Name the slots you are allowed to use:
    GPUS=0,2            two GPUs, $SLOTS_PER_GPU slots each
    GPUS=0:4,2:2        GPU0 with 4 slots, GPU2 with 2
  Visible on this host:
$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /')"

declare -a GPU_LIST=() SLOT_LIST=()
IFS=',' read -ra _entries <<< "$GPUS"
for e in "${_entries[@]}"; do
  gpu="${e%%:*}"; slots="${e#*:}"
  [[ "$slots" == "$e" ]] && slots="$SLOTS_PER_GPU"
  [[ "$gpu" =~ ^[0-9]+$ && "$slots" =~ ^[0-9]+$ && "$slots" -ge 1 ]] \
    || die "bad GPUS entry '$e' (expected N or N:slots)"
  nvidia-smi -i "$gpu" >/dev/null 2>&1 || die "GPU $gpu does not exist on this host"
  GPU_LIST+=("$gpu"); SLOT_LIST+=("$slots")
done

# --- host prerequisites ------------------------------------------------------
command -v docker >/dev/null || die "docker is not installed.
  Ubuntu:  curl -fsSL https://get.docker.com | sh"

command -v nvidia-smi >/dev/null || die "no NVIDIA driver found (nvidia-smi missing)."

# The container toolkit is the one piece that must exist on the host: it is what makes
# the driver visible inside the container. Test it for real rather than checking for a
# package name, since a present-but-unconfigured install fails the same way.
# `--entrypoint` is a `docker run` option and must precede the image name; putting it
# after would pass it to the container as an argument and the probe would always fail.
if ! docker run --rm --gpus all --entrypoint python "$IMAGE" \
       -c 'import jax,sys; sys.exit(0 if jax.devices()[0].platform=="gpu" else 1)' \
       >/dev/null 2>&1; then
    die "GPUs are not usable from a container — NVIDIA Container Toolkit missing or unconfigured.
  Ubuntu:
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

  Already installed? Check it is wired into docker:
    docker info | grep -i nvidia
    docker run --rm --gpus all $IMAGE --help"
fi


mkdir -p "$SCRATCH"
_total=0; _plan=""
for i in "${!GPU_LIST[@]}"; do
  _plan+="${GPU_LIST[$i]}x${SLOT_LIST[$i]} "; _total=$(( _total + SLOT_LIST[i] ))
done
echo "image   : $IMAGE"
echo "gpus    : $_plan ($_total worker(s))"
echo "arms    : $METHODS   seed: $SEED"
echo "scratch : $SCRATCH"
echo

docker pull "$IMAGE"

for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  for slot in $(seq 1 "${SLOT_LIST[$i]}"); do
    name="cantabile-g${gpu}-s${slot}"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" --restart unless-stopped \
      --gpus "\"device=${gpu}\"" \
      -e WANDB_API_KEY -e HF_TOKEN \
      -e WANDB_ENTITY="${WANDB_ENTITY:-cantabile}" \
      -e WANDB_PROJECT="${WANDB_PROJECT:-cantabile}" \
      -v "$SCRATCH/g${gpu}s${slot}:/work/rl/tmp" \
      "$IMAGE" --gpu 0 --seed "$SEED" --methods $METHODS >/dev/null
    echo "started $name"
  done
done

cat <<EOF

$_total worker(s) running on GPU(s) ${GPU_LIST[*]}.

  docker logs -f cantabile-g${GPU_LIST[0]}-s1     follow one
  docker ps --filter name=cantabile               list
  docker rm -f \$(docker ps -q --filter name=cantabile)   stop all

Queue: https://huggingface.co/datasets/well-balanced/cantabile-runs
A worker that finds nothing claimable sleeps and re-polls, so it is fine to leave
these running while the queue is empty.
EOF
