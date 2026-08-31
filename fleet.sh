#!/usr/bin/env bash
# Start training workers on this machine. Intended to be the only thing you run on a
# new box:
#
#   curl -fsSL https://raw.githubusercontent.com/well-balanced/cantabile/eval51/fleet.sh -o fleet.sh
#   WANDB_API_KEY=... HF_TOKEN=... bash fleet.sh
#
# Everything the training needs is in the image. The one thing that cannot be shipped
# in an image is the host's NVIDIA plumbing, so this checks for it and tells you the
# exact command if it is missing, rather than failing later with a black render or a
# CPU-only JAX.
set -euo pipefail

IMAGE="${IMAGE:-wellbalanced/cantabile:v1}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"
GPUS="${GPUS:-all}"          # "all", or a comma list like "0,2"
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

# --- host prerequisites ------------------------------------------------------
command -v docker >/dev/null || die "docker is not installed.
  Ubuntu:  curl -fsSL https://get.docker.com | sh"

command -v nvidia-smi >/dev/null || die "no NVIDIA driver found (nvidia-smi missing)."

# The container toolkit is the one piece that must exist on the host: it is what makes
# the driver visible inside the container. Test it for real rather than checking for a
# package name, since a present-but-unconfigured install fails the same way.
if ! docker run --rm --gpus all "$IMAGE" -c 'import jax,sys; sys.exit(0 if jax.devices()[0].platform=="gpu" else 1)' \
     --entrypoint python >/dev/null 2>&1; then
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    die "NVIDIA Container Toolkit is not installed or not configured.
  Ubuntu:
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
fi

# --- which GPUs --------------------------------------------------------------
if [[ "$GPUS" == "all" ]]; then
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  IFS=',' read -ra GPU_LIST <<< "$GPUS"
fi
[[ ${#GPU_LIST[@]} -gt 0 ]] || die "no GPUs selected"

mkdir -p "$SCRATCH"
echo "image   : $IMAGE"
echo "gpus    : ${GPU_LIST[*]}  x $SLOTS_PER_GPU slots"
echo "arms    : $METHODS   seed: $SEED"
echo "scratch : $SCRATCH"
echo

docker pull "$IMAGE"

for gpu in "${GPU_LIST[@]}"; do
  for slot in $(seq 1 "$SLOTS_PER_GPU"); do
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

$(( ${#GPU_LIST[@]} * SLOTS_PER_GPU )) worker(s) running.

  docker logs -f cantabile-g${GPU_LIST[0]}-s1     follow one
  docker ps --filter name=cantabile               list
  docker rm -f \$(docker ps -q --filter name=cantabile)   stop all

Queue: https://huggingface.co/datasets/well-balanced/cantabile-runs
A worker that finds nothing claimable sleeps and re-polls, so it is fine to leave
these running while the queue is empty.
EOF
