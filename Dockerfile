# Training worker for the Cantabile dynamics study.
#
# Layers are ordered slowest-changing first so that editing a training script ships a
# 2 MB layer rather than 10 GB: CUDA, then the pip stack, then the assets git does not
# carry, and the source last.
#
# Build:  docker build -t ghcr.io/well-balanced/cantabile:v1 .
# Run:    docker run --rm --gpus '"device=0"' \
#           -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e HF_TOKEN \
#           -v /scratch/cantabile:/work/rl/tmp \
#           ghcr.io/well-balanced/cantabile:v1 --gpu 0
#
# Passing `-e VAR` with no value forwards the host's value, which keeps tokens out of
# the command line and shell history. Never bake them into a layer — they would persist
# in the image forever, and the run-queue repo is public.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# EGL is what makes MuJoCo render headlessly. libegl1 + libgles2 are the client
# libraries; the driver side arrives through the NVIDIA container toolkit at `--gpus`
# time, which is why this image needs no driver of its own. Without these the run does
# not error — it renders black frames and completes.
#
# build-essential + portaudio19-dev are for PyAudio, which robopianist requires and
# which ships no wheel — pip builds it from an sdist, and without the headers the
# image build fails partway through the pip layer. (Verified by resolving the
# dependency graph: PyAudio 0.2.14 is the only sdist in the whole tree.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libegl1 libgles2 libglib2.0-0 libosmesa6 \
        libsndfile1 fluidsynth \
        build-essential portaudio19-dev \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /work

# --- pip stack: the bulk of the image, and the layer that should almost never rebuild
COPY rl/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
        "jax[cuda12]==0.6.2" \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir \
        mujoco dm_control note_seq pretty_midi soundfile \
        huggingface_hub torch --extra-index-url https://download.pytorch.org/whl/cpu

# torch is CPU-only on purpose: it is used solely to write .pt state_dicts at export
# time. Training is JAX. A CUDA torch build would add several GB that never executes.

# --- assets git does not carry (12 MB total)
#
# Three directories are gitignored but required at runtime, and a container built from
# a clean checkout fails without them. Note what is NOT here: env/third_party/
# mujoco_menagerie is 688 MB and nothing under env/robopianist references it — the hand
# model resolves to models/hands/third_party/shadow_hand. A "clone all submodules" build
# step would inflate every pull by 60x the assets actually loaded.
COPY env/robopianist/models/hands/third_party /work/env/robopianist/models/hands/third_party
COPY env/robopianist/soundfonts              /work/env/robopianist/soundfonts
COPY env/robopianist/music/data/pig_single_finger /work/env/robopianist/music/data/pig_single_finger

# --- source: last, so a code change invalidates only this layer
COPY env /work/env
COPY rl  /work/rl

RUN python -m pip install --no-cache-dir -e /work/env

ENV MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_DIR=/work/rl \
    PYTHONPATH=/work/rl

# MUJOCO_EGL_DEVICE_ID indexes the GPUs *visible to the container*, so it is 0 whenever
# a single device is passed with --gpus. Setting it to the host's GPU number is the
# most common way to get a black render.

WORKDIR /work/rl
ENTRYPOINT ["python", "worker.py"]
CMD ["--gpu", "0"]
