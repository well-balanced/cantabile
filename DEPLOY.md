# Running workers on a new machine

Everything the training needs is in the image. A machine that already runs GPU
containers needs nothing installed — one `docker run` per worker slot and it starts
pulling work from the queue.

There is exactly one host prerequisite that an image cannot ship: the **NVIDIA
Container Toolkit**, which is what makes the driver visible inside a container. Most
machines that have ever run a GPU container already have it.

---

## Quick path — the machine is already set up

```bash
export WANDB_API_KEY=...   # entity: cantabile
export HF_TOKEN=...        # needs write access to well-balanced/*

docker pull wellbalanced/cantabile:v1

for g in 4 5; do                       # only the GPUs you are allowed to use
  for i in 1 2 3 4; do                 # slots per GPU
    docker run -d --name cantabile-g${g}-s${i} --restart unless-stopped \
      --gpus "\"device=${g}\"" \
      -e WANDB_API_KEY -e HF_TOKEN \
      -e WANDB_ENTITY=cantabile -e WANDB_PROJECT=cantabile \
      wellbalanced/cantabile:v1 --gpu 0
  done
done
```

That is the whole deployment. No git clone, no Python environment, no dataset
download — the song list, the environment, the soundfonts and the MIDI corpus are all
inside the image, and the work queue is read from HuggingFace.

**`--gpu 0` is always `0`.** `--gpus "device=4"` isolates the container to GPU 4, and
inside the container that GPU is numbered 0. Passing the host's number here is the
usual way to get "no GPU visible". `MUJOCO_EGL_DEVICE_ID` is set to 0 in the image for
the same reason.

**`-e VAR` with no value** forwards the host's environment variable without putting the
secret on the command line, where it would land in shell history and `ps`.

---

## First-time setup on a bare machine

### 1. Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Group membership only applies to *new* logins. Rather than logging out, run commands
through `sg` in the current shell:

```bash
sg docker -c "docker version"
```

### 2. NVIDIA Container Toolkit

Check first — it is often already there:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If that prints the GPU table, skip this step. Otherwise:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

A toolkit that is installed but not wired into docker fails exactly like a missing one,
so verify by running the check above rather than by checking the package is present.

### 3. Credentials

Put them in a file rather than in your shell history:

```bash
cat > .env <<'EOF'
WANDB_API_KEY=...
HF_TOKEN=...
EOF
chmod 600 .env

set -a; . ./.env; set +a
```

`.env` is gitignored. Both are checked before any slow work starts, because each fails
expensively otherwise: a missing `HF_TOKEN` surfaces only at upload, fifteen hours in,
with the checkpoints about to be deleted along with the container; a bad
`WANDB_API_KEY` is worse than a crash, because wandb silently falls back to offline
mode and the run completes with its metrics written somewhere nobody collects.

### 4. Start workers

Use the loop from the quick path above, or the launcher, which adds the same checks
and a per-GPU slot count:

```bash
GPUS=4,5 bash fleet.sh              # 4 slots each (default)
GPUS=4:6,5:6 bash fleet.sh          # 6 slots each
```

`GPUS` is required and never inferred. These machines are shared, and a launcher that
detected "all GPUs" would quietly take slots someone else is mid-run on. With `GPUS`
unset the error lists the host's GPUs with their current memory and utilisation, so
the free ones are visible at the point of choosing.

---

## Kubernetes

Inside a pod you cannot run `docker` — the pod is already a container, and the daemon
is not there. Two paths, depending on what your access allows.

### If you can create workloads: use the image as the pod image

This is the natural fit. Everything the quick path does with `docker run`, Kubernetes
does with replicas.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cantabile-worker
spec:
  replicas: 8                      # one per worker slot
  selector:
    matchLabels: {app: cantabile-worker}
  template:
    metadata:
      labels: {app: cantabile-worker}
    spec:
      containers:
        - name: worker
          image: wellbalanced/cantabile:v1
          args: ["--gpu", "0"]
          resources:
            limits:
              nvidia.com/gpu: 1    # see the note below
          env:
            - {name: WANDB_ENTITY,  value: cantabile}
            - {name: WANDB_PROJECT, value: cantabile}
            - name: WANDB_API_KEY
              valueFrom: {secretKeyRef: {name: cantabile-creds, key: wandb}}
            - name: HF_TOKEN
              valueFrom: {secretKeyRef: {name: cantabile-creds, key: hf}}
          volumeMounts:
            - {name: scratch, mountPath: /work/rl/tmp}
      volumes:
        - name: scratch
          emptyDir: {}
```

```bash
kubectl create secret generic cantabile-creds \
  --from-literal=wandb="$WANDB_API_KEY" --from-literal=hf="$HF_TOKEN"
kubectl apply -f worker.yaml
kubectl logs -f deploy/cantabile-worker
```

Two things differ from the Docker path:

**`nvidia.com/gpu: 1` gives a pod one whole GPU, and it cannot be shared.** The device
plugin allocates GPUs exclusively, so eight replicas means eight GPUs, not eight
workers on two. To put several workers on one GPU — which is the point, since a run
uses under 800 MiB — you need either one pod running several worker processes, or a
cluster with time-slicing or MPS configured. The simplest version is one pod per GPU
with the worker started several times:

```yaml
          command: ["bash", "-c"]
          args:
            - |
              for i in $(seq 1 6); do python worker.py --gpu 0 & done
              wait
```

**`replicas` is not the same as picking GPUs.** Kubernetes chooses which physical GPU
each pod gets; you cannot say "4 and 5" the way `--gpus device=4` does. If particular
cards are reserved for you, that is a scheduling concern — a node selector, a taint, or
whatever your cluster uses — not something the worker controls.

### If you only have a shell in an existing pod: run it directly

No container involved. The pod already has the GPU; it needs the code, the Python
stack, and the assets git does not carry.

```bash
git clone -b eval51 <repo-url> && cd cantabile

# System libraries the image would have provided. Needs root in the pod;
# most pod shells have it.
apt-get update && apt-get install -y --no-install-recommends \
  libegl1 libgles2 libglib2.0-0 libosmesa6 libsndfile1 fluidsynth \
  build-essential portaudio19-dev

python -m venv .venv && . .venv/bin/activate
pip install "jax[cuda12]==0.6.2"
pip install -r rl/requirements.txt \
  mujoco dm_control note_seq pretty_midi soundfile huggingface_hub \
  torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e ./env

export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY=... HF_TOKEN=... WANDB_ENTITY=cantabile WANDB_PROJECT=cantabile

cd rl
for i in 1 2 3 4; do python worker.py --gpu 0 & done
```

The three asset directories are gitignored, so a fresh clone does not have them:
`env/robopianist/models/hands/third_party`, `env/robopianist/soundfonts`, and
`env/robopianist/music/data/pig_single_finger` — 12 MB in total. Copy them across from
a machine that has them, or run `env/scripts/get_soundfonts.sh` for the second. Without
them the environment fails to load, which preflight catches before anything is claimed.

`--gpu` indexes what the pod can see. If the pod was allocated one GPU it is `0`
regardless of which physical card that is; `nvidia-smi` inside the pod shows what you
actually have.

---

## How many slots per GPU

Memory is not the limit. Measured on an RTX 5090: **794 MiB per training run**, so
even eight of them use under 7 GB of 32. Compute is what saturates — four concurrent
runs already put the GPU at roughly 50% utilisation, and past that each run slows down.

| slots / GPU | VRAM | note |
|---|---|---|
| 4 | ~3.2 GB | the density these numbers were measured at |
| 6 | ~4.8 GB | reasonable, worth trying |
| 8+ | ~6.4 GB | contention starts costing more than it adds |

Two GPUs at six slots is twelve workers, which takes the 51-song baseline arm from
about four days down to about two and a half.

This works at all because the image sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Without it the first JAX process claims roughly three quarters of the card's memory
and every later container on that GPU dies out of memory.

---

## Watching it

```bash
docker ps --filter name=cantabile
docker logs -f cantabile-g4-s1
docker rm -f $(docker ps -q --filter name=cantabile)     # stop everything
```

Progress is visible in the queue itself, and the repo is public so this needs no
credentials from anywhere:

**https://huggingface.co/datasets/well-balanced/cantabile-runs**

```
main/<song>/<method>/.gitkeep                 queued, nothing has claimed it
main/<song>/<method>/<seed>/CLAIM-<worker>    a worker holds it
main/<song>/<method>/<seed>/*.pt              done
main/<song>/<method>/<seed>/FAILED            crashed, needs a human
```

State is nothing but file presence, so there is no status field that can disagree with
the artifacts. A worker uploads its checkpoints *before* releasing its claim, so a cell
that looks finished always is.

A worker that finds nothing claimable sleeps and re-polls rather than exiting. It is
fine to leave workers running against an empty queue — they pick up new cells as soon
as the queue grows.

---

## When something goes wrong

**A cell is marked `FAILED`.** Failures are deliberately not retried: a run that
crashed for a real reason crashes identically on the next machine, and auto-retry turns
one bug into a fleet-wide spin. Read the reason inside the `FAILED` file, fix it, then
delete the file to return the cell to the queue.

**A worker died and its cell is stuck.** A claim whose file has not been touched for 60
minutes is treated as abandoned and another worker takes it over. Nothing to do.

**Everything fails immediately.** That is preflight doing its job — it checks the wandb
key, the HF token, GPU visibility and a real EGL render before claiming anything. The
error names which one failed.

**Renders are black and runs complete anyway.** EGL is not working. The container needs
`MUJOCO_GL=egl` (set in the image) and the host needs the NVIDIA Container Toolkit;
check `MUJOCO_EGL_DEVICE_ID` is 0 and not the host's GPU number.

---

## Rebuilding the image

Only needed when the training code changes.

```bash
cd cantabile
docker build -t wellbalanced/cantabile:v2 .
docker push wellbalanced/cantabile:v2
```

Layers are ordered slowest-changing first — CUDA, then the pip stack, then the assets
git does not carry, then the source — so a code change rebuilds and ships a ~2 MB
layer rather than the whole ~10 GB.

**Always use an explicit tag, never `latest`.** Workers record the image they ran in
their claim, and two machines silently on different code is a debugging problem that
appears weeks later in the results rather than at run time.
