"""Export a trained actor checkpoint (TanhNormal(MLP, ..) family) to a real,
directly `load_state_dict()`-able PyTorch module -- not just a flat weight dump.

Unlike `export_hf.py`'s `model.pt` (flax param names/shapes, flattened -- needs the
flax module code to make sense of), this produces a plain `torch.nn.Sequential`-based
module whose state_dict keys match the RP1M authors' own released `actor.pt` files
(`mlp.0/2/4.weight`, `mean_layer.weight`, `log_std_layer.weight`) -- loadable in pure
PyTorch, no JAX/Flax dependency, by anyone who defines the same small `TorchActor`
class below (or an equivalent `nn.Sequential`).

Saves to a fixed layout so checkpoints across methods/songs/seeds stay organized:
    <out-root>/<method>/<baseline|cantabile>/<song>/actor_<seed>.pt

Usage:
    python export_torch.py --checkpoint ../tmp/checkpoints/base-nocturne-s43/checkpoint_8000000_actor.flax \\
        --obs-dim 1474 --action-dim 39 --method base --variant baseline \\
        --song nocturne --seed 43
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# This export does one tiny forward pass for a correctness cross-check (see
# _cross_check below) -- not worth a GPU, and JAX's GPU matmuls default to
# reduced-precision (TF32-like) accumulation, which was found to produce a real
# ~1e-2 spurious diff against PyTorch's exact-float32 CPU matmul (vs ~1e-5 on
# JAX-CPU) -- forcing CPU here keeps the cross-check meaningful.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.nn as tnn
import tyro
from flax import serialization
from flax.training.train_state import TrainState

from distributions import TanhNormal
from networks import MLP

_ACTIVATIONS = {
    # flax's nn.gelu defaults to approximate=True (tanh approximation) -- torch's
    # nn.GELU() defaults to the exact erf-based formula, a real numerical mismatch
    # if left as default (confirmed via the flax/torch cross-check below; this is
    # the same activation-mismatch bug class found earlier for the RP1M specialist,
    # see cantabile-rp1m-adapter memory).
    "gelu": lambda: tnn.GELU(approximate="tanh"),
    "relu": tnn.ReLU,
    "tanh": tnn.Tanh,
}


@dataclass(frozen=True)
class ExportArgs:
    checkpoint: str
    """Path to an actor sidecar (*_actor.flax) or full agent checkpoint."""
    out_root: str = "../torch_models_pig"
    method: str = "base"
    variant: str = "baseline"  # "baseline" | "cantabile"
    song: str = ""
    seed: int = 0
    obs_dim: int = 1474
    action_dim: int = 39
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    activation: str = "gelu"
    full_agent_checkpoint: bool = False
    """Set if `checkpoint` is a full agent .flax (SAC/ResidualSAC) rather
    than an actor-only sidecar -- only the `actor` sub-tree is used."""


class TorchActor(tnn.Module):
    """mlp.{0,2,4,...}.{weight,bias} + mean_layer/log_std_layer -- same key
    naming as the RP1M authors' released actor.pt (see rp1m_specialist.py's
    `load_specialist_params` for the flax-side counterpart of this mapping)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...],
                 activation: str = "gelu"):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(tnn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        self.mlp = tnn.Sequential(*layers)
        self.mean_layer = tnn.Linear(in_dim, action_dim)
        self.log_std_layer = tnn.Linear(in_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(obs)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    def mode(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: tanh(mean) -- matches the flax side's `dist.mode()`."""
        mean, _ = self(obs)
        return torch.tanh(mean)


def _load_flax_actor(args: ExportArgs) -> TrainState:
    actor_base_cls = lambda: MLP(
        hidden_dims=args.hidden_dims,
        activation=getattr(nn, args.activation),
        activate_final=True,
    )
    actor_def = TanhNormal(actor_base_cls, args.action_dim)
    dummy_obs = jnp.zeros((1, args.obs_dim))
    params = actor_def.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    template = TrainState.create(apply_fn=actor_def.apply, params=params, tx=optax.adam(3e-4))

    with open(args.checkpoint, "rb") as f:
        raw = f.read()
    if args.full_agent_checkpoint:
        loaded = serialization.from_bytes({"actor": template}, raw)
        return loaded["actor"]
    return serialization.from_bytes(template, raw)


def _flax_to_torch_state_dict(flax_params, n_hidden: int) -> dict:
    """flax MLP_0.Dense_i (kernel=(in,out)) -> torch mlp.{2i}.weight (out,in);
    OutputDenseMean/OutputDenseLogStd -> mean_layer/log_std_layer."""
    mlp = flax_params["MLP_0"]
    state_dict = {}
    for i in range(n_hidden):
        dense = mlp[f"Dense_{i}"]
        state_dict[f"mlp.{2 * i}.weight"] = torch.from_numpy(np.asarray(dense["kernel"]).T.copy())
        state_dict[f"mlp.{2 * i}.bias"] = torch.from_numpy(np.asarray(dense["bias"]).copy())
    for flax_name, torch_name in [("OutputDenseMean", "mean_layer"), ("OutputDenseLogStd", "log_std_layer")]:
        layer = flax_params[flax_name]
        state_dict[f"{torch_name}.weight"] = torch.from_numpy(np.asarray(layer["kernel"]).T.copy())
        state_dict[f"{torch_name}.bias"] = torch.from_numpy(np.asarray(layer["bias"]).copy())
    return state_dict


def _cross_check(flax_actor: TrainState, torch_actor: TorchActor, obs_dim: int) -> float:
    """Feeds the same (small-magnitude) random obs through both nets, returns max
    abs diff between the pre-tanh `mean` on each side.

    Deliberately compares the *pre-tanh* mean, not `dist.mode()`/`torch_actor.mode()`
    (post-tanh) -- unit-scale random input is wildly out of the real observation
    distribution, so 3 GELU layers blow it up enough that mean values routinely
    exceed +-10, at which point tanh saturates to +-1.0 on both sides regardless of
    real (tiny, benign) JAX-vs-PyTorch matmul rounding differences -- comparing
    post-tanh outputs was found to spuriously "fail" this way (max diff ~0.07) even
    though the conversion was already correct. Comparing the unsquashed mean checks
    the actual thing this function cares about (did the weights convert correctly)
    without saturation amplifying float noise."""
    rng = np.random.RandomState(0)
    x_np = (rng.randn(8, obs_dim) * 0.05).astype(np.float32)
    flax_mean = np.asarray(
        flax_actor.apply_fn({"params": flax_actor.params}, jnp.asarray(x_np)).distribution.mode()
    )
    torch_actor.eval()
    with torch.no_grad():
        torch_mean, _ = torch_actor(torch.from_numpy(x_np))
        torch_mean = torch_mean.numpy()
    return float(np.abs(flax_mean - torch_mean).max())


def main(args: ExportArgs) -> None:
    if not args.song:
        raise ValueError("--song is required")

    flax_actor = _load_flax_actor(args)
    state_dict = _flax_to_torch_state_dict(flax_actor.params, len(args.hidden_dims))

    torch_actor = TorchActor(args.obs_dim, args.action_dim, args.hidden_dims, args.activation)
    torch_actor.load_state_dict(state_dict, strict=True)

    max_diff = _cross_check(flax_actor, torch_actor, args.obs_dim)
    if max_diff > 1e-3:
        raise RuntimeError(
            f"flax vs torch forward-pass mismatch too large (max diff {max_diff}) -- "
            "conversion is likely wrong, refusing to save."
        )

    out_path = Path(args.out_root) / args.method / args.variant / args.song / f"actor_{args.seed}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_actor.state_dict(), out_path)
    print(f"Exported -> {out_path} (flax/torch cross-check max diff: {max_diff:.2e})")


if __name__ == "__main__":
    main(tyro.cli(ExportArgs, description=__doc__))
