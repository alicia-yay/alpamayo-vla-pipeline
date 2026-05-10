"""Flow matching training methods for Alpamayo 1.5.

Alpamayo 1.5 ships only the inference-time `FlowMatching.sample()` method
(decorated with @torch.no_grad). The training-time methods needed to compute
loss against ground-truth trajectories are not in the 1.5 inference repo.

NVIDIA shipped them in the Alpamayo 1.0 repo (NVlabs/alpamayo) under
finetune/sft/. Specifically:
  - FlowMatching.construct_training_data: samples a timestep t, interpolates
    between data x and noise to produce noisy_x; returns the dict the action
    expert consumes.
  - FlowMatching.compute_loss_from_pred: MSE between predicted velocity and
    the optimal-transport target (x - noise).

Architecturally Alpamayo 1.5 inherits the same FlowMatching base, so we add
these methods to the class at runtime. No fork of the alpamayo1_5 package
needed.

Reference:
  https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/diffusion/flow_matching.py
  Apache-2.0 licensed (NVIDIA).
"""

import torch


def _construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Sample a timestep t, interpolate between data x and noise.

    Returns a dict the action expert consumes: noisy_x, timesteps, original
    x and noise (kept for the loss). Timesteps are sampled from a Beta(1.5,
    1.0) distribution by default; this concentrates training near t=0
    (close to noise) and t=1 (close to data), which is standard practice.

    Lazily initializes the beta distribution on first call so we can patch
    onto an already-instantiated FlowMatching without touching __init__.
    """
    if not hasattr(self, "beta_dist"):
        self.beta_dist = torch.distributions.beta.Beta(
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        self.beta_scale_constant = 0.999
        self.train_timestep_sampler = "beta"

    batch_size = x.shape[0]

    if self.train_timestep_sampler == "uniform":
        t = torch.rand((batch_size,), device=x.device)
    elif self.train_timestep_sampler == "beta":
        t = self.beta_dist.sample((batch_size,)).to(x.device)
        t = self.beta_scale_constant - t * self.beta_scale_constant
    else:
        raise ValueError(f"Invalid timestep sampler: {self.train_timestep_sampler}")

    while len(t.shape) < len(x.shape):
        t = t.unsqueeze(-1)

    # Beta distribution samples are float32; cast to x.dtype so we don't
    # upcast x in the multiplication below (mixed precision matters here).
    t = t.to(dtype=x.dtype)

    noise = torch.randn_like(x)
    noisy_x = t * x + (1 - t) * noise

    return {
        "x": x,
        "noisy_x": noisy_x,
        "timesteps": t,
        "noise": noise,
        "is_drop_guidance": None,
    }


def _compute_loss_from_pred(
    self, training_data: dict[str, torch.Tensor], pred: torch.Tensor
) -> torch.Tensor:
    """Flow matching loss: MSE between predicted velocity and (x - noise).

    The optimal-transport flow matching objective predicts the velocity field
    that pushes noise to data. The target velocity at any t is constant and
    equals (x - noise).
    """
    x = training_data["x"]
    noise = training_data["noise"]
    target = (x - noise).to(dtype=pred.dtype)
    return torch.nn.functional.mse_loss(target, pred)


def patch_flow_matching():
    """Add training methods to alpamayo1_5.FlowMatching at runtime.

    Idempotent: marks the class with _TRAINING_PATCH_APPLIED so repeat calls
    are safe.
    """
    from alpamayo1_5.diffusion.flow_matching import FlowMatching

    if hasattr(FlowMatching, "_TRAINING_PATCH_APPLIED"):
        return

    FlowMatching.construct_training_data = _construct_training_data
    FlowMatching.compute_loss_from_pred = _compute_loss_from_pred
    FlowMatching._TRAINING_PATCH_APPLIED = True
