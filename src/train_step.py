"""Single training step for Alpamayo 1.5.

One forward + flow matching loss computation per call. Designed to be the
function a Ray Train loop calls per micro-batch. Mirrors the training-side
forward in NVIDIA's Alpamayo 1.0 SFT code (TrainableAlpamayoR1.forward()),
adapted for Alpamayo 1.5's class structure.

Simplification vs NVIDIA's reference:
  This step skips the full VLM rollout and trajectory-token-fusion that
  conditions the action expert on visual context. Instead it runs the expert
  on an empty KV cache, which lets the math run end-to-end and validates the
  pipeline plumbing, but does NOT condition on cameras. Use this for smoke
  testing the Ray Data + Ray Train wiring; for production fine-tuning, port
  the fuse_traj_tokens path from NVlabs/alpamayo/finetune/sft/models/
  sft_alpamayo_r1.py.

Dtype handling:
  - Trajectory inputs stay in float32 because traj_to_action calls a
    torch.linalg.cholesky which is float32-only on CUDA.
  - The resulting `action` is cast to the model's dtype (typically bfloat16)
    before feeding into action_in_proj.
  - The diffusion timestep `t` (sampled from Beta) is also float32 by default
    and is cast to x.dtype inside action_loss._construct_training_data.

Reference:
  https://github.com/NVlabs/alpamayo/blob/main/finetune/sft/models/sft_alpamayo_r1.py
  Apache-2.0 licensed (NVIDIA).
"""

from typing import Any

import einops
import torch


def alpamayo_train_step(
    policy: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    """One training step. Returns the (scalar) flow matching loss tensor.

    Args:
        policy: an Alpamayo1_5 instance with the action_loss patch applied
            (handled automatically by load_alpamayo_policy()).
        batch: dict with keys
            ego_history_xyz: (B, 1, 16, 3)   float32 or bfloat16
            ego_history_rot: (B, 1, 16, 3, 3)
            ego_future_xyz:  (B, 1, 64, 3)
            ego_future_rot:  (B, 1, 64, 3, 3)

    Returns:
        Scalar loss tensor with grad enabled. Caller should run
        loss.backward() and step the optimizer.
    """
    device = next(policy.parameters()).device
    model_dtype = next(policy.parameters()).dtype

    # traj_to_action does a Cholesky decomp which is float32-only on CUDA,
    # so keep these in float32. We cast `action` to model dtype after.
    ego_history_xyz = batch["ego_history_xyz"].to(device, dtype=torch.float32).squeeze(1)
    ego_history_rot = batch["ego_history_rot"].to(device, dtype=torch.float32).squeeze(1)
    ego_future_xyz = batch["ego_future_xyz"].to(device, dtype=torch.float32).squeeze(1)
    ego_future_rot = batch["ego_future_rot"].to(device, dtype=torch.float32).squeeze(1)

    # 1. Convert trajectory to action space (unicycle: acceleration + curvature)
    action = policy.action_space.traj_to_action(
        traj_history_xyz=ego_history_xyz,
        traj_history_rot=ego_history_rot,
        traj_future_xyz=ego_future_xyz,
        traj_future_rot=ego_future_rot,
    )
    action = action.reshape(-1, *policy.action_space.get_action_space_dims())
    action = action.to(dtype=model_dtype)

    # 2. Diffusion: sample timestep, interpolate between action and noise
    training_data = policy.diffusion.construct_training_data(action)

    # 3. Project (noisy_x, t) into the expert's hidden space
    action_embeds = policy.action_in_proj(
        training_data["noisy_x"], training_data["timesteps"]
    )

    # 4. Expert forward pass.
    #    Simplification: empty KV cache, no VLM context. Production training
    #    feeds the VLM hidden states as past_key_values; see module docstring.
    batch_size = action_embeds.shape[0]
    num_expert_tokens = action_embeds.shape[1]

    position_ids = torch.arange(num_expert_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()

    forward_kwargs = {}
    if getattr(policy.config, "expert_non_causal_attention", False):
        forward_kwargs["is_causal"] = False

    expert_outputs = policy.expert(
        inputs_embeds=action_embeds,
        position_ids=position_ids,
        attention_mask=None,
        use_cache=False,
        **forward_kwargs,
    )
    diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1]:]

    # 5. Project expert output back to action space
    pred = policy.action_out_proj(diffusion_out)
    pred = pred.view(-1, *policy.action_space.get_action_space_dims())

    # 6. Flow matching loss
    loss = policy.diffusion.compute_loss_from_pred(
        training_data=training_data,
        pred=pred,
    )
    return loss
