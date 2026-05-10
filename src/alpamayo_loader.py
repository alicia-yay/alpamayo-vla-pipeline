"""Alpamayo-1.5-10B model loader with backbone freezing.

Loads NVIDIA's Alpamayo-1.5-10B from HuggingFace, freezes the Cosmos-Reason2-8B
backbone, and leaves the action expert plus its input/output projections
trainable. This is the standard fine-tuning regime for VLA models: freeze the
expensive vision-language pretraining, train only the action head on
domain-specific data.

Submodule structure (verified from src/alpamayo1_5/models/alpamayo1_5.py):
  self.vlm              Cosmos-Reason2-8B backbone (FROZEN)
  self.expert           Action expert transformer (TRAINABLE, ~2.28B)
  self.action_in_proj   Projection from action space to expert hidden (TRAINABLE)
  self.action_out_proj  Projection from expert hidden to action space (TRAINABLE)
  self.diffusion        Flow matching wrapper, no learnable params at top level
  self.action_space     Config object, not a Module
  self.tokenizer        Not a Module

Verified parameter counts:
  Total:     11.08B
  Trainable: 2.28B  (20.6 percent)

Note on attention implementation:
  Alpamayo 1.5 does NOT support flash_attention_2 or sdpa. Use "eager".
  Slower but works in any environment without flash-attn build issues.

Source: alicia-yay, 2026
"""

import torch


# Submodules to keep trainable. Everything else gets frozen, including the
# 8.8B Cosmos-Reason2-8B backbone (self.vlm).
TRAINABLE = {"expert", "diffusion", "action_in_proj", "action_out_proj"}


def load_alpamayo_policy(
    pretrained_path: str = "nvidia/Alpamayo-1.5-10B",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
):
    """Load Alpamayo, freeze the VLM backbone, leave action heads trainable.

    Args:
        pretrained_path: HuggingFace repo or local path. Requires gated access
            to both nvidia/Alpamayo-1.5-10B AND nvidia/Cosmos-Reason2-8B.
        dtype: torch dtype for model weights. bfloat16 strikes the right
            tradeoff for fine-tuning on A100/H100.
        attn_implementation: must be "eager" for Alpamayo 1.5 today.

    Returns:
        torch.nn.Module ready for training. Caller still needs to .to(device)
        and .train() it.
    """
    # Apply the action-loss patch as a side effect of loading. See
    # action_loss.py for why: Alpamayo 1.5 ships only the inference-time
    # FlowMatching.sample(); we add the training methods at runtime.
    from action_loss import patch_flow_matching
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    patch_flow_matching()

    policy = Alpamayo1_5.from_pretrained(
        pretrained_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    # Freeze everything first
    for p in policy.parameters():
        p.requires_grad = False

    # Then unfreeze named action submodules
    for name, module in policy.named_children():
        if name in TRAINABLE:
            for p in module.parameters():
                p.requires_grad = True

    return policy


def print_param_summary(policy: torch.nn.Module) -> None:
    """Print trainable vs frozen params, plus per-submodule breakdown.

    Useful for sanity-checking the freeze worked. Expected output:
      vlm                        8.798B  frozen
      expert                     2.279B  TRAINABLE
      action_in_proj             0.001B  TRAINABLE
      action_out_proj            0.000B  TRAINABLE
    """
    print("=== Per-submodule param counts ===")
    for name, module in policy.named_children():
        n_total = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n_total > 0:
            status = "TRAINABLE" if n_train > 0 else "frozen"
            print(f"  {name:25s}  {n_total/1e9:.3f}B  {status}")

    print()
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print("=== Totals ===")
    print(f"Total:     {total/1e9:.2f}B")
    print(f"Trainable: {trainable/1e9:.2f}B  ({100*trainable/total:.1f}%)")


if __name__ == "__main__":
    print("Loading Alpamayo-1.5-10B (downloads ~22GB on first run)...")
    policy = load_alpamayo_policy()
    print()
    print_param_summary(policy)
