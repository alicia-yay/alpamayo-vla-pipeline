"""End-to-end smoke test: load model, fetch real sample, one forward + backward.

Dispatches to a Ray GPU worker via @ray.remote(num_gpus=1). The test:
  1. Loads Alpamayo-1.5-10B in bfloat16 on the worker
  2. Fetches one PhysicalAI-AV sample directly via NVIDIA's loader
     (bypassing Ray Data round-trip for simplicity in this smoke test)
  3. Runs a forward pass through action_in_proj, the action expert, and
     action_out_proj
  4. Computes flow matching loss
  5. Runs backward, asserts loss is finite and gradients flow through
     trainable params

GPU sizing:
  L4 (24 GB): forward pass alone peaks at ~23.3 GB, OOMs on backward.
  A100-80GB: clears with headroom.
  H100-80GB: clears with substantial headroom.

Run from the project root:
  python tests/test_pipeline.py
"""

import os
import sys

os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
sys.path.insert(0, "src")

import ray


HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN env var not set")

ray.init(
    ignore_reinit_error=True,
    runtime_env={
        "working_dir": "src",
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    },
)


@ray.remote(num_gpus=1)
def run_training_step():
    """Runs on a GPU worker. All imports are inside so the driver doesn't
    need to import torch or alpamayo packages."""
    import sys
    import torch

    sys.path.insert(0, ".")

    from alpamayo_loader import load_alpamayo_policy, print_param_summary
    from train_step import alpamayo_train_step

    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    print("=== End-to-end smoke test on GPU worker ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    print("Loading Alpamayo-1.5-10B in bfloat16...")
    policy = load_alpamayo_policy(dtype=torch.bfloat16, attn_implementation="eager")
    policy = policy.to("cuda")
    policy.train()
    print_param_summary(policy)
    print(f"\nVRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nFetching one PhysicalAI-AV sample...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_id = avdi.clip_index.index.tolist()[0]
    sample = load_physical_aiavdataset(clip_id=clip_id, t0_us=5_100_000, avdi=avdi)
    print(f"clip_id: {sample['clip_id']}")

    batch = {
        "ego_history_xyz": sample["ego_history_xyz"],
        "ego_history_rot": sample["ego_history_rot"],
        "ego_future_xyz": sample["ego_future_xyz"],
        "ego_future_rot": sample["ego_future_rot"],
    }

    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}  {v.dtype}")

    print("\nRunning forward pass...")
    try:
        loss = alpamayo_train_step(policy, batch)
        print(f"loss: {loss.item():.6f}")
        print(f"VRAM after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        if not torch.isfinite(loss):
            return {"ok": False, "error": "Loss is NaN or Inf"}

        print("\nRunning backward pass...")
        loss.backward()
        print(f"VRAM after backward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.parameters() if p.requires_grad
        )
        if not has_grad:
            return {"ok": False, "error": "No gradients flowed through trainable params"}

        print("backward + gradients: OK")
        print("\n=== SMOKE TEST PASSED ===")
        return {"ok": True, "loss": float(loss.item())}

    except torch.cuda.OutOfMemoryError as e:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return {
            "ok": False,
            "error": "OOM",
            "message": str(e),
            "peak_vram_gb": peak_gb,
        }


if __name__ == "__main__":
    print("Dispatching training step to a GPU worker...")
    result = ray.get(run_training_step.remote())
    print(f"\nDriver received: {result}")

    if not result.get("ok"):
        sys.exit(1)
