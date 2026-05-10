# Alpamayo VLA Pipeline

Reference for fine-tuning NVIDIA's [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) on the [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset, using Ray Data and Ray Train on Anyscale.

<p align="center">
  <img src="media/architecture.png" alt="Architecture diagram" width="900"/>
</p>

## Validation

A single training step runs end to end on A100-80GB.

```
GPU: NVIDIA A100-SXM4-80GB
VRAM total: 85.1 GB

Model: Alpamayo-1.5-10B
  Total parameters:     11.08B
  Trainable:            2.28B  (20.6%)
  Frozen backbone:      Cosmos-Reason2-8B (8.798B)
  Trainable expert:     2.279B

Sample: PhysicalAI-AV NCore subset
  clip_id:              25cd4769-5dcf-4b53-a351-bf2c5deb6124

Memory profile (bfloat16, eager attention, batch size 1):
  After model load:     22.17 GB
  After forward:        22.59 GB
  After backward:       26.75 GB

Loss: 1.976562 (flow matching MSE)
Gradients: flowed through all trainable parameters
```

## Components

| File | Description |
|------|-------------|
| `src/ncore_datasource.py` | Ray Data datasource that streams PhysicalAI-AV samples via NVIDIA's `physical_ai_av` package. One row per `(clip_id, t0_us)`. |
| `src/cluster_setup.py` | Distributes Alpamayo dependencies to the system Python on Ray workers. |
| `src/alpamayo_loader.py` | Loads Alpamayo-1.5-10B, freezes the Cosmos-Reason2-8B backbone (8.8B), leaves the action expert plus projections trainable (2.28B). |
| `src/action_loss.py` | Adds flow matching training methods to Alpamayo 1.5's `FlowMatching` at runtime. Ported from Alpamayo 1.0 (Apache-2.0). |
| `src/train_step.py` | One training step: trajectory to action space, diffusion training data, expert forward, flow matching loss. |
| `tests/test_pipeline.py` | End-to-end smoke test. Loads the model on a GPU worker, runs forward and backward on one real sample. |

## Architecture

Fine-tuning Alpamayo splits into two workloads with different resource profiles. Streaming and decoding multi-camera driving clips is CPU-bound and I/O-heavy. Forward and backward passes on a 10B-parameter model are GPU-bound. Ray Data handles the first on auto-scaled CPU workers, Ray Train handles the second on GPU workers, and the two stages communicate via backpressured queues. The same pipeline scales from 1 GPU to N GPUs by changing `ScalingConfig.num_workers`.

The training step in this repository runs the action expert with an empty KV cache rather than conditioning on visual context. The math is correct end to end and the plumbing is validated; production fine-tuning should mirror the full forward in NVIDIA's Alpamayo 1.0 SFT code at [finetune/sft/models/sft_alpamayo_r1.py](https://github.com/NVlabs/alpamayo/blob/main/finetune/sft/models/sft_alpamayo_r1.py), which fuses trajectory tokens into the VLM and feeds the resulting hidden states into the expert as past key values.

The flow matching loss is taken directly from Alpamayo 1.0. NVIDIA shipped supervised fine-tuning and RL training code in the 1.0 repository but the 1.5 repository is inference-only, so the two missing methods (`construct_training_data` and `compute_loss_from_pred`) are patched onto Alpamayo 1.5's `FlowMatching` class at runtime. No fork of the package is required.

## Setup

```
uv venv
source .venv/bin/activate
uv sync
```

`flash-attn` is optional. Alpamayo 1.5 falls back to `attn_implementation="eager"`.

Three gated NVIDIA assets need separate access requests on HuggingFace: [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B), [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B), and [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). Once approved, authenticate:

```
export HF_TOKEN=hf_...
hf auth login --token $HF_TOKEN
```

Ray workers run from system Python, not from the venv. After starting the cluster, install the dependencies on the GPU workers:

```
python src/cluster_setup.py
```

## GPU requirements

| Configuration | VRAM | Notes |
|---------------|------|-------|
| Inference (single sample) | ~24 GB | Per Alpamayo 1.5 model card |
| Inference (multi-sample with CFG) | ~60 GB | Per Alpamayo 1.5 model card |
| Training (forward only) | ~23 GB peak | Action expert forward |
| Training (forward + backward) | ~27 GB peak | Measured on A100-80GB |

L4 (24 GB) is insufficient for training. The forward pass alone peaks at 23.3 GB on L4 before the backward graph allocates. A100-80GB and H100-80GB both have substantial headroom.

## Quickstart

Stream samples through Ray Data:

```python
import os, sys
os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
sys.path.insert(0, "src")

import ray
from ncore_datasource import NCoreDatasource

ray.init(runtime_env={
    "working_dir": "src",
    "env_vars": {"HF_TOKEN": os.environ["HF_TOKEN"]},
})

ds = ray.data.read_datasource(NCoreDatasource(samples_per_clip=1, max_clips=5))
print(f"Total rows: {ds.count()}")
```

Run the smoke test on a GPU worker:

```
python tests/test_pipeline.py
```

## References

* [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) (inference)
* [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) (SFT and RL training code)
