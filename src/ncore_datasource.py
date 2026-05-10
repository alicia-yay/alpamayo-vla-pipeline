"""Ray Data datasource for NVIDIA's PhysicalAI-Autonomous-Vehicles dataset.

Wraps NVIDIA's `physical_ai_av` Python package (PyPI) so PhysicalAI-AV samples
stream into Ray Data without intermediate Parquet conversion. One Ray Data
row corresponds to one (clip_id, t0_us) sample, with all the multi-camera +
ego trajectory tensors that Alpamayo expects.

Why a custom datasource and not ray.data.read_*():
  - PhysicalAI-AV ships as zarr-itar archives with multi-rate sensors
    (cameras, lidar, ego pose) that need timestamp alignment per sample.
  - NVIDIA's `physical_ai_av` package handles all of that internally; we
    just need to surface it to Ray Data as one row per sample.
  - Avoids a 1-2 TB Parquet preprocessing step.

Source: alicia-yay, 2026
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import pyarrow as pa
from ray.data import Datasource, ReadTask
from ray.data.block import BlockMetadata


# Multi-dimensional numpy arrays don't survive pa.Table.from_pylist directly,
# so we pickle them into bytes columns and unpickle in the consumer.
_BYTES_COLS = (
    "image_frames",
    "ego_history_xyz",
    "ego_history_rot",
    "ego_future_xyz",
    "ego_future_rot",
)


def _rows_to_arrow_table(rows: list[dict]) -> pa.Table:
    """Convert a list of sample dicts to an Arrow table, pickling array fields."""
    transformed = []
    for row in rows:
        out = {}
        for k, v in row.items():
            if k in _BYTES_COLS and isinstance(v, np.ndarray):
                buf = io.BytesIO()
                pickle.dump(v, buf, protocol=pickle.HIGHEST_PROTOCOL)
                out[k] = buf.getvalue()
            else:
                out[k] = v
        transformed.append(out)
    return pa.Table.from_pylist(transformed)


@dataclass
class NCoreDatasourceMetadata:
    """Counts and shapes for the dataset, populated lazily on first use."""

    num_clips: int
    samples_per_clip: int

    @property
    def total_rows(self) -> int:
        return self.num_clips * self.samples_per_clip


class NCoreReadTask(ReadTask):
    """One Ray ReadTask per clip. Yields one Arrow block per clip."""

    def __init__(self, clip_id: str, samples_per_clip: int, per_task_row_limit: Optional[int] = None):
        self.clip_id = clip_id
        self.samples_per_clip = samples_per_clip
        self.per_task_row_limit = per_task_row_limit

        meta = BlockMetadata(
            num_rows=min(samples_per_clip, per_task_row_limit or samples_per_clip),
            size_bytes=None,
            input_files=None,
            exec_stats=None,
        )
        super().__init__(self._read, meta)

    def _read(self) -> Iterator[pa.Table]:
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        import physical_ai_av

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

        # PhysicalAI-AV exposes per-clip valid t0_us ranges; we walk a fixed
        # number of samples per clip across that range.
        clip_meta = avdi.clip_index.loc[self.clip_id]
        t0_min, t0_max = int(clip_meta["t0_us_min"]), int(clip_meta["t0_us_max"])
        if self.samples_per_clip == 1:
            t0_list = [t0_min + (t0_max - t0_min) // 2]
        else:
            t0_list = list(np.linspace(t0_min, t0_max, self.samples_per_clip, dtype=int))

        if self.per_task_row_limit is not None:
            t0_list = t0_list[: self.per_task_row_limit]

        rows = []
        for t0_us in t0_list:
            sample = load_physical_aiavdataset(
                clip_id=self.clip_id, t0_us=int(t0_us), avdi=avdi,
            )
            rows.append({
                "clip_id": self.clip_id,
                "t0_us": int(t0_us),
                "image_frames": np.asarray(sample["image_frames"]),
                "ego_history_xyz": np.asarray(sample["ego_history_xyz"]),
                "ego_history_rot": np.asarray(sample["ego_history_rot"]),
                "ego_future_xyz": np.asarray(sample["ego_future_xyz"]),
                "ego_future_rot": np.asarray(sample["ego_future_rot"]),
            })

        yield _rows_to_arrow_table(rows)


class NCoreDatasource(Datasource):
    """Ray Datasource over NVIDIA's PhysicalAI-AV NCore subset.

    Args:
        samples_per_clip: number of (clip_id, t0_us) samples drawn per clip.
        max_clips: cap on the number of clips. None means use all clips.
    """

    def __init__(self, samples_per_clip: int = 1, max_clips: Optional[int] = None):
        self.samples_per_clip = samples_per_clip
        self.max_clips = max_clips
        self._metadata: Optional[NCoreDatasourceMetadata] = None
        self._clip_ids: Optional[List[str]] = None

    def _ensure_metadata(self):
        if self._metadata is not None:
            return
        import physical_ai_av

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        all_clip_ids = avdi.clip_index.index.tolist()
        if self.max_clips is not None:
            all_clip_ids = all_clip_ids[: self.max_clips]
        self._clip_ids = all_clip_ids
        self._metadata = NCoreDatasourceMetadata(
            num_clips=len(all_clip_ids),
            samples_per_clip=self.samples_per_clip,
        )

    def estimate_inmemory_data_size(self) -> Optional[int]:
        return None

    def get_read_tasks(self, parallelism: int, **kwargs) -> List[ReadTask]:
        self._ensure_metadata()
        per_task_row_limit = kwargs.get("per_task_row_limit")
        return [
            NCoreReadTask(
                clip_id=cid,
                samples_per_clip=self.samples_per_clip,
                per_task_row_limit=per_task_row_limit,
            )
            for cid in self._clip_ids
        ]


if __name__ == "__main__":
    # Smoke test: stream 5 samples through Ray Data
    import os
    import sys

    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    sys.path.insert(0, ".")

    import ray

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN env var not set")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": "src", "env_vars": {"HF_TOKEN": HF_TOKEN}},
    )

    ds = ray.data.read_datasource(NCoreDatasource(samples_per_clip=1, max_clips=5))
    rows = ds.take(5)
    print(f"Total rows: {len(rows)}")
    for row in rows:
        # Unpickle bytes cols for inspection
        ifr = pickle.loads(row["image_frames"])
        ego_future = pickle.loads(row["ego_future_xyz"])
        print(f"  clip_id: {row['clip_id']}")
        print(f"    image_frames shape: {ifr.shape}")
        print(f"    ego_future_xyz shape: {ego_future.shape}")
