import os
import jax
import numpy as np
import grain
import cv2
from typing import Any, Optional


class CALVINDataSource(grain.sources.RandomAccessDataSource):
    """
    A Grain data source that reads demonstration episodes from CALVIN NPZ datasets.

    Each record is a tuple of (data_dir, start_frame_id, end_frame_id) identifying
    a single episode defined in ep_start_end_ids.npy.
    """

    def __init__(
        self,
        data_dirs: list[str],
        image_key: str = "rgb_static",
    ):
        """
        Args:
            data_dirs: List of CALVIN data directories (each must contain
                       ep_start_end_ids.npy and episode_*.npz files).
            image_key: Observation key for image data (e.g. "rgb_static",
                       "rgb_gripper"). Stored for informational purposes only.
        """
        self._records: list[tuple[str, int, int]] = []  # (data_dir, start_frame_id, end_frame_id)
        for data_dir in data_dirs:
            ep_path = os.path.join(data_dir, "ep_start_end_ids.npy")
            ep_ids = np.load(ep_path)  # shape: (num_episodes, 2), dtype int64
            for row in ep_ids:
                start_frame_id = int(row[0])
                end_frame_id = int(row[1])  # inclusive
                self._records.append((data_dir, start_frame_id, end_frame_id))

        self._image_key = image_key
        print(
            f"CALVINDataSource: {len(self._records)} episodes from "
            f"{len(data_dirs)} dir(s), image_key={image_key}"
        )

    def __repr__(self) -> str:
        return (
            f"CALVINDataSource(num_records={len(self._records)}, "
            f"image_key={self._image_key!r}, "
            f"records={self._records!r})"
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        return self._records[index]


class LoadAndSliceCALVIN(grain.transforms.RandomMap):
    """
    Loads a random temporal slice of consecutive NPZ frames from a CALVIN episode.

    Returns a dict with "videos": (seq_len, H, W, C) uint8 array.
    """

    def __init__(
        self,
        seq_len: int,
        image_key: str = "rgb_static",
        image_h: Optional[int] = None,
        image_w: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.image_key = image_key
        self.image_h = image_h
        self.image_w = image_w

    def random_map(self, record: tuple, rng: np.random.Generator) -> Optional[dict]:
        data_dir, start_frame_id, end_frame_id = record

        # end_frame_id is inclusive
        num_frames = end_frame_id - start_frame_id + 1
        if num_frames < self.seq_len:
            return None

        offset = rng.integers(0, num_frames - self.seq_len + 1)

        frames_list = []
        npz_path = None
        try:
            for i in range(self.seq_len):
                frame_id = start_frame_id + offset + i
                npz_path = os.path.join(data_dir, f"episode_{frame_id:07d}.npz")
                with np.load(npz_path) as npz_data:
                    frame = npz_data[self.image_key].copy()  # e.g. (200, 200, 3) uint8
                frames_list.append(frame)
        except Exception as e:
            print(f"Error loading CALVIN frame {npz_path}: {e}")
            return None

        frames = np.stack(frames_list, axis=0)  # (T, H, W, C) uint8

        if self.image_h is not None and self.image_w is not None:
            if frames.shape[1] != self.image_h or frames.shape[2] != self.image_w:
                resized = np.empty(
                    (self.seq_len, self.image_h, self.image_w, frames.shape[3]),
                    dtype=np.uint8,
                )
                for i in range(self.seq_len):
                    resized[i] = cv2.resize(
                        frames[i],
                        (self.image_w, self.image_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                frames = resized

        return {"videos": frames}


class FilterNone(grain.transforms.Filter):
    """Filter out None values (failed loads or too-short episodes)."""

    def filter(self, element: Any) -> bool:
        return element is not None


def get_calvin_dataloader(
    data_dirs: list[str],
    seq_len: int,
    global_batch_size: int,
    image_key: str = "rgb_static",
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    num_workers: int = 4,
    prefetch_buffer_size: int = 2,
    seed: int = 42,
):
    """
    Creates a Grain data loading pipeline for CALVIN NPZ datasets.

    Yields batches of {"videos": (B, T, H, W, C) uint8} compatible with
    the dreamer4 tokenizer training loop.

    Args:
        data_dirs: List of CALVIN data directories (each must contain
                   ep_start_end_ids.npy and episode_*.npz files).
                   Pass separate loaders for train vs. validation splits
                   (e.g. ".../training/" and ".../validation/").
        seq_len: Number of frames per sample.
        global_batch_size: Total batch size across all JAX processes.
        image_key: NPZ key for image data. One of:
                   "rgb_static" (200x200x3),
                   "rgb_gripper" (84x84x3),
                   "rgb_tactile" (160x120x6).
        image_h: Target image height (None = keep original).
        image_w: Target image width (None = keep original).
        num_workers: Number of data loading workers.
        prefetch_buffer_size: Prefetch buffer size.
        seed: Random seed.

    Returns:
        A Grain DataLoader yielding {"videos": (B, T, H, W, C) uint8}.
    """
    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = CALVINDataSource(data_dirs, image_key=image_key)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        LoadAndSliceCALVIN(
            seq_len=seq_len,
            image_key=image_key,
            image_h=image_h,
            image_w=image_w,
        ),
        FilterNone(),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=2,
    )

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=4,
        read_options=read_options,
    )

    return dataloader
