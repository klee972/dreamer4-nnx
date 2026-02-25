import jax
import numpy as np
import grain
import h5py
import cv2
from typing import Any, Optional


class RobomimicHDF5DataSource(grain.sources.RandomAccessDataSource):
    """
    A Grain data source that reads demonstration trajectories from robomimic HDF5 files.

    Each record is a tuple of (hdf5_path, demo_key) identifying a single demonstration.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        image_key: str = "agentview_image",
        filter_key: Optional[str] = None,
    ):
        """
        Args:
            hdf5_paths: List of HDF5 file paths.
            image_key: Observation key for image data (e.g. "agentview_image").
            filter_key: Optional filter key (e.g. "train", "valid") to select a
                        subset of demos from the mask/ group in the HDF5 file.
        """
        self._records: list[tuple[str, str, int]] = []  # (path, demo_key, num_samples)
        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if filter_key and f"mask/{filter_key}" in f:
                    demo_keys = [k.decode() if isinstance(k, bytes) else k for k in f[f"mask/{filter_key}"][:]]
                else:
                    demo_keys = sorted(
                        [k for k in f["data"].keys() if k.startswith("demo")],
                        key=lambda x: int(x.split("_")[-1]),
                    )
                for dk in demo_keys:
                    num_samples = int(f[f"data/{dk}"].attrs["num_samples"])
                    self._records.append((path, dk, num_samples))

        self._image_key = image_key
        print(
            f"RobomimicHDF5DataSource: {len(self._records)} demos from "
            f"{len(hdf5_paths)} file(s), image_key={image_key}"
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        return self._records[index]


class LoadAndSliceRobomimic(grain.transforms.RandomMap):
    """
    Loads image observations from a robomimic HDF5 demo and extracts a random
    temporal slice suitable for tokenizer training.

    Returns a dict with "videos": (seq_len, H, W, C) uint8 array.
    """

    def __init__(
        self,
        seq_len: int,
        image_key: str = "agentview_image",
        image_h: Optional[int] = None,
        image_w: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.image_key = image_key
        self.image_h = image_h
        self.image_w = image_w

    def random_map(self, record: tuple, rng: np.random.Generator) -> Optional[dict]:
        hdf5_path, demo_key, num_samples = record

        if num_samples < self.seq_len:
            return None

        start = rng.integers(0, num_samples - self.seq_len + 1)
        end = start + self.seq_len

        try:
            with h5py.File(hdf5_path, "r", swmr=True) as f:
                frames = f[f"data/{demo_key}/obs/{self.image_key}"][start:end]  # (T, H, W, C)
        except Exception as e:
            print(f"Error reading {hdf5_path}/{demo_key}: {e}")
            return None

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
    """Filter out None values (failed loads or too-short demos)."""

    def filter(self, element: Any) -> bool:
        return element is not None


def get_robomimic_dataloader(
    hdf5_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_key: str = "agentview_image",
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    filter_key: Optional[str] = None,
    num_workers: int = 4,
    prefetch_buffer_size: int = 2,
    seed: int = 42,
):
    """
    Creates a Grain data loading pipeline for robomimic HDF5 datasets.

    Yields batches of {"videos": (B, T, H, W, C) uint8} compatible with
    the dreamer4 tokenizer training loop.

    Args:
        hdf5_paths: List of robomimic HDF5 file paths.
        seq_len: Number of frames per sample.
        global_batch_size: Total batch size across all JAX processes.
        image_key: Observation key for images (e.g. "agentview_image",
                   "robot0_eye_in_hand_image").
        image_h: Target image height (None = keep original).
        image_w: Target image width (None = keep original).
        filter_key: Optional mask key to select demo subset (e.g. "train", "valid").
        num_workers: Number of data loading workers.
        prefetch_buffer_size: Prefetch buffer size.
        seed: Random seed.

    Returns:
        A Grain DataLoader.
    """
    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = RobomimicHDF5DataSource(hdf5_paths, image_key=image_key, filter_key=filter_key)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        LoadAndSliceRobomimic(
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
