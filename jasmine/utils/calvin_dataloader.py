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

    Returns a dict with:
      "videos": (seq_len, H, W, C) uint8 array
      "actions": (seq_len, 7) float64 array  — only if load_actions=True
    """

    def __init__(
        self,
        seq_len: int,
        image_key: str = "rgb_static",
        image_h: Optional[int] = None,
        image_w: Optional[int] = None,
        load_actions: bool = False,
        action_key: str = "rel_actions",  # "actions" or "rel_actions"
    ):
        self.seq_len = seq_len
        self.image_key = image_key
        self.image_h = image_h
        self.image_w = image_w
        self.load_actions = load_actions
        self.action_key = action_key

    def random_map(self, record: tuple, rng: np.random.Generator) -> Optional[dict]:
        data_dir, start_frame_id, end_frame_id = record

        # end_frame_id is inclusive
        num_frames = end_frame_id - start_frame_id + 1
        if num_frames < self.seq_len:
            return None

        offset = rng.integers(0, num_frames - self.seq_len + 1)

        frames_list = []
        actions_list = []
        npz_path = None
        try:
            for i in range(self.seq_len):
                frame_id = start_frame_id + offset + i
                npz_path = os.path.join(data_dir, f"episode_{frame_id:07d}.npz")
                with np.load(npz_path) as npz_data:
                    frame = npz_data[self.image_key].copy()  # e.g. (200, 200, 3) uint8
                    if self.load_actions:
                        actions_list.append(npz_data[self.action_key].copy())  # (7,) float64
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

        result = {"videos": frames}
        if self.load_actions:
            result["actions"] = np.stack(actions_list, axis=0)  # (T, 7) float64
        return result


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
    load_actions: bool = False,
    action_key: str = "rel_actions",
):
    """
    Creates a Grain data loading pipeline for CALVIN NPZ datasets.

    Yields batches of {"videos": (B, T, H, W, C) uint8} compatible with
    the dreamer4 tokenizer training loop.  When load_actions=True also
    yields {"actions": (B, T, 7) float64}.

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
        load_actions: If True, also load actions from the NPZ files.
        action_key: NPZ key for action data ("actions" or "rel_actions").

    Returns:
        A Grain DataLoader yielding {"videos": (B, T, H, W, C) uint8}
        and optionally {"actions": (B, T, 7) float64} when load_actions=True.
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
            load_actions=load_actions,
            action_key=action_key,
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
        worker_buffer_size=32,
        read_options=read_options,
    )

    return dataloader


class CALVINLangDataSource(grain.sources.RandomAccessDataSource):
    """Language-annotated CALVIN segments with precomputed sentence embeddings."""

    def __init__(
        self,
        data_dirs: list[str],
        lang_folder: str = "lang_all-mpnet-base-v2",
    ):
        self._records: list[tuple[str, int, int, np.ndarray, str]] = []
        for data_dir in data_dirs:
            lang_path = os.path.join(data_dir, lang_folder, "auto_lang_ann.npy")
            lang_data = np.load(lang_path, allow_pickle=True).item()
            spans = lang_data["info"]["indx"]
            embeddings = lang_data["language"]["emb"]
            annotations = lang_data["language"].get("ann", [""] * len(spans))
            for ann_idx, (start_frame_id, end_frame_id) in enumerate(spans):
                sentence_embedding = np.asarray(embeddings[ann_idx], dtype=np.float32).reshape(-1)
                annotation = str(annotations[ann_idx])
                self._records.append(
                    (
                        data_dir,
                        int(start_frame_id),
                        int(end_frame_id),
                        sentence_embedding,
                        annotation,
                    )
                )

        print(
            f"CALVINLangDataSource: {len(self._records)} annotated segments from "
            f"{len(data_dirs)} dir(s), lang_folder={lang_folder}"
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        return self._records[index]


class LoadAndSliceCALVINLang(grain.transforms.RandomMap):
    """Load a fixed-length CALVIN language segment with left padding and sparse terminal reward."""

    def __init__(
        self,
        seq_len: int,
        image_key: str = "rgb_static",
        image_h: Optional[int] = None,
        image_w: Optional[int] = None,
        action_key: str = "rel_actions",
        reward_at_end: bool = True,
    ):
        self.seq_len = seq_len
        self.image_key = image_key
        self.image_h = image_h
        self.image_w = image_w
        self.action_key = action_key
        self.reward_at_end = reward_at_end

    def random_map(
        self,
        record: tuple[str, int, int, np.ndarray, str],
        rng: np.random.Generator,
    ) -> Optional[dict]:
        del rng
        data_dir, start_frame_id, end_frame_id, task_embedding, _annotation = record

        if end_frame_id < start_frame_id:
            return None

        # Align each sample to the annotated segment end so the sparse success reward is visible.
        load_start = max(start_frame_id, end_frame_id + 1 - self.seq_len)
        frame_ids = list(range(load_start, end_frame_id + 1))
        actual_len = len(frame_ids)
        if actual_len <= 0:
            return None

        frames_list = []
        actions_list = []
        npz_path = None
        try:
            for frame_id in frame_ids:
                npz_path = os.path.join(data_dir, f"episode_{frame_id:07d}.npz")
                with np.load(npz_path) as npz_data:
                    frames_list.append(npz_data[self.image_key].copy())
                    actions_list.append(np.asarray(npz_data[self.action_key], dtype=np.float32))
        except Exception as e:
            print(f"Error loading CALVIN lang frame {npz_path}: {e}")
            return None

        frames = np.stack(frames_list, axis=0)
        actions = np.stack(actions_list, axis=0).astype(np.float32)

        if self.image_h is not None and self.image_w is not None:
            if frames.shape[1] != self.image_h or frames.shape[2] != self.image_w:
                resized = np.empty(
                    (actual_len, self.image_h, self.image_w, frames.shape[3]),
                    dtype=np.uint8,
                )
                for i in range(actual_len):
                    resized[i] = cv2.resize(
                        frames[i],
                        (self.image_w, self.image_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                frames = resized

        pad_len = self.seq_len - actual_len
        valid_mask = np.zeros((self.seq_len,), dtype=np.bool_)
        valid_mask[pad_len:] = True

        if pad_len > 0:
            first_frame = frames[0]
            pad_frames = np.repeat(first_frame[None], pad_len, axis=0)
            pad_actions = np.full((pad_len, actions.shape[-1]), np.nan, dtype=np.float32)
            frames = np.concatenate([pad_frames, frames], axis=0)
            actions = np.concatenate([pad_actions, actions], axis=0)

        action_mask = valid_mask.copy()
        last_valid_idx = self.seq_len - 1
        action_mask[last_valid_idx] = False

        rewards = np.zeros((self.seq_len,), dtype=np.float32)
        if self.reward_at_end:
            rewards[last_valid_idx] = 1.0

        return {
            "videos": frames,
            "actions": actions,
            "rewards": rewards,
            "valid_mask": valid_mask,
            "action_mask": action_mask,
            "task_embedding": np.asarray(task_embedding, dtype=np.float32),
        }


def get_calvin_lang_dataloader(
    data_dirs: list[str],
    seq_len: int,
    global_batch_size: int,
    lang_folder: str = "lang_all-mpnet-base-v2",
    image_key: str = "rgb_static",
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    num_workers: int = 4,
    prefetch_buffer_size: int = 2,
    seed: int = 42,
    action_key: str = "rel_actions",
    reward_at_end: bool = True,
):
    """Create a Grain dataloader for CALVIN language segments with sentence embeddings."""
    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = CALVINLangDataSource(data_dirs, lang_folder=lang_folder)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        LoadAndSliceCALVINLang(
            seq_len=seq_len,
            image_key=image_key,
            image_h=image_h,
            image_w=image_w,
            action_key=action_key,
            reward_at_end=reward_at_end,
        ),
        FilterNone(),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=2,
    )

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=32,
        read_options=read_options,
    )
