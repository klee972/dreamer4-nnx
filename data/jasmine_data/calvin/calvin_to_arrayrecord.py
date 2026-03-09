"""
Convert CALVIN NPZ per-frame dataset to ArrayRecord format.

Each ArrayRecord record stores one episode as a pickled dict:
    {
        "raw_video":        bytes  — (T, H, W, 3) uint8 raw bytes
        "sequence_length":  int    — number of frames T
        "actions":          list   — (T, 7) float64 actions (optional)
    }

This matches the CoinRun ArrayRecord format so the existing dataloader.py
(get_dataloader / ProcessEpisodeAndSlice) can be used directly.

Usage (defaults cover ABCD_D → 96p):
    python calvin_to_arrayrecord.py
    python calvin_to_arrayrecord.py --size 200 --no-actions
"""

import argparse
import os
import pickle
import glob
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
import array_record.python.array_record_module as array_record_lib


# ---------------------------------------------------------------------------
# Per-episode worker
# ---------------------------------------------------------------------------

def convert_episode(
    record: tuple[str, int, int],
    size: int,
    load_actions: bool,
) -> bytes | None:
    """Load one CALVIN episode, resize rgb_static, return serialised bytes."""
    data_dir, start_id, end_id = record
    num_frames = end_id - start_id + 1

    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    actions_list = []

    try:
        for i in range(num_frames):
            fid = start_id + i
            path = os.path.join(data_dir, f"episode_{fid:07d}.npz")
            with np.load(path) as npz:
                img = npz["rgb_static"]
                if img.shape[0] != size or img.shape[1] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                frames[i] = img
                if load_actions:
                    actions_list.append(npz["rel_actions"].copy())
    except Exception as e:
        print(f"  ERROR episode {start_id}-{end_id} in {data_dir}: {e}")
        return None

    record_dict: dict = {
        "raw_video": frames.tobytes(),
        "sequence_length": num_frames,
    }
    if load_actions:
        record_dict["actions"] = np.stack(actions_list, axis=0).tolist()

    return pickle.dumps(record_dict)


# ---------------------------------------------------------------------------
# Per-split conversion
# ---------------------------------------------------------------------------

def convert_split(
    src_dir: str,
    dst_dir: str,
    size: int,
    load_actions: bool,
    workers: int,
    shard_size: int,
) -> None:
    src_dir = os.path.abspath(src_dir)
    split_name = os.path.basename(src_dir)
    os.makedirs(dst_dir, exist_ok=True)

    ep_ids = np.load(os.path.join(src_dir, "ep_start_end_ids.npy"))
    records = [(src_dir, int(r[0]), int(r[1])) for r in ep_ids]
    print(f"{split_name}: {len(records)} episodes → {dst_dir}")

    fn = partial(convert_episode, size=size, load_actions=load_actions)

    def open_writer(idx: int):
        path = os.path.join(dst_dir, f"data_{idx:04d}.array_record")
        return array_record_lib.ArrayRecordWriter(path, "group_size:1")

    shard_idx = 0
    shard_count = 0
    total_written = 0
    writer = None

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(fn, records, chunksize=1),
            total=len(records),
            desc=split_name,
            unit="ep",
            dynamic_ncols=True,
            smoothing=0.05,
        ):
            if result is None:
                continue
            if writer is None:
                writer = open_writer(shard_idx)
            writer.write(result)
            shard_count += 1
            total_written += 1
            if shard_count >= shard_size:
                writer.close()
                writer = None
                shard_idx += 1
                shard_count = 0

    if writer is not None:
        writer.close()
    print(f"{split_name}: done — {total_written} episodes in {shard_idx + 1} shard(s).\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dirs", nargs="+",
        default=[
            "/home/4bkang/rl/calvin/dataset/task_ABCD_D/training",
            "/home/4bkang/rl/calvin/dataset/task_ABCD_D/validation",
        ],
    )
    parser.add_argument(
        "--dst-dirs", nargs="+",
        default=[
            "/home/4bkang/rl/worldmodel/data/calvin_96p/train",
            "/home/4bkang/rl/worldmodel/data/calvin_96p/val",
        ],
        help="Must match --src-dirs in length",
    )
    parser.add_argument("--size", type=int, default=96, help="Target H=W for rgb_static")
    parser.add_argument("--no-actions", action="store_true", help="Skip action loading")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=1,
                        help="Episodes per ArrayRecord shard file")
    args = parser.parse_args()

    if len(args.src_dirs) != len(args.dst_dirs):
        raise ValueError("--src-dirs and --dst-dirs must have the same length")

    for src, dst in zip(args.src_dirs, args.dst_dirs):
        convert_split(
            src_dir=src,
            dst_dir=dst,
            size=args.size,
            load_actions=not args.no_actions,
            workers=args.workers,
            shard_size=args.shard_size,
        )



if __name__ == "__main__":
    main()
