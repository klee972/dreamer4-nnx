"""
기존 array_record 파일(가변 길이 에피소드)을 읽어서
정확히 seq_len 프레임짜리 클립으로 잘라 새 array_record로 저장.

각 출력 파일에 clips_per_file 개의 클립이 들어감 (기본값 1).
파일 하나 = 클립 하나 = seq_len 프레임.

Usage:
    python reshard_to_clips.py \
        --src-dirs data/calvin_96p/train data/calvin_96p/val \
        --dst-dirs data/calvin_96p_clips/train data/calvin_96p_clips/val \
        --seq-len 96 \
        --clips-per-file 1 \
        --stride 48          # 0이면 non-overlapping (stride=seq_len)
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from functools import partial

import numpy as np
import array_record.python.array_record_module as array_record_lib
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_array_record(path: str) -> list[bytes]:
    """파일 하나에서 raw bytes 레코드 목록 반환."""
    reader = array_record_lib.ArrayRecordReader(path)
    n = reader.num_records()
    records = [reader.read() for _ in range(n)]
    reader.close()
    return records


def extract_clips(raw_bytes: bytes, seq_len: int, stride: int) -> list[tuple]:
    """
    에피소드 bytes → (T, H, W, C) 배열 → seq_len짜리 클립 리스트.
    stride=0 이면 non-overlapping (stride=seq_len).
    각 클립은 (frames, actions_or_None) 튜플.
    """
    record = pickle.loads(raw_bytes)
    T = record["sequence_length"]
    if T < seq_len:
        return []

    raw = np.frombuffer(record["raw_video"], dtype=np.uint8)
    C = 3
    HW = raw.size // (T * C)
    side = int(round(HW ** 0.5))  # H == W (square)
    frames = raw.reshape(T, side, side, C)

    actions = None
    if "actions" in record:
        actions = np.array(record["actions"])  # (T, action_dim)

    eff_stride = stride if stride > 0 else seq_len
    starts = range(0, T - seq_len + 1, eff_stride)
    clips = []
    for s in starts:
        clip_frames = frames[s : s + seq_len]
        clip_actions = actions[s : s + seq_len] if actions is not None else None
        clips.append((clip_frames, clip_actions))
    return clips


def clip_to_bytes(clip_frames: np.ndarray, clip_actions) -> bytes:
    T = clip_frames.shape[0]
    record = {
        "raw_video": clip_frames.tobytes(),
        "sequence_length": T,
    }
    if clip_actions is not None:
        record["actions"] = clip_actions.tolist()
    return pickle.dumps(record)


# ---------------------------------------------------------------------------
# Per-split conversion
# ---------------------------------------------------------------------------

def convert_split(
    src_dir: str,
    dst_dir: str,
    seq_len: int,
    stride: int,
    clips_per_file: int,
    workers: int,
) -> None:
    import glob as _glob

    src_files = sorted(_glob.glob(os.path.join(src_dir, "*.array_record")))
    if not src_files:
        raise ValueError(f"No .array_record files found in {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)
    split_name = os.path.basename(dst_dir)
    fn = partial(extract_clips, seq_len=seq_len, stride=stride)

    file_idx = 0
    buf_count = 0
    total_clips = 0
    writer = None

    def open_writer(idx: int):
        path = os.path.join(dst_dir, f"data_{idx:04d}.array_record")
        return array_record_lib.ArrayRecordWriter(path, "group_size:1")

    print(f"[{split_name}] Processing {len(src_files)} source file(s)...")
    with Pool(workers) as pool:
        for src_file in tqdm(src_files, desc=f"[{split_name}]", unit="src"):
            raw_records = read_array_record(src_file)
            for clips in pool.imap(fn, raw_records, chunksize=4):
                for clip_frames, clip_actions in clips:
                    if writer is None:
                        writer = open_writer(file_idx)
                    writer.write(clip_to_bytes(clip_frames, clip_actions))
                    buf_count += 1
                    total_clips += 1
                    if buf_count >= clips_per_file:
                        writer.close()
                        writer = None
                        file_idx += 1
                        buf_count = 0

    if writer is not None:
        writer.close()
    total_files = file_idx + (1 if buf_count > 0 else 0)
    print(f"[{split_name}] Done — {total_clips} clips → {total_files} file(s) in {dst_dir}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dirs", nargs="+", default=[
        "/home/4bkang/rl/worldmodel/data/calvin_96p/train",
        "/home/4bkang/rl/worldmodel/data/calvin_96p/val",
    ])
    parser.add_argument("--dst-dirs", nargs="+", default=[
        "/home/4bkang/rl/worldmodel/data/calvin_96p_clips/train",
        "/home/4bkang/rl/worldmodel/data/calvin_96p_clips/val",
    ])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument(
        "--stride", type=int, default=0,
        help="클립 추출 stride. 0이면 non-overlapping (stride=seq_len).",
    )
    parser.add_argument(
        "--clips-per-file", type=int, default=1,
        help="출력 파일 하나에 저장할 클립 수. 기본값 1 = 파일당 96프레임.",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if len(args.src_dirs) != len(args.dst_dirs):
        raise ValueError("--src-dirs와 --dst-dirs 길이가 같아야 합니다.")

    for src, dst in zip(args.src_dirs, args.dst_dirs):
        convert_split(
            src_dir=src,
            dst_dir=dst,
            seq_len=args.seq_len,
            stride=args.stride,
            clips_per_file=args.clips_per_file,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
