"""
wandb 비교 이미지에서 row 0 과 row 1 을 가로로 붙여 비교 MP4를 만드는 스크립트.
이미지 구조: 가로 64, 세로 3으로 배열된 프레임 그리드
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_comparison_video(
    image_path: str,
    output_path: str,
    row_a: int = 0,
    row_b: int = 1,
    n_cols: int = 64,
    n_rows: int = 3,
    fps: int = 20,
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"이미지 크기: {W} x {H}")

    frame_w = W // n_cols
    frame_h = H // n_rows
    print(f"개별 프레임 크기: {frame_w} x {frame_h}")
    print(f"비교: row {row_a} (왼쪽) | row {row_b} (오른쪽), 프레임 수: {n_cols}")

    img_np = np.array(img)

    y_a = row_a * frame_h
    y_b = row_b * frame_h

    out_w = frame_w * 2
    out_h = frame_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for col in range(n_cols):
        x_start = col * frame_w
        x_end = x_start + frame_w

        frame_a = img_np[y_a:y_a + frame_h, x_start:x_end]
        frame_b = img_np[y_b:y_b + frame_h, x_start:x_end]

        combined = np.concatenate([frame_a, frame_b], axis=1)  # 가로로 붙이기
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

    out.release()
    print(f"저장 완료: {output_path}  ({out_w} x {out_h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str,
        default="/home/4bkang/rl/jasmine/wandb/run-20260224_065640-6gwkpub8/files/media/images/val/shortcut_d4_pure_AR_comparison_419_1efff643456107031fcb.png",
        help="입력 PNG 이미지 경로",
    )
    parser.add_argument("--output", "-o", type=str, default="comparison.mp4", help="출력 MP4 경로")
    parser.add_argument("--row_a", type=int, default=0, help="왼쪽에 올 row (0-indexed)")
    parser.add_argument("--row_b", type=int, default=1, help="오른쪽에 올 row (0-indexed)")
    parser.add_argument("--cols", type=int, default=64, help="가로 프레임 수")
    parser.add_argument("--rows", type=int, default=3, help="세로 프레임 수")
    parser.add_argument("--fps", type=int, default=20, help="출력 FPS")
    args = parser.parse_args()

    extract_comparison_video(
        image_path=args.image_path,
        output_path=args.output,
        row_a=args.row_a,
        row_b=args.row_b,
        n_cols=args.cols,
        n_rows=args.rows,
        fps=args.fps,
    )

