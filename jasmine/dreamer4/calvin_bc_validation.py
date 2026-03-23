"""Environment validation utilities for CALVIN BC Dreamer4 training.

This module keeps policy inference inside the worldmodel JAX process and talks
to a separate CALVIN worker process that can run in its own Python environment.
"""

from __future__ import annotations

import base64
import json
import os
import select
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from einops import reduce

from jasmine.utils.dreamer4_utils import pack_bottleneck_to_spatial

_JSON_PREFIX = "__CALVIN_JSON__"


@dataclass
class CalvinValRecord:
    data_dir: str
    start_frame_id: int
    end_frame_id: int
    task: str
    annotation: str
    task_embedding: np.ndarray


@dataclass
class CalvinWorkerClient:
    process: subprocess.Popen[str]
    env_data_dir: str
    timeout_sec: float


@dataclass
class CalvinEnvValidationState:
    records: list[CalvinValRecord]
    worker: CalvinWorkerClient


def load_calvin_val_records(data_dirs: list[str], lang_folder: str) -> list[CalvinValRecord]:
    records: list[CalvinValRecord] = []
    for data_dir in data_dirs:
        lang_path = Path(data_dir) / lang_folder / "auto_lang_ann.npy"
        lang_data = np.load(lang_path, allow_pickle=True).item()
        spans = lang_data["info"]["indx"]
        anns = lang_data["language"].get("ann", [""] * len(spans))
        tasks = lang_data["language"].get("task", [""] * len(spans))
        embs = lang_data["language"]["emb"]
        for idx, (start_frame_id, end_frame_id) in enumerate(spans):
            records.append(
                CalvinValRecord(
                    data_dir=str(data_dir),
                    start_frame_id=int(start_frame_id),
                    end_frame_id=int(end_frame_id),
                    task=str(tasks[idx]),
                    annotation=str(anns[idx]),
                    task_embedding=np.asarray(embs[idx], dtype=np.float32).reshape(-1),
                )
            )
    return records


def _read_ready_line(stream: Any, timeout_sec: float) -> str:
    ready, _, _ = select.select([stream], [], [], timeout_sec)
    if not ready:
        raise TimeoutError(f"Timed out waiting {timeout_sec:.1f}s for CALVIN worker response.")
    line = stream.readline()
    if not line:
        raise RuntimeError("CALVIN worker closed its stdout pipe unexpectedly.")
    return line


def _read_stderr_snippet(process: subprocess.Popen[str]) -> str:
    if process.stderr is None:
        return ""
    try:
        ready, _, _ = select.select([process.stderr], [], [], 0.0)
    except (ValueError, OSError):
        return ""
    if not ready:
        return ""
    try:
        return os.read(process.stderr.fileno(), 8192).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _format_worker_failure(worker: CalvinWorkerClient, context: str) -> str:
    code = worker.process.poll()
    stderr = _read_stderr_snippet(worker.process)
    details = [context]
    if code is not None:
        details.append(f"exit_code={code}")
    if stderr:
        details.append(f"stderr={stderr}")
    return "; ".join(details)


def _send_worker_command(worker: CalvinWorkerClient, payload: dict[str, Any]) -> dict[str, Any]:
    if worker.process.stdin is None or worker.process.stdout is None:
        raise RuntimeError("CALVIN worker pipes are unavailable.")
    if worker.process.poll() is not None:
        raise RuntimeError(_format_worker_failure(worker, "CALVIN worker is not running"))

    try:
        worker.process.stdin.write(json.dumps(payload) + "\n")
        worker.process.stdin.flush()
    except BrokenPipeError as exc:
        raise RuntimeError(_format_worker_failure(worker, "Failed to write to CALVIN worker")) from exc

    while True:
        try:
            line = _read_ready_line(worker.process.stdout, worker.timeout_sec)
        except TimeoutError as exc:
            cmd = payload.get("cmd", "unknown")
            raise RuntimeError(
                _format_worker_failure(worker, f"Timed out waiting for CALVIN worker command {cmd!r}")
            ) from exc
        if not line.startswith(_JSON_PREFIX):
            continue
        response = json.loads(line[len(_JSON_PREFIX) :])
        if not response.get("ok", False):
            cmd = payload.get("cmd", "unknown")
            raise RuntimeError(
                _format_worker_failure(worker, f"CALVIN worker command {cmd!r} failed: {response.get('error', 'unknown error')}")
            )
        return response


def _worker_script_path() -> Path:
    path = Path(__file__).with_name("calvin_env_worker.py").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing CALVIN worker script: {path}")
    return path


def _start_calvin_worker(args: Any, data_dir: str) -> CalvinWorkerClient:
    calvin_python_bin = str(getattr(args, "calvin_python_bin", "")).strip()
    if not calvin_python_bin:
        raise ValueError(
            "Set --calvin-python-bin to the Python executable inside the separate CALVIN environment, "
            "or disable env validation with --val-interval 0."
        )

    repo_root = Path(args.calvin_repo_root).resolve()
    worker_env = os.environ.copy()
    python_paths = [
        str(repo_root),
        str(repo_root / "calvin_env"),
        str(repo_root / "calvin_models"),
    ]
    existing_pythonpath = worker_env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    worker_env["PYTHONPATH"] = os.pathsep.join(python_paths)

    worker_env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        calvin_python_bin,
        "-u",
        str(_worker_script_path()),
        "--calvin-repo-root",
        str(repo_root),
        "--env-data-dir",
        str(Path(data_dir).resolve()),
    ]
    if getattr(args, "val_show_gui", False):
        cmd.append("--show-gui")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=worker_env,
    )
    worker = CalvinWorkerClient(
        process=process,
        env_data_dir=str(Path(data_dir).resolve()),
        timeout_sec=float(getattr(args, "val_worker_timeout_sec", 120.0)),
    )
    _send_worker_command(worker, {"cmd": "ping"})
    return worker


def _restart_worker_for_data_dir(
    args: Any,
    state: CalvinEnvValidationState,
    data_dir: str,
) -> CalvinWorkerClient:
    close_calvin_env_validation(state)
    worker = _start_calvin_worker(args, data_dir)
    state.worker = worker
    return worker


def init_calvin_env_validation(args: Any) -> CalvinEnvValidationState:
    if not args.val_data_dirs:
        raise ValueError("val_data_dirs must be set for CALVIN environment validation.")
    records = load_calvin_val_records(args.val_data_dirs, args.lang_folder)
    worker = _start_calvin_worker(args, args.val_data_dirs[0])
    return CalvinEnvValidationState(records=records, worker=worker)


def close_calvin_env_validation(state: CalvinEnvValidationState | None) -> None:
    if state is None:
        return
    worker = getattr(state, "worker", None)
    if worker is None:
        return
    process = worker.process
    if process.poll() is None:
        try:
            _send_worker_command(worker, {"cmd": "close"})
        except Exception:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    state.worker = None  # type: ignore[assignment]


def _shift_actions(raw_actions: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    shifted = raw_actions[:, :-1].astype(dtype)
    sentinel = jnp.full((raw_actions.shape[0], 1, raw_actions.shape[-1]), jnp.nan, dtype=dtype)
    return jnp.concatenate([sentinel, shifted], axis=1)


def _build_context_window(
    frames_history: list[np.ndarray],
    actions_history: list[np.ndarray],
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    actual_len = min(len(frames_history), context_len)
    frames_ctx = frames_history[-actual_len:]
    global_start = len(frames_history) - actual_len

    raw_actions_ctx = np.full((actual_len, 7), np.nan, dtype=np.float32)
    for local_idx in range(actual_len - 1):
        global_action_idx = global_start + local_idx
        if global_action_idx < len(actions_history):
            raw_actions_ctx[local_idx] = actions_history[global_action_idx]

    frames = np.stack(frames_ctx, axis=0).astype(np.uint8)
    if actual_len < context_len:
        pad_len = context_len - actual_len
        pad_frames = np.repeat(frames[:1], pad_len, axis=0)
        pad_actions = np.full((pad_len, 7), np.nan, dtype=np.float32)
        frames = np.concatenate([pad_frames, frames], axis=0)
        raw_actions_ctx = np.concatenate([pad_actions, raw_actions_ctx], axis=0)
    return frames, raw_actions_ctx


def _predict_action(
    args: Any,
    tokenizer: nnx.Module,
    model: nnx.Module,
    context_frames: np.ndarray,
    context_actions: np.ndarray,
    task_embedding: np.ndarray,
) -> np.ndarray:
    T = context_frames.shape[0]
    if context_frames.shape[1] != args.image_height or context_frames.shape[2] != args.image_width:
        context_frames = np.stack([
            cv2.resize(frame, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
            for frame in context_frames
        ])
    videos = jnp.asarray(context_frames[None], dtype=jnp.float32) / 255.0
    raw_actions = jnp.asarray(context_actions[None], dtype=jnp.float32)
    task_emb = jnp.asarray(task_embedding[None], dtype=args.dtype)

    z_btld = tokenizer.mask_and_encode(videos.astype(args.dtype), rng=None, training=False)["z"]
    z1 = pack_bottleneck_to_spatial(
        z_btld,
        n_spatial=args.dyna_n_spatial,
        k=args.dyna_packing_factor,
    ).astype(args.dtype)
    actions_in = _shift_actions(raw_actions, args.dtype)
    step_idx = jnp.full((1, T), jnp.log2(args.dyna_k_max).astype(jnp.int32), dtype=jnp.int32)
    signal_idx = jnp.full((1, T), args.dyna_k_max - 1, dtype=jnp.int32)
    agent_tokens = jnp.broadcast_to(
        task_emb[:, None, None, :],
        (1, T, args.dyna_n_agent, args.dyna_d_model),
    )

    _, h_btnd = model.dynamics(
        actions_in,
        step_idx,
        signal_idx,
        z1,
        agent_tokens=agent_tokens,
    )
    if h_btnd is None:
        raise ValueError("Dynamics did not return agent readouts during CALVIN env validation.")

    h_pooled = reduce(h_btnd, "b t n_agent d_model -> b t d_model", "mean")
    action_pred = model.policy_head(h_pooled, deterministic=True)
    action = np.array(jax.device_get(action_pred[0, -1, 0]), dtype=np.float32, copy=True)
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = 1.0 if action[6] >= 0.0 else -1.0
    return action


def _load_demo_frames(record: CalvinValRecord, max_steps: int) -> list[np.ndarray]:
    max_frame_id = min(record.end_frame_id, record.start_frame_id + max_steps)
    frames: list[np.ndarray] = []
    for frame_id in range(record.start_frame_id, max_frame_id + 1):
        npz_path = Path(record.data_dir) / f"episode_{frame_id:07d}.npz"
        with np.load(npz_path) as npz_data:
            frames.append(np.asarray(npz_data["rgb_static"], dtype=np.uint8))
    return frames


def _annotate_frame(frame: np.ndarray, annotation: str, panel: str, status: str) -> np.ndarray:
    img = frame.copy()
    cv2.putText(img, panel, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    color = (60, 220, 60) if status == "success" else (80, 80, 255) if status == "fail" else (255, 255, 255)
    cv2.putText(img, status, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if annotation:
        cv2.putText(img, annotation, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, annotation, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _make_comparison_video(
    demo_frames: list[np.ndarray],
    rollout_frames: list[np.ndarray],
    annotation: str,
    success: bool,
) -> np.ndarray:
    if not demo_frames:
        raise ValueError("demo_frames must not be empty")
    if not rollout_frames:
        raise ValueError("rollout_frames must not be empty")

    total = max(len(demo_frames), len(rollout_frames))
    demo_last = demo_frames[-1]
    rollout_last = rollout_frames[-1]
    out = []
    status = "success" if success else "fail"
    for t in range(total):
        demo = demo_frames[t] if t < len(demo_frames) else demo_last
        rollout = rollout_frames[t] if t < len(rollout_frames) else rollout_last
        demo = _annotate_frame(demo, annotation, "demo", "reference")
        rollout = _annotate_frame(rollout, annotation, "policy", status)
        out.append(np.concatenate([demo, rollout], axis=1))
    return np.stack(out, axis=0).astype(np.uint8)


def _decode_worker_frame(frame_jpg_b64: str) -> np.ndarray:
    encoded = base64.b64decode(frame_jpg_b64.encode("ascii"))
    frame_bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode frame returned by CALVIN worker.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _reset_rollout(worker: CalvinWorkerClient, record: CalvinValRecord) -> np.ndarray:
    response = _send_worker_command(
        worker,
        {
            "cmd": "reset",
            "data_dir": str(Path(record.data_dir).resolve()),
            "start_frame_id": int(record.start_frame_id),
            "task": record.task,
        },
    )
    return _decode_worker_frame(response["frame_jpg_b64"])


def _step_rollout(worker: CalvinWorkerClient, action: np.ndarray) -> tuple[np.ndarray, bool]:
    response = _send_worker_command(
        worker,
        {
            "cmd": "step",
            "action": np.asarray(action, dtype=np.float32).tolist(),
        },
    )
    frame = _decode_worker_frame(response["frame_jpg_b64"])
    return frame, bool(response["success"])


def run_calvin_env_validation(
    args: Any,
    tokenizer: nnx.Module,
    model: nnx.Module,
    step: int,
    state: CalvinEnvValidationState,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if args.val_num_rollouts <= 0:
        return {}, {}
    if not state.records:
        raise ValueError("No CALVIN validation language records were found.")

    rng = np.random.default_rng(args.seed + int(step))
    sample_count = min(args.val_num_rollouts, len(state.records))
    sample_indices = rng.choice(len(state.records), size=sample_count, replace=False)

    successes = []
    rollout_steps = []
    videos: dict[str, np.ndarray] = {}

    for video_idx, rec_idx in enumerate(sample_indices):
        record = state.records[int(rec_idx)]
        worker = state.worker
        resolved_data_dir = str(Path(record.data_dir).resolve())
        if worker.env_data_dir != resolved_data_dir:
            worker = _restart_worker_for_data_dir(args, state, resolved_data_dir)

        first_frame = _reset_rollout(worker, record)
        frames_history = [first_frame]
        actions_history: list[np.ndarray] = []
        rollout_frames = [first_frame]
        demo_frames = _load_demo_frames(record, args.val_max_steps)

        success = False
        executed_steps = 0
        for rollout_step in range(args.val_max_steps):
            context_frames, context_actions = _build_context_window(
                frames_history,
                actions_history,
                context_len=args.val_context_len,
            )
            action = _predict_action(
                args,
                tokenizer,
                model,
                context_frames,
                context_actions,
                record.task_embedding,
            )
            frame, success = _step_rollout(worker, action)
            actions_history.append(action.astype(np.float32))
            frames_history.append(frame)
            rollout_frames.append(frame)
            executed_steps = rollout_step + 1
            if success:
                break

        successes.append(float(success))
        rollout_steps.append(float(executed_steps))

        if video_idx < args.val_num_videos:
            key = f"rollout_{video_idx:02d}_{record.task}"
            videos[key] = _make_comparison_video(demo_frames, rollout_frames, record.annotation, success)

    metrics = {
        "val/env_success_rate": float(np.mean(successes)) if successes else 0.0,
        "val/env_mean_steps": float(np.mean(rollout_steps)) if rollout_steps else 0.0,
        "val/env_num_rollouts": float(sample_count),
    }
    return metrics, videos
