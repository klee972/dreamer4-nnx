"""Standalone CALVIN environment worker.

Run this script with the separate CALVIN Python environment. It reads JSON
commands from stdin and writes prefixed JSON responses to stdout.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _patch_numpy_legacy_aliases() -> None:
    alias_map = {
        "int": int,
        "float": float,
        "bool": bool,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, value)


_patch_numpy_legacy_aliases()

_JSON_PREFIX = "__CALVIN_JSON__"


def _ensure_calvin_imports(calvin_repo_root: str) -> None:
    root = Path(calvin_repo_root).resolve()
    candidates = [root, root / "calvin_models", root / "calvin_env"]
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass
class WorkerState:
    calvin_repo_root: Path
    default_env_data_dir: str
    show_gui: bool
    get_env_fn: Any | None = None
    env: Any | None = None
    task_oracle: Any | None = None
    env_data_dir: str = ""
    start_info: Any = None
    current_task: str = ""


def _build_state(args: argparse.Namespace) -> WorkerState:
    return WorkerState(
        calvin_repo_root=Path(args.calvin_repo_root).resolve(),
        default_env_data_dir=str(Path(args.env_data_dir).resolve()),
        show_gui=bool(args.show_gui),
    )


def _ensure_runtime_loaded(state: WorkerState) -> None:
    if state.get_env_fn is not None and state.task_oracle is not None:
        return

    _ensure_calvin_imports(str(state.calvin_repo_root))

    from omegaconf import OmegaConf
    from calvin_env.envs.play_table_env import get_env
    from calvin_env.envs.tasks import Tasks

    tasks_cfg_path = (
        state.calvin_repo_root
        / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_cfg = OmegaConf.load(tasks_cfg_path)
    state.task_oracle = Tasks(task_cfg.tasks)
    state.get_env_fn = get_env


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(_JSON_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _encode_frame(frame_rgb: np.ndarray) -> str:
    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok:
        raise ValueError("Failed to JPEG-encode CALVIN rollout frame.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _ensure_env_for_dir(state: WorkerState, data_dir: str) -> None:
    _ensure_runtime_loaded(state)
    resolved = str(Path(data_dir).resolve())
    if state.env is not None and resolved == state.env_data_dir:
        return
    if state.env is not None:
        state.env.close()
    if state.get_env_fn is None:
        raise RuntimeError("CALVIN get_env function was not initialized.")
    state.env = state.get_env_fn(resolved, show_gui=state.show_gui)
    state.env_data_dir = resolved


def _handle_reset(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    data_dir = str(payload.get("data_dir", state.default_env_data_dir))
    _ensure_env_for_dir(state, data_dir)
    start_frame_id = int(payload["start_frame_id"])
    state.current_task = str(payload["task"])

    npz_path = Path(state.env_data_dir) / f"episode_{start_frame_id:07d}.npz"
    with np.load(npz_path) as npz_data:
        robot_obs = np.asarray(npz_data["robot_obs"], dtype=np.float32)
        scene_obs = np.asarray(npz_data["scene_obs"], dtype=np.float32)

    if state.env is None:
        raise RuntimeError("CALVIN environment was not initialized.")
    obs = state.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    state.start_info = state.env.get_info()
    frame = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
    return {
        "ok": True,
        "frame_jpg_b64": _encode_frame(frame),
    }


def _handle_step(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    if state.start_info is None:
        raise RuntimeError("reset must be called before step")
    if state.env is None or state.task_oracle is None:
        raise RuntimeError("CALVIN environment runtime is not initialized.")
    action = np.asarray(payload["action"], dtype=np.float32)
    obs, _, _, current_info = state.env.step(action)
    frame = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
    success = bool(
        state.task_oracle.get_task_info_for_set(
            state.start_info,
            current_info,
            {state.current_task},
        )
    )
    return {
        "ok": True,
        "frame_jpg_b64": _encode_frame(frame),
        "success": success,
    }


def _handle_command(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    cmd = payload.get("cmd")
    if cmd == "ping":
        return {"ok": True, "message": "pong"}
    if cmd == "reset":
        return _handle_reset(state, payload)
    if cmd == "step":
        return _handle_step(state, payload)
    if cmd == "close":
        return {"ok": True, "closing": True}
    raise ValueError(f"Unknown command: {cmd}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-repo-root", required=True)
    parser.add_argument("--env-data-dir", required=True)
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()

    state = _build_state(args)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                response = _handle_command(state, payload)
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            _write_response(response)
            if response.get("closing"):
                break
    finally:
        if state.env is not None:
            state.env.close()


if __name__ == "__main__":
    main()
