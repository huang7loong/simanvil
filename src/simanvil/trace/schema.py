"""Canonical `Episode` memory model and 4-line `AnvilRecorder` (`simanvil.trace`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import mujoco as mj
import numpy as np


INTEGRATION_STATE = mj.mjtState.mjSTATE_INTEGRATION


@dataclass
class Episode:
    """Model-agnostic representation of a recorded MuJoCo episode."""

    key: str
    label: str
    model_key: str
    robot_type: str
    task: str
    fps: int
    model: mj.MjModel
    times: list[float]
    xpos: np.ndarray  # (T, nbody, 3) float32
    xquat: np.ndarray  # (T, nbody, 4) float32 [w, x, y, z]
    obs_state: np.ndarray  # (T, D_s) float32
    state_names: list[str]
    actions: np.ndarray  # (T, D_a) float32
    action_names: list[str]
    contacts: list[list[dict]]
    telemetry: list[dict]
    series_defs: list[dict]
    overview_rgb: list[np.ndarray]
    wrist_rgb: list[np.ndarray]
    audit: dict
    action_substeps: np.ndarray | None = None  # (T, K * D_a) float32
    overview_cam_label: str = "OVERVIEW CAM"
    wrist_cam_label: str = "WRIST / FOCUS CAM"
    forged_assets: dict[str, tuple[Path, dict]] = field(default_factory=dict)
    use_shrinkwrap_mesh: bool = False


class AnvilRecorder:
    """Drop-in causal recorder for any MuJoCo physics loop.

    Saves `(state, qpos, qvel, controls, final_state, steps_per_frame, phase)` so trajectories
    can be causally replay-verified (`Δqpos == 0`), inspected in 3D, and assayed to LeRobot v3.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *,
        task: str = "MuJoCo manipulation episode",
        robot_type: str = "generic_mujoco",
        key: str | None = None,
        label: str | None = None,
        fps: int = 25,
        forged_assets: dict[str, Path] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.task = task
        self.robot_type = robot_type
        self.key = key
        self.label = label
        self.fps = int(fps)
        self.steps_per_frame = max(1, round(1.0 / (self.fps * model.opt.timestep)))
        self.forged_assets = {k: str(Path(v).resolve()) for k, v in (forged_assets or {}).items()}

        self._state_size = mj.mj_stateSize(model, INTEGRATION_STATE)
        self._step_counter = 0
        self._states: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._time: list[float] = []
        self._controls: list[np.ndarray] = []
        self._phases: list[str] = []
        self._last_complete_state: np.ndarray | None = None

    def _snapshot(self) -> np.ndarray:
        buf = np.empty(self._state_size, dtype=np.float64)
        mj.mj_getState(self.model, self.data, buf, INTEGRATION_STATE)
        return buf

    def step(self, ctrl: np.ndarray | None = None, phase: str = "EXECUTE/step") -> None:
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        if self._step_counter % self.steps_per_frame == 0:
            self._states.append(self._snapshot())
            self._qpos.append(self.data.qpos.copy())
            self._qvel.append(self.data.qvel.copy())
            self._time.append(float(self.data.time))
            self._phases.append(str(phase))
        self._controls.append(self.data.ctrl.copy())
        mj.mj_step(self.model, self.data)
        self._step_counter += 1
        if self._step_counter % self.steps_per_frame == 0:
            self._last_complete_state = self._snapshot()

    def save(
        self,
        npz_path: Path,
        *,
        passed: bool = True,
        verdict: str | None = None,
        anomaly_from_frame: int | None = None,
        use_shrinkwrap_mesh: bool = False,
        save_sibling_xml: bool = True,
    ) -> Path:
        out_npz = Path(npz_path).resolve()
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        # Trim trailing partial-frame substeps so len(controls) == len(states) * steps_per_frame
        n_frames = len(self._controls) // self.steps_per_frame
        if n_frames <= 0:
            raise ValueError(
                f"Cannot save episode with 0 complete frames (recorded {len(self._controls)} steps, "
                f"steps_per_frame={self.steps_per_frame})."
            )
        states_arr = np.asarray(self._states[:n_frames], dtype=np.float64)
        qpos_arr = np.asarray(self._qpos[:n_frames], dtype=np.float64)
        qvel_arr = np.asarray(self._qvel[:n_frames], dtype=np.float64)
        time_arr = np.asarray(self._time[:n_frames], dtype=np.float64)
        phases_arr = np.asarray(self._phases[:n_frames])
        ctrl_arr = np.asarray(self._controls[: n_frames * self.steps_per_frame], dtype=np.float64)
        final_state = self._last_complete_state if self._last_complete_state is not None else self._snapshot()

        if save_sibling_xml:
            try:
                mj.mj_saveLastXML(str(out_npz.with_suffix(".xml")), self.model)
            except mj.FatalError:
                # Procedural MjSpec models have no cached XML; save self-contained binary .mjb
                mj.mj_saveModel(self.model, str(out_npz.with_suffix(".mjb")))

        forged_assets_rel: dict[str, str] = {}
        for k, adir_str in self.forged_assets.items():
            try:
                forged_assets_rel[k] = os.path.relpath(adir_str, out_npz.parent)
            except ValueError:
                pass

        meta = {
            "key": self.key or out_npz.stem,
            "label": self.label or f"{self.robot_type} · {out_npz.stem}",
            "passed": bool(passed),
            "verdict": verdict or "",
            "anomaly_from_frame": int(anomaly_from_frame) if anomaly_from_frame is not None else None,
            "use_shrinkwrap_mesh": bool(use_shrinkwrap_mesh),
            "forged_assets": self.forged_assets,
            "forged_assets_rel": forged_assets_rel,
        }

        np.savez_compressed(
            out_npz,
            schema="simanvil.trace.v0.1",
            task=self.task,
            robot_type=self.robot_type,
            fps=np.int64(self.fps),
            steps_per_frame=np.int64(self.steps_per_frame),
            state=states_arr,
            final_state=final_state,
            time=time_arr,
            qpos=qpos_arr,
            qvel=qvel_arr,
            controls=ctrl_arr,
            phase=phases_arr,
            anvil_meta=json.dumps(meta),
        )
        return out_npz
