"""Tests for `simanvil.trace` & `simanvil.inspect`: causal replay, 0.15x anomaly weighting & HTML bundle."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import mujoco as mj
import numpy as np

from simanvil import AnvilRecorder, export_inspection_bundle, load_episodes


DRAWER_XML = """
<mujoco model="drawer_rig">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0.3 -0.3 1.0" dir="0 0.2 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.02"/>
    <camera name="overview_cam" pos="0.55 -0.45 0.38" xyaxes="0.64 0.77 0 -0.42 0.35 0.84" fovy="42"/>
    <body name="cabinet_frame" pos="0.25 0 0.08">
      <geom name="cabinet_shell" type="box" size="0.09 0.12 0.08" contype="0" conaffinity="0" rgba="0.22 0.28 0.36 0.4"/>
      <body name="sliding_drawer" pos="-0.02 0 0.01">
        <joint name="drawer_slide" type="slide" axis="-1 0 0" range="0 0.14" damping="2.0"/>
        <geom name="drawer_box" type="box" size="0.075 0.095 0.035" mass="0.35" rgba="0.85 0.62 0.18 1"/>
      </body>
    </body>
    <body name="pusher_finger" pos="0.05 0 0.09">
      <joint name="finger_x" type="slide" axis="1 0 0" range="-0.05 0.22" damping="4.0"/>
      <geom name="finger_pad" type="sphere" size="0.016" mass="0.15" rgba="0.22 0.72 0.92 1"/>
      <camera name="wrist_camera" pos="-0.06 0 0.08" xyaxes="0 -1 0 0.45 0 0.89" fovy="58"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_drawer" joint="drawer_slide" kp="60" ctrlrange="0 0.14"/>
    <position name="act_finger" joint="finger_x" kp="80" ctrlrange="-0.05 0.22"/>
  </actuator>
</mujoco>
"""


def _assert_valid_glb(path: Path) -> None:
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw)
    assert magic == b"glTF" and version == 2 and length == len(raw)


def test_causal_recorder_and_html_inspection_bundle(tmp_path: Path) -> None:
    model = mj.MjModel.from_xml_string(DRAWER_XML)
    data = mj.MjData(model)
    rec = AnvilRecorder(model, data, task="Open and close drawer", robot_type="drawer_rig", fps=25)

    for f in range(20):
        phase = "S2_PULL/retry_slip" if (8 <= f < 13) else ("S1_OPEN/pull" if f < 8 else "S3_CLOSE/push")
        cmd = np.array([0.10, 0.11] if f < 13 else [0.01, 0.02], dtype=np.float64)
        for _ in range(rec.steps_per_frame):
            rec.step(cmd, phase=phase)

    npz_path = rec.save(tmp_path / "drawer_ep.npz", passed=True)
    episodes = load_episodes(npz_path, max_frames=0, render_cameras=True)
    assert len(episodes) == 1
    ep = episodes[0]

    # Exact causal pure-actuator replay check
    assert ep.audit["causal_replay_max_qpos_err"] < 1e-10

    # Verify 5 retry_slip frames were downweighted to 0.15x with progress slope clamped
    weights = [t["bc_loss_weight"] for t in ep.telemetry]
    assert weights.count(0.15) == 5
    assert weights.count(1.0) == 15
    assert ep.telemetry[8]["task_progress_pct"] == ep.telemetry[12]["task_progress_pct"]

    # Export static HTML Session Viewer bundle
    session_dir = tmp_path / "session"
    res = export_inspection_bundle(session_dir, episodes)
    assert (Path(res["session_dir"]) / "index.html").is_file()
    payload = json.loads((session_dir / "session_data.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "simanvil.session.v0.1"
    for mspec in payload["models"].values():
        for b in mspec["bodies"]:
            if b["visual_glb"]:
                _assert_valid_glb(session_dir / b["visual_glb"])
