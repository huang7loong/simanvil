"""Tests for `simanvil.assay`: weighted LeRobot v3 export & PyAV readback verification."""

from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np

from simanvil import AnvilRecorder, assay_to_lerobot_v3, load_episodes


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


def test_assay_lerobot_v3_weighted_export(tmp_path: Path) -> None:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("lerobot optional dependency not installed")

    model = mj.MjModel.from_xml_string(DRAWER_XML)
    data = mj.MjData(model)
    rec = AnvilRecorder(model, data, task="Pull drawer", robot_type="drawer_rig", fps=25)

    for f in range(16):
        phase = "S2/fault_slip" if (6 <= f < 10) else "S1/pull"
        for _ in range(rec.steps_per_frame):
            rec.step(np.array([0.09, 0.10], dtype=np.float64), phase=phase)

    npz_path = rec.save(tmp_path / "ep.npz", passed=True)
    episodes = load_episodes(npz_path, max_frames=0, render_cameras=True)

    ds_dir = tmp_path / "lerobot_v3"
    manifest = assay_to_lerobot_v3(
        episodes,
        ds_dir,
        repo_id="local/simanvil_test_ds",
        anomaly_policy="downweight",
    )
    assert manifest["total_frames"] == 16
    assert manifest["clean_frames_1x"] == 12
    assert manifest["downweighted_frames"] == 4
    assert np.isclose(manifest["effective_weighted_frames"], 12 * 1.0 + 4 * 0.15, atol=1e-2)
    assert manifest["max_decoded_rgb_mae_0_to_1"] < 0.05
    assert (ds_dir / "meta/simanvil_assay.json").is_file()
