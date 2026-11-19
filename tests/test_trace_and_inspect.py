"""Tests for `simanvil.trace` & `simanvil.inspect`: causal replay, 0.15x anomaly weighting & HTML bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import mujoco as mj
import numpy as np

from simanvil import (
    AnvilRecorder,
    export_inspection_bundle,
    load_episodes,
    render_contact_mechanics_reel,
)


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


SMALL_FB_XML = """
<mujoco model="small_fb_block">
  <option timestep="0.002"/>
  <visual><global offwidth="160" offheight="120"/></visual>
  <worldbody>
    <light pos="0.3 -0.3 1.0" dir="0 0.2 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.02"/>
    <camera name="overview_cam" pos="0.35 -0.35 0.25" xyaxes="0.7 0.7 0 -0.35 0.35 0.86" fovy="45"/>
    <camera name="wrist_camera" pos="0.20 -0.20 0.16" xyaxes="0.7 0.7 0 -0.35 0.35 0.86" fovy="55"/>
    <body name="test_block" pos="0 0 0.021">
      <freejoint/>
      <geom name="block_geom" type="box" size="0.02 0.02 0.02" mass="0.1" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_small_framebuffer_contact_sign_and_distinct_model_keys(tmp_path: Path) -> None:
    # Model A: small framebuffer (160x120) with block resting on floor
    m_a = mj.MjModel.from_xml_string(SMALL_FB_XML)
    d_a = mj.MjData(m_a)
    rec_a = AnvilRecorder(m_a, d_a, task="Block rest", robot_type="rig_a", key="run_a", fps=25)
    for _ in range(10 * rec_a.steps_per_frame):
        rec_a.step(phase="S1/settle")
    npz_a = rec_a.save(tmp_path / "dir_a/scene.npz", passed=True)

    # Model B: different scene ALSO saved as `scene.npz` in another dir
    m_b = mj.MjModel.from_xml_string(DRAWER_XML)
    d_b = mj.MjData(m_b)
    rec_b = AnvilRecorder(m_b, d_b, task="Drawer pull", robot_type="rig_b", key="run_b", fps=25)
    for _ in range(6 * rec_b.steps_per_frame):
        rec_b.step(np.array([0.05, 0.06]), phase="S1/pull")
    npz_b = rec_b.save(tmp_path / "dir_b/scene.npz", passed=True)

    ep_a = load_episodes(npz_a, max_frames=0, render_cameras=True)[0]
    ep_b = load_episodes(npz_b, max_frames=0, render_cameras=True)[0]

    # Framebuffer dimensions must be restored on the caller's model
    assert m_a.vis.global_.offwidth == 160
    assert m_a.vis.global_.offheight == 120

    # Distinct scenes must receive distinct model keys even with identical filenames
    assert ep_a.model_key != ep_b.model_key

    # Support force acting ON the resting block must point upward (+Z)
    settled_contacts = ep_a.contacts[-1]
    assert settled_contacts, "Expected resting contact between floor and block"
    assert all(c["force"][2] > 0.0 for c in settled_contacts)

    res = export_inspection_bundle(tmp_path / "multi_session", [ep_a, ep_b])
    assert len(res["models_exported"]) == 2


def test_failed_episode_anomaly_policy_and_single_frame_force_video(tmp_path: Path) -> None:
    m = mj.MjModel.from_xml_string(SMALL_FB_XML)
    d = mj.MjData(m)

    # Case 1: failed episode WITH explicit retry phase tags -> only tagged frames downweighted
    rec1 = AnvilRecorder(m, d, task="Explicit fault", key="ep_explicit", fps=25)
    for f in range(10):
        ph = "S2/retry_slip" if f >= 8 else "S1/move"
        for _ in range(rec1.steps_per_frame):
            rec1.step(phase=ph)
    p1 = rec1.save(tmp_path / "explicit.npz", passed=False)
    ep1 = load_episodes(p1, max_frames=0, render_cameras=False)[0]
    w1 = [t["bc_loss_weight"] for t in ep1.telemetry]
    assert w1[:8] == [1.0] * 8
    assert w1[8:] == [0.15, 0.15]

    # Case 2: failed episode WITHOUT any anomaly phase tags -> tail fallback from 55%
    d2 = mj.MjData(m)
    rec2 = AnvilRecorder(m, d2, task="Unannotated fail", key="ep_fallback", fps=25)
    for _ in range(10 * rec2.steps_per_frame):
        rec2.step(phase="S1/move")
    p2 = rec2.save(tmp_path / "fallback.npz", passed=False)
    ep2 = load_episodes(p2, max_frames=0, render_cameras=False)[0]
    w2 = [t["bc_loss_weight"] for t in ep2.telemetry]
    assert w2[:5] == [1.0] * 5
    assert w2[5:] == [0.15] * 5

    # Single-frame force video render should not divide by zero and must restore model alpha/offwidth
    mj.mj_forward(m, d2)
    cam = mj.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.02]
    cam.azimuth = 135.0
    cam.elevation = -25.0
    cam.distance = 0.3
    orig_alpha = float(m.geom_rgba[m.geom("block_geom").id, 3])
    out = render_contact_mechanics_reel(
        m,
        [(0.0, "S1/settle", 1.0, d2)],
        target_body_name="test_block",
        ee_geoms=set(),
        closeup_cam=cam,
        wrist_cam_name="wrist_camera",
        out_mp4=tmp_path / "single.mp4",
        out_gif=tmp_path / "single.gif",
    )
    assert Path(out["mp4"]).is_file()
    assert Path(out["gif"]).is_file()
    assert float(m.geom_rgba[m.geom("block_geom").id, 3]) == orig_alpha
    assert m.vis.global_.offwidth == 160


def test_import_simanvil_does_not_mutate_hf_hub_offline() -> None:
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import os, simanvil; print(os.environ.get('HF_HUB_OFFLINE', 'UNSET'))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "UNSET"


def test_partial_frame_causal_replay_and_duplicate_run_key_dedup(tmp_path: Path) -> None:
    model = mj.MjModel.from_xml_string(DRAWER_XML)
    data = mj.MjData(model)
    rec = AnvilRecorder(model, data, task="Partial step count", key="dup/key", fps=25)

    # Record 5 complete frames + 7 trailing substeps (not a multiple of steps_per_frame=20)
    for _ in range(5 * rec.steps_per_frame + 7):
        rec.step(np.array([0.08, 0.09], dtype=np.float64), phase="S1/pull")

    p1 = rec.save(tmp_path / "r1/episode.npz", passed=True)
    p2 = rec.save(tmp_path / "r2/episode.npz", passed=True)

    ep1 = load_episodes(p1, max_frames=0, render_cameras=True)[0]
    ep2 = load_episodes(p2, max_frames=0, render_cameras=True)[0]
    assert ep1.audit["causal_replay_max_qpos_err"] < 1e-10

    session_dir = tmp_path / "dedup_session"
    res = export_inspection_bundle(session_dir, [ep1, ep2])
    assert list(res["runs_exported"].keys()) == ["dup_key", "dup_key_2"]
    assert (session_dir / "videos/dup_key_overview.mp4").is_file()
    assert (session_dir / "videos/dup_key_2_overview.mp4").is_file()
