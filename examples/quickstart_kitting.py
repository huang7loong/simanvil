"""Self-contained SimAnvil Quickstart (zero external robot assets required).

1. Forges a dual-pocket concave `kitting_tray` via Headless Blender + CoACD (`group="2"`/`"3"`/`"4"`).
2. Simulates a 4-DoF Cartesian parallel-jaw gripper picking & seating a precision block under:
   - `coacd`      (24 convex hulls -> seats cleanly on 8 mm pocket floor, PASS)
   - `shrinkwrap` (naive single convex hull -> blocked by rim air-wall, 0.15x BC downweight)
3. Exports a zero-backend HTML Session Viewer bundle to `outputs/quickstart_session`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco as mj
import numpy as np

from simanvil import (
    AnvilRecorder,
    ForgeConfig,
    attach_forged_asset,
    export_inspection_bundle,
    forge_asset,
    load_episodes,
)


def _build_gantry_scene(tray_dir: Path, mode: str) -> tuple[mj.MjModel, dict]:
    report = json.loads((tray_dir / "forge_report.json").read_text(encoding="utf-8"))
    bmin = report["blender"]["bounds_min_m"]
    bmax = report["blender"]["bounds_max_m"]

    spec = mj.MjSpec()
    spec.option.timestep = 0.002
    spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC

    spec.worldbody.add_light(pos=[0.3, -0.4, 1.1], dir=[-0.1, 0.25, -1.0], diffuse=[0.85, 0.85, 0.88])
    spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.10], size=[1.5, 1.5, 0.02], contype=0, conaffinity=0)

    table = spec.worldbody.add_body(name="workspace_table", pos=[0.0, 0.0, 0.0])
    table.add_geom(name="table_top", type=mj.mjtGeom.mjGEOM_BOX, size=[0.28, 0.28, 0.015], rgba=[0.22, 0.25, 0.30, 1.0])

    spec.worldbody.add_camera(name="overview_cam", pos=[0.42, -0.38, 0.34], xyaxes=[0.66, 0.75, 0, -0.38, 0.34, 0.86], fovy=42)

    tray_z = 0.015 - float(bmin[2])
    attach_forged_asset(spec, tray_dir, body_name="kitting_tray", pos=(0.06, 0.04, tray_z), dynamic=False, mode=mode)

    cube = spec.worldbody.add_body(name="datum_cube", pos=[-0.12, -0.08, 0.015 + 0.0165 + 0.0005])
    cube.add_freejoint(name="datum_cube_free")
    cube.add_geom(
        name="datum_cube_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.0165, 0.0165, 0.0165],
        mass=0.05,
        condim=4,
        friction=[0.95, 0.01, 0.0003],
        rgba=[0.88, 0.20, 0.22, 1.0],
    )

    # Lightweight 3-axis Cartesian wrist + symmetric parallel-jaw pads + eye-in-hand camera
    carriage = spec.worldbody.add_body(name="gripper_carriage", pos=[-0.12, -0.08, 0.16])
    carriage.add_joint(name="gx", type=mj.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.25, 0.25], damping=18.0)
    carriage.add_joint(name="gy", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.25, 0.25], damping=18.0)
    carriage.add_joint(name="gz", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.14, 0.12], damping=22.0)
    carriage.add_geom(name="palm", type=mj.mjtGeom.mjGEOM_BOX, size=[0.022, 0.042, 0.012], rgba=[0.30, 0.34, 0.40, 1.0])
    carriage.add_camera(name="wrist_camera", pos=[0.045, 0.0, -0.01], xyaxes=[0, 1, 0, -0.35, 0, 0.94], fovy=58)

    for side, sgn in (("left", -1.0), ("right", 1.0)):
        f = carriage.add_body(name=f"{side}_finger", pos=[0.0, sgn * 0.032, -0.036])
        f.add_joint(name=f"jaw_{side}", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, -sgn, 0], range=[0.0, 0.024], damping=6.0)
        f.add_geom(
            name=f"{side}_pad",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.004, 0.024],
            mass=0.04,
            condim=4,
            friction=[1.1, 0.01, 0.0005],
            rgba=[0.22, 0.72, 0.92, 1.0],
        )

    for jname, kp, lo, hi in [
        ("gx", 380.0, -0.25, 0.25),
        ("gy", 380.0, -0.25, 0.25),
        ("gz", 450.0, -0.14, 0.12),
        ("jaw_left", 160.0, 0.0, 0.024),
        ("jaw_right", 160.0, 0.0, 0.024),
    ]:
        act = spec.add_actuator(name=f"act_{jname}", target=jname, trntype=mj.mjtTrn.mjTRN_JOINT)
        act.dyntype = mj.mjtDyn.mjDYN_NONE
        act.gaintype = mj.mjtGain.mjGAIN_FIXED
        act.biastype = mj.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.ctrlrange[:] = [lo, hi]

    return spec.compile(), {"tray_rim_z": 0.015 + 0.038, "bmax_z": float(bmax[2])}


def run_quickstart(out_root: Path = Path("outputs/quickstart")) -> dict:
    out_root = out_root.resolve()
    tray_dir = out_root / "assets/kitting_tray"
    if not (tray_dir / "forge_report.json").is_file():
        forge_asset(
            ForgeConfig(
                asset_name="kitting_tray",
                preset="kitting_tray",
                material="aluminum_6061",
                threshold=0.02,
                max_convex_hull=24,
                preprocess_resolution=65,
                output_dir=tray_dir,
            )
        )

    saved_traces: list[Path] = []
    for mode in ("coacd", "shrinkwrap"):
        model, info = _build_gantry_scene(tray_dir, mode=mode)
        data = mj.MjData(model)
        mj.mj_forward(model, data)

        rec = AnvilRecorder(
            model,
            data,
            task="Pick red datum block and seat into Left Pocket of concave kitting tray",
            robot_type="cartesian_parallel_jaw",
            key=mode,
            label=f"Run · {'CoACD 24 Hulls (PASS)' if mode == 'coacd' else 'Shrink-Wrap 1 Hull (BLOCKED · 0.15×)'}",
            fps=25,
            forged_assets={"kitting_tray": tray_dir},
        )

        # Carriage home is (-0.12, -0.08, 0.16); pocket L center is (0.06, 0.04 - 0.055) = (0.06, -0.015)
        waypoints = [
            ([0.0, 0.0, -0.105, 0.0, 0.0], 18, "S1_PICK/descend"),
            ([0.0, 0.0, -0.105, 0.020, 0.020], 14, "S1_PICK/close_jaws"),
            ([0.0, 0.0, -0.010, 0.020, 0.020], 16, "S1_PICK/lift"),
            ([0.18, 0.065, -0.010, 0.020, 0.020], 22, "S2_INSERT/align_pocket"),
            ([0.18, 0.065, -0.096, 0.020, 0.020], 20, "S2_INSERT/seat_into_pocket"),
            ([0.18, 0.065, -0.096, 0.0, 0.0], 12, "S2_INSERT/release"),
            ([0.18, 0.065, 0.01, 0.0, 0.0], 16, "S3_VERIFY/retract"),
        ]
        prev_u = np.zeros(5, dtype=np.float64)
        for target_u, n_frames, phase in waypoints:
            goal_u = np.asarray(target_u, dtype=np.float64)
            for f in range(n_frames):
                alpha = (f + 1) / float(n_frames)
                u = (1.0 - alpha) * prev_u + alpha * goal_u
                for _ in range(rec.steps_per_frame):
                    rec.step(u, phase=phase)
            prev_u = goal_u

        cube_bottom_z = float(data.body("datum_cube").xpos[2]) - 0.0165
        seat_depth_mm = (info["tray_rim_z"] - cube_bottom_z) * 1000.0
        passed = seat_depth_mm >= 24.0
        verdict = (
            f"PASS: Block seated {seat_depth_mm:.1f} mm below tray rim inside pocket."
            if passed
            else f"BLOCKED (SHRINK-WRAP): Block stranded at {seat_depth_mm:.1f} mm on rim air-wall; tail downweighted to 0.15×."
        )
        npz_out = rec.save(
            out_root / f"runs/{mode}_run.npz",
            passed=passed,
            verdict=verdict,
            use_shrinkwrap_mesh=(mode == "shrinkwrap"),
        )
        saved_traces.append(npz_out)

    episodes = []
    for p in saved_traces:
        episodes.extend(load_episodes(p, max_frames=0, render_cameras=True))
    return export_inspection_bundle(out_root / "session", episodes, title="SimAnvil Quickstart · CoACD vs Shrink-Wrap")


if __name__ == "__main__":
    res = run_quickstart()
    print(json.dumps(res, indent=2))
