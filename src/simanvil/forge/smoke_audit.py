"""200-step MuJoCo free-fall stability & concave cavity clearance smoke tests (`simanvil.forge`)."""

from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np


def run_physics_smoke_audit(
    asset_name: str,
    collision_obj: Path,
    hull_paths: list[Path],
    bounds_min: list[float],
    bounds_max: list[float],
    material_spec: dict[str, object],
    probe_xy: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Drop-tests the CoACD asset in MuJoCo and quantifies cavity clearance vs naive single hull."""

    def build_drop_model(mesh_files: list[Path], probe: bool = False) -> mj.MjModel:
        spec = mj.MjSpec()
        spec.option.timestep = 0.002
        spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
        spec.worldbody.add_geom(
            name="ground",
            type=mj.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            size=[1.0, 1.0, 0.01],
            friction=[0.9, 0.01, 0.001],
        )
        z_spawn = -float(bounds_min[2]) + (0.004 if not probe else 0.0005)
        b = spec.worldbody.add_body(name=asset_name, pos=[0.0, 0.0, z_spawn])
        if not probe:
            b.add_freejoint()
        for idx, mf in enumerate(mesh_files):
            mname = f"m_{idx:02d}"
            ms = spec.add_mesh(name=mname)
            ms.file = str(mf)
            b.add_geom(
                name=f"col_{idx:02d}",
                type=mj.mjtGeom.mjGEOM_MESH,
                meshname=mname,
                group=3,
                condim=4,
                density=float(material_spec["density"]),
                friction=list(material_spec["friction"]),
            )
        if probe:
            z_top = -float(bounds_min[2]) + float(bounds_max[2])
            pb = spec.worldbody.add_body(name="cavity_probe", pos=[probe_xy[0], probe_xy[1], z_top + 0.025])
            pb.add_freejoint()
            pb.add_geom(
                name="probe_sphere",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.007, 0.0, 0.0],
                density=1200.0,
            )
        return spec.compile()

    m_drop = build_drop_model(hull_paths, probe=False)
    d_drop = mj.MjData(m_drop)
    mj.mj_forward(m_drop, d_drop)
    total_mass_kg = float(m_drop.body_subtreemass[m_drop.body(asset_name).id])
    peak_qvel = 0.0
    has_nan = False
    for _ in range(200):
        mj.mj_step(m_drop, d_drop)
        if np.isnan(d_drop.qpos).any() or np.isnan(d_drop.qvel).any():
            has_nan = True
            break
        peak_qvel = max(peak_qvel, float(np.max(np.abs(d_drop.qvel))))
    settled_qvel = float(np.max(np.abs(d_drop.qvel))) if not has_nan else float("inf")

    m_coacd_probe = build_drop_model(hull_paths, probe=True)
    d_coacd_probe = mj.MjData(m_coacd_probe)
    for _ in range(220):
        mj.mj_step(m_coacd_probe, d_coacd_probe)
    z_probe_coacd = float(d_coacd_probe.body("cavity_probe").xpos[2])

    m_naive_probe = build_drop_model([collision_obj], probe=True)
    d_naive_probe = mj.MjData(m_naive_probe)
    for _ in range(220):
        mj.mj_step(m_naive_probe, d_naive_probe)
    z_probe_naive = float(d_naive_probe.body("cavity_probe").xpos[2])

    passed = (not has_nan) and (peak_qvel < 25.0) and (settled_qvel < 0.25)
    return {
        "passed": passed,
        "steps_tested": 200,
        "has_nan": has_nan,
        "peak_qvel": round(peak_qvel, 5),
        "settled_qvel": round(settled_qvel, 5),
        "computed_mass_kg": round(total_mass_kg, 4),
        "probe_xy_m": [probe_xy[0], probe_xy[1]],
        "probe_rest_z_coacd_m": round(z_probe_coacd, 5),
        "probe_rest_z_naive_m": round(z_probe_naive, 5),
        "cavity_clearance_gain_mm": round((z_probe_naive - z_probe_coacd) * 1000.0, 2),
    }
