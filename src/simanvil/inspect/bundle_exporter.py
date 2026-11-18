"""Static HTML Session Viewer Bundle Exporter (`simanvil.inspect`)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Sequence

import av
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from simanvil.trace.adapters import compute_workspace_center
from simanvil.trace.schema import Episode


def _mj_quat_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    w, x, y, z = quat_wxyz
    norm = np.hypot(np.hypot(w, x), np.hypot(y, z))
    if norm > 1e-9:
        mat[:3, :3] = Rotation.from_quat([x / norm, y / norm, z / norm, w / norm]).as_matrix()
    mat[:3, 3] = pos
    return mat


def _geom_to_trimesh(model: mj.MjModel, gid: int, rgba_override: list[int] | None = None) -> trimesh.Trimesh | None:
    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if gtype == int(mj.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        tm = trimesh.Trimesh(
            vertices=model.mesh_vert[va : va + vn].copy(),
            faces=model.mesh_face[fa : fa + fn].copy(),
            process=False,
        )
    elif gtype == int(mj.mjtGeom.mjGEOM_BOX):
        tm = trimesh.creation.box(extents=size[:3] * 2.0)
    elif gtype == int(mj.mjtGeom.mjGEOM_SPHERE):
        tm = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif gtype == int(mj.mjtGeom.mjGEOM_CYLINDER):
        tm = trimesh.creation.cylinder(radius=float(size[0]), height=float(size[1]) * 2.0, sections=24)
    elif gtype == int(mj.mjtGeom.mjGEOM_CAPSULE):
        tm = trimesh.creation.capsule(radius=float(size[0]), height=float(size[1]) * 2.0, count=[12, 16])
    else:
        return None

    tm.apply_transform(_mj_quat_to_matrix(model.geom_pos[gid], model.geom_quat[gid]))

    if rgba_override is not None:
        rgba_f = [c / 255.0 for c in rgba_override[:3]] + [1.0]
    else:
        mat_id = int(model.geom_matid[gid])
        raw_rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[gid]
        rgba_f = [
            float(np.clip(raw_rgba[0], 0.0, 1.0)),
            float(np.clip(raw_rgba[1], 0.0, 1.0)),
            float(np.clip(raw_rgba[2], 0.0, 1.0)),
            1.0,
        ]
    tm.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=f"geom_mat_{gid}",
            baseColorFactor=rgba_f,
            metallicFactor=0.20,
            roughnessFactor=0.42,
        )
    )
    return tm


def _encode_intra_mp4(frames_rgb: list[np.ndarray], out_mp4: Path, fps: int = 25) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    container = av.open(str(out_mp4), mode="w")
    stream = container.add_stream(
        "libx264",
        rate=max(1, int(fps)),
        options={"g": "1", "crf": "19", "preset": "fast"},
    )
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for frame_arr in frames_rgb:
        vf = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for pkt in stream.encode(vf):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def _extract_phase_spans(telemetry: list[dict]) -> list[dict]:
    if not telemetry:
        return []
    spans: list[dict] = []
    cur = str(telemetry[0]["g2_stage"])
    start = 0
    for i, t in enumerate(telemetry):
        if str(t["g2_stage"]) != cur:
            spans.append({"stage": cur, "start": start, "end": i - 1})
            cur = str(t["g2_stage"])
            start = i
    spans.append({"stage": cur, "start": start, "end": len(telemetry) - 1})
    return spans


def _detect_floor_z_and_camera_presets(model: mj.MjModel, xpos0: np.ndarray) -> tuple[float, dict]:
    floor_z = -0.14
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) == int(mj.mjtGeom.mjGEOM_PLANE):
            bid = int(model.geom_bodyid[gid])
            floor_z = float(xpos0[bid, 2] + model.geom_pos[gid, 2])
            break

    fx, fy, fz = compute_workspace_center(model, xpos0).tolist()
    body_span = float(np.ptp(xpos0[1:, 2])) if len(xpos0) > 1 else 0.5
    dist_scale = 1.25 if body_span > 0.75 else 1.0

    presets = {
        "hero": {
            "pos": [round(fx + 0.52 * dist_scale, 3), round(fy - 0.58 * dist_scale, 3), round(fz + 0.44 * dist_scale, 3)],
            "target": [round(fx - 0.03, 3), round(fy, 3), round(fz + 0.03, 3)],
        },
        "tray": {
            "pos": [round(fx + 0.25, 3), round(fy - 0.22, 3), round(fz + 0.20, 3)],
            "target": [round(fx, 3), round(fy, 3), round(fz, 3)],
        },
        "side": {
            "pos": [round(fx, 3), round(fy - 0.54, 3), round(fz + 0.06, 3)],
            "target": [round(fx, 3), round(fy, 3), round(fz + 0.01, 3)],
        },
        "top": {
            "pos": [round(fx, 3), round(fy - 0.001, 3), round(fz + 0.74, 3)],
            "target": [round(fx, 3), round(fy, 3), round(fz, 3)],
        },
    }
    return round(floor_z, 4), presets


def _export_model_geometries(out_root: Path, model_key: str, ep: Episode) -> dict:
    meshes_dir = out_root / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    model = ep.model

    forged_summary: dict[str, dict] = {}
    for bname, (adir, rep) in ep.forged_assets.items():
        shutil.copy2(adir / "visual.glb", meshes_dir / f"{model_key}_{bname}_visual.glb")
        shutil.copy2(adir / "hulls_debug.glb", meshes_dir / f"{model_key}_{bname}_coacd.glb")
        shutil.copy2(adir / "naive_shrinkwrap.glb", meshes_dir / f"{model_key}_{bname}_shrinkwrap.glb")
        forged_summary[bname] = {
            "hull_count": rep["hull_count"],
            "z_up_delta_m": rep["z_up_alignment_delta_m"],
            "shrinkwrap_vol_cm3": rep["naive_shrinkwrap_volume_cm3"],
            "coacd_vol_cm3": rep["coacd_total_volume_cm3"],
            "freed_void_cm3": rep["occluded_void_volume_cm3"],
            "cavity_gain_mm": rep["smoke_test"].get("cavity_clearance_gain_mm", 0.0),
            "mass_kg": rep["smoke_test"]["computed_mass_kg"],
        }

    bodies_manifest: list[dict] = []
    for bid in range(model.nbody):
        bname = model.body(bid).name or f"body_{bid}"
        if bname in ep.forged_assets:
            bodies_manifest.append(
                {
                    "id": bid,
                    "name": bname,
                    "is_compiled_asset": True,
                    "is_focus_part": True,
                    "visual_glb": f"meshes/{model_key}_{bname}_visual.glb",
                    "coacd_glb": f"meshes/{model_key}_{bname}_coacd.glb",
                    "shrinkwrap_glb": f"meshes/{model_key}_{bname}_shrinkwrap.glb",
                }
            )
            continue

        jadr, jnum = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
        has_freejoint = any(int(model.jnt_type[jadr + k]) == int(mj.mjtJoint.mjJNT_FREE) for k in range(jnum))

        vis_scene = trimesh.Scene()
        col_scene = trimesh.Scene()
        vis_count, col_count = 0, 0

        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            gtype = int(model.geom_type[gid])
            if gtype == int(mj.mjtGeom.mjGEOM_PLANE):
                continue
            gname = (model.geom(gid).name or f"geom_{gid}").lower()
            ggroup = int(model.geom_group[gid])
            is_pad = "pad" in gname

            if ggroup <= 2 or is_pad:
                tm = _geom_to_trimesh(model, gid)
                if tm is not None:
                    vis_scene.add_geometry(tm, node_name=f"v_{gid}", geom_name=f"v_{gid}")
                    vis_count += 1

            is_colliding = (ggroup == 3) or (int(model.geom_contype[gid]) > 0 or int(model.geom_conaffinity[gid]) > 0)
            if is_colliding and "table" not in bname.lower():
                tm_c = _geom_to_trimesh(model, gid, rgba_override=[56, 189, 248, 255])
                if tm_c is not None:
                    if gtype == int(mj.mjtGeom.mjGEOM_MESH) and len(tm_c.vertices) >= 4:
                        try:
                            tm_c = tm_c.convex_hull
                        except Exception:
                            pass
                    col_scene.add_geometry(tm_c, node_name=f"c_{gid}", geom_name=f"c_{gid}")
                    col_count += 1

        vis_rel = f"meshes/{model_key}_b{bid:02d}_vis.glb" if vis_count > 0 else None
        col_rel = f"meshes/{model_key}_b{bid:02d}_col.glb" if col_count > 0 else None
        if vis_rel:
            (out_root / vis_rel).write_bytes(vis_scene.export(file_type="glb"))
        if col_rel:
            (out_root / col_rel).write_bytes(col_scene.export(file_type="glb"))

        bodies_manifest.append(
            {
                "id": bid,
                "name": bname,
                "is_compiled_asset": False,
                "is_focus_part": bool(has_freejoint),
                "visual_glb": vis_rel,
                "coacd_glb": col_rel,
                "shrinkwrap_glb": col_rel,
            }
        )

    floor_z, cam_presets = _detect_floor_z_and_camera_presets(model, ep.xpos[0])
    if forged_summary:
        ts = next(iter(forged_summary.values()))
        note = (
            "<strong>SimAnvil Forge:</strong> "
            + " · ".join(
                f"{k}: {v['hull_count']} hulls (freed {v['freed_void_cm3']} cm³)"
                for k, v in forged_summary.items()
            )
            + f" · Z-up Δ={ts['z_up_delta_m']} m"
        )
    else:
        n_col = sum(1 for b in bodies_manifest if b["coacd_glb"])
        note = (
            f"<strong>SimAnvil Inspector:</strong> Embodiment <code>{ep.robot_type}</code> · "
            f"{len(bodies_manifest)} rigid bodies ({n_col} collision-hull groups) · Z-up [0,0,1]"
        )

    return {
        "model_key": model_key,
        "robot_type": ep.robot_type,
        "floor_z": floor_z,
        "camera_presets": cam_presets,
        "compiler_note": note,
        "compiled_assets": forged_summary,
        "bodies": bodies_manifest,
    }


def export_inspection_bundle(
    output_dir: Path,
    episodes: Sequence[Episode],
    *,
    title: str | None = None,
) -> dict:
    """Write a complete, zero-backend static HTML Session Viewer bundle to `output_dir`."""
    if not episodes:
        raise ValueError("At least one Episode is required to build an inspection bundle.")

    out = Path(output_dir).resolve()
    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    models_payload: dict[str, dict] = {}
    runs_payload: dict[str, dict] = {}

    for ep in episodes:
        if ep.model_key not in models_payload:
            models_payload[ep.model_key] = _export_model_geometries(out, ep.model_key, ep)

        ov_rel = f"videos/{ep.key}_overview.mp4"
        wr_rel = f"videos/{ep.key}_wrist.mp4"
        _encode_intra_mp4(ep.overview_rgb, out / ov_rel, fps=ep.fps)
        _encode_intra_mp4(ep.wrist_rgb, out / wr_rel, fps=ep.fps)

        runs_payload[ep.key] = {
            "key": ep.key,
            "label": ep.label,
            "model_key": ep.model_key,
            "robot_type": ep.robot_type,
            "mode": ep.audit.get("mode", ep.key),
            "use_shrinkwrap_mesh": bool(ep.use_shrinkwrap_mesh),
            "g1_instruction": ep.task,
            "fps": ep.fps,
            "frame_count": len(ep.times),
            "duration_s": ep.times[-1] if ep.times else 0.0,
            "audit": ep.audit,
            "series_defs": ep.series_defs,
            "camera_labels": {"overview": ep.overview_cam_label, "wrist": ep.wrist_cam_label},
            "phases": _extract_phase_spans(ep.telemetry),
            "times": ep.times,
            "xpos": np.round(ep.xpos, 5).tolist(),
            "xquat": np.round(ep.xquat, 5).tolist(),
            "contacts": ep.contacts,
            "telemetry": ep.telemetry,
            "videos": {"overview": ov_rel, "wrist": wr_rel},
        }

    session_title = title or (
        episodes[0].task if len(episodes) == 1 else f"SimAnvil Inspection · {len(episodes)} Runs"
    )
    session_payload = {
        "schema": "simanvil.session.v0.1",
        "title": session_title,
        "g1_instruction": episodes[0].task,
        "coordinate_convention": {
            "world_up": [0, 0, 1],
            "mujoco_quat_order": "[w, x, y, z]",
            "threejs_quat_order": "[x, y, z, w]",
        },
        "models": models_payload,
        "run_order": [ep.key for ep in episodes],
        "runs": runs_payload,
    }
    (out / "session_data.json").write_text(json.dumps(session_payload, separators=(",", ":")), encoding="utf-8")

    web_src = Path(__file__).with_name("web")
    for fname in ("index.html", "viewer.css", "viewer.js"):
        shutil.copy2(web_src / fname, out / fname)
    vendor_dst = out / "vendor"
    vendor_dst.mkdir(parents=True, exist_ok=True)
    for vfile in (web_src / "vendor").glob("*"):
        if vfile.is_file():
            shutil.copy2(vfile, vendor_dst / vfile.name)

    return {
        "session_dir": str(out),
        "index_html": str(out / "index.html"),
        "models_exported": list(models_payload.keys()),
        "runs_exported": {ep.key: ep.audit for ep in episodes},
    }
