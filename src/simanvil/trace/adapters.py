"""Universal MuJoCo Episode Extractor & Pluggable Adapter Registry (`simanvil.trace`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import mujoco as mj
import numpy as np

from simanvil.assay.quality_filter import compute_episode_telemetry
from simanvil.trace.replay_verify import verify_causal_replay
from simanvil.trace.schema import Episode, INTEGRATION_STATE


CAMERA_H = 270
CAMERA_W = 480

AdapterPredicate = Callable[[Path, Path | None], bool]
AdapterLoader = Callable[[Path, Path | None, int | None, bool], list[Episode]]

_ADAPTER_REGISTRY: list[tuple[str, AdapterPredicate, AdapterLoader]] = []


def register_episode_adapter(
    name: str,
    predicate: AdapterPredicate,
    loader: AdapterLoader,
) -> None:
    """Register an external episode loader (e.g. `.mcap` or domain-specific benchmark folders)."""
    _ADAPTER_REGISTRY.insert(0, (name, predicate, loader))


def compute_workspace_center(model: mj.MjModel, xpos_sample: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for bid in range(1, model.nbody):
        bname = (model.body(bid).name or "").lower()
        if any(k in bname for k in ("block", "cube", "tray", "bracket", "part", "drawer", "workpiece")):
            candidates.append(xpos_sample[bid])
    if candidates:
        return np.median(np.stack(candidates, axis=0), axis=0)
    for bid in range(1, model.nbody):
        if "table" in (model.body(bid).name or "").lower():
            return xpos_sample[bid] + np.array([0.0, 0.0, 0.08])
    return np.mean(xpos_sample[1:], axis=0) if len(xpos_sample) > 1 else np.array([0.45, 0.0, 0.12])


def _discover_ee_and_object_ids(model: mj.MjModel) -> tuple[set[int], list[int]]:
    ee_keywords = ("pad", "finger", "jaw", "gripper", "swift", "support_link", "hand", "tip", "follower")
    ee_geoms: set[int] = set()
    for gid in range(model.ngeom):
        gname = (model.geom(gid).name or "").lower()
        bname = (model.body(int(model.geom_bodyid[gid])).name or "").lower()
        if any(k in gname or k in bname for k in ee_keywords):
            ee_geoms.add(gid)

    obj_bodies: list[int] = []
    for bid in range(1, model.nbody):
        bname = (model.body(bid).name or "").lower()
        if any(skip in bname for skip in ("fr3_", "panda", "ur5", "arm_", "body_link", "base", "table", "floor", "pedestal", "gantry")):
            continue
        jadr = int(model.body_jntadr[bid])
        jnum = int(model.body_jntnum[bid])
        if jnum > 0:
            jtypes = [int(model.jnt_type[jadr + k]) for k in range(jnum)]
            if int(mj.mjtJoint.mjJNT_FREE) in jtypes or any(
                k in bname for k in ("drawer", "door", "lid", "tray", "block", "cube", "bracket", "part", "obj")
            ):
                obj_bodies.append(bid)
    return ee_geoms, obj_bodies


def _select_dual_cameras(
    model: mj.MjModel,
    focus_xyz: np.ndarray,
) -> tuple[str | mj.MjvCamera, str, str | mj.MjvCamera, str]:
    cam_names = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]

    wrist_cam: str | mj.MjvCamera | None = None
    wrist_label = "WRIST / FOCUS CAM"
    for preferred in ("gripper/wrist_camera", "r_wrist_camera", "wrist_camera", "wrist_cam", "gripper_camera"):
        if preferred in cam_names:
            wrist_cam = preferred
            wrist_label = f"WRIST ({preferred.split('/')[-1]})"
            break
    if wrist_cam is None:
        for cn in cam_names:
            if any(k in cn.lower() for k in ("wrist", "hand", "grip", "eye")):
                wrist_cam = cn
                wrist_label = f"WRIST ({cn})"
                break

    overview_cam: str | mj.MjvCamera | None = None
    overview_label = "OVERVIEW CAM"
    for preferred in ("overview_cam", "front_camera", "main_cam"):
        if preferred in cam_names and preferred != wrist_cam:
            overview_cam = preferred
            overview_label = f"OVERVIEW ({preferred})"
            break
    if overview_cam is None:
        for cn in cam_names:
            if cn != wrist_cam and cn != "inspection":
                overview_cam = cn
                overview_label = f"CAMERA ({cn})"
                break

    if overview_cam is None:
        free_ov = mj.MjvCamera()
        free_ov.lookat[:] = focus_xyz
        free_ov.azimuth = 138.0
        free_ov.elevation = -28.0
        free_ov.distance = 0.95
        overview_cam = free_ov
        overview_label = "OVERVIEW (AUTO-ORBIT)"

    if wrist_cam is None:
        free_focus = mj.MjvCamera()
        free_focus.lookat[:] = focus_xyz
        free_focus.azimuth = 48.0
        free_focus.elevation = -20.0
        free_focus.distance = 0.48
        wrist_cam = free_focus
        wrist_label = "WORKSPACE CLOSE-UP"

    return overview_cam, overview_label, wrist_cam, wrist_label


def _extract_state_and_action_spec(model: mj.MjModel) -> tuple[np.ndarray, list[str], list[str]]:
    qpos_indices: list[int] = []
    state_names: list[str] = []
    seen_jnts: set[int] = set()

    for aid in range(model.nu):
        trntype = int(model.actuator_trntype[aid])
        if trntype == int(mj.mjtTrn.mjTRN_JOINT):
            jid = int(model.actuator_trnid[aid, 0])
            if jid not in seen_jnts and int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_FREE):
                seen_jnts.add(jid)
                qpos_indices.append(int(model.jnt_qposadr[jid]))
                state_names.append(model.joint(jid).name or f"joint_{jid}")
        elif trntype == int(mj.mjtTrn.mjTRN_TENDON):
            for cand in ("gripper/left_driver_joint", "left_driver_joint"):
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, cand)
                if jid >= 0 and jid not in seen_jnts:
                    seen_jnts.add(jid)
                    qpos_indices.append(int(model.jnt_qposadr[jid]))
                    state_names.append(cand)

    if not qpos_indices:
        for jid in range(model.njnt):
            if int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_FREE):
                qpos_indices.append(int(model.jnt_qposadr[jid]))
                state_names.append(model.joint(jid).name or f"joint_{jid}")
                if len(qpos_indices) >= 14:
                    break

    action_names = [(model.actuator(aid).name or f"act_{aid}") for aid in range(model.nu)] or state_names.copy()
    return np.asarray(qpos_indices, dtype=np.int64), state_names, action_names


def _compute_indices(total: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames <= 0 or total <= max_frames:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, max_frames).round().astype(np.int64))


def extract_episode_from_callback(
    *,
    key: str,
    label: str,
    model_key: str,
    robot_type: str,
    task: str,
    model: mj.MjModel,
    num_frames: int,
    fps: int,
    apply_frame_fn: Callable[[mj.MjModel, mj.MjData, int], tuple[float, str, np.ndarray | None, np.ndarray | None]],
    episode_passed: bool = True,
    verdict_text: str | None = None,
    force_anomaly_from_frame: int | None = None,
    extra_audit: dict | None = None,
    forged_assets: dict[str, tuple[Path, dict]] | None = None,
    use_shrinkwrap_mesh: bool = False,
    render_cameras: bool = True,
) -> Episode:
    """Universal frame-by-frame forward kinematics, 3D contact-force & dual-camera extractor."""
    data = mj.MjData(model)
    ee_geoms, obj_bodies = _discover_ee_and_object_ids(model)
    obj_set = set(obj_bodies)
    qpos_idx, state_names, action_names = _extract_state_and_action_spec(model)

    apply_frame_fn(model, data, 0)
    mj.mj_forward(model, data)
    focus_xyz = compute_workspace_center(model, data.xpos)
    ov_cam, ov_label, wr_cam, wr_label = _select_dual_cameras(model, focus_xyz)

    renderer = mj.Renderer(model, height=CAMERA_H, width=CAMERA_W) if render_cameras else None
    scene_opt = mj.MjvOption()
    scene_opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]

    times: list[float] = []
    xpos_list: list[np.ndarray] = []
    xquat_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    sub_list: list[np.ndarray] = []
    has_substeps = True
    contacts_list: list[list[dict]] = []
    raw_phases: list[str] = []
    ee_forces: list[float] = []
    env_forces: list[float] = []
    ov_frames: list[np.ndarray] = []
    wr_frames: list[np.ndarray] = []

    init_obj_z = {bid: float(data.xpos[bid, 2]) for bid in obj_bodies}
    wrench = np.zeros(6, dtype=np.float64)
    peak_episode_force = 0.0

    for f_idx in range(num_frames):
        t_sec, raw_phase, act_vec, sub_vec = apply_frame_fn(model, data, f_idx)
        mj.mj_forward(model, data)

        times.append(round(float(t_sec), 4))
        raw_phases.append(raw_phase)
        xpos_list.append(data.xpos.copy().astype(np.float32))
        xquat_list.append(data.xquat.copy().astype(np.float32))

        obs_vec = data.qpos[qpos_idx].astype(np.float32) if len(qpos_idx) else np.zeros(1, dtype=np.float32)
        obs_list.append(obs_vec)
        if act_vec is None:
            act_vec = data.ctrl.copy().astype(np.float32) if model.nu > 0 else obs_vec.copy()
        act_list.append(np.asarray(act_vec, dtype=np.float32))
        if sub_vec is not None:
            sub_list.append(np.asarray(sub_vec, dtype=np.float32).ravel())
        else:
            has_substeps = False

        ee_f = 0.0
        env_f = 0.0
        frame_contacts: list[dict] = []

        for k in range(data.ncon):
            con = data.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in ee_geoms) and (g2 in ee_geoms):
                continue
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            inv_ee = (g1 in ee_geoms) or (g2 in ee_geoms)
            inv_obj = (b1 in obj_set) or (b2 in obj_set)
            if not (inv_ee or inv_obj):
                continue

            mj.mj_contactForce(model, data, k, wrench)
            fnorm = float(np.linalg.norm(wrench[:3]))
            if fnorm < 0.05 or not np.isfinite(fnorm):
                continue

            fnorm_capped = min(fnorm, 250.0)
            f_world = con.frame.reshape(3, 3).T @ (wrench[:3] * (fnorm_capped / max(fnorm, 1e-9)))
            peak_episode_force = max(peak_episode_force, fnorm_capped)

            if inv_ee:
                ee_f += float(min(abs(wrench[0]), 250.0))
            elif inv_obj:
                env_f += fnorm_capped

            frame_contacts.append(
                {
                    "pos": [round(float(v), 5) for v in con.pos],
                    "force": [round(float(v), 3) for v in f_world],
                    "mag": round(fnorm_capped, 2),
                    "kind": "gripper" if inv_ee else "part_tray",
                    "pair": f"{model.body(b1).name or b1} ↔ {model.body(b2).name or b2}",
                }
            )

        contacts_list.append(frame_contacts)
        ee_forces.append(ee_f)
        env_forces.append(env_f)

        if renderer is not None:
            renderer.update_scene(data, camera=ov_cam, scene_option=scene_opt)
            ov_frames.append(renderer.render().copy())
            renderer.update_scene(data, camera=wr_cam, scene_option=scene_opt)
            wr_frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()

    xpos_arr = np.stack(xpos_list, axis=0)
    ranked_objs = sorted(obj_bodies, key=lambda bid: float(np.ptp(xpos_arr[:, bid, 2])), reverse=True)[:2]
    palette = ["rgb(248, 113, 113)", "rgb(251, 191, 36)"]
    series_defs = [
        {
            "key": f"obj_{rank}_dz_mm",
            "label": f"{model.body(bid).name} ΔZ",
            "unit": "mm",
            "color": palette[rank % len(palette)],
        }
        for rank, bid in enumerate(ranked_objs)
    ]
    extra_series = {
        f"obj_{rank}_dz_mm": [
            (float(xpos_arr[f_i, bid, 2]) - init_obj_z[bid]) * 1000.0 for f_i in range(num_frames)
        ]
        for rank, bid in enumerate(ranked_objs)
    }

    telemetry_list = compute_episode_telemetry(
        phases=raw_phases,
        ee_forces_n=ee_forces,
        env_forces_n=env_forces,
        extra_series=extra_series,
        force_anomaly_from_frame=force_anomaly_from_frame,
    )

    downweighted = sum(1 for t in telemetry_list if float(t["bc_loss_weight"]) < 1.0)
    default_verdict = (
        f"PASS: Verified {num_frames} frames ({robot_type}); {num_frames - downweighted}/{num_frames} clean 1.00× frames."
        if episode_passed
        else f"ANOMALY / BLOCKED: {downweighted}/{num_frames} frames clamped & downweighted to 0.15× BC loss weight."
    )
    badge = (
        f"AUDIT PASS · {num_frames} FRAMES"
        if episode_passed
        else f"ANOMALY · {downweighted} FRAMES @ 0.15×"
    )
    audit = {
        "mode": key,
        "passed": bool(episode_passed),
        "badge_text": badge,
        "peak_part_tray_force_n": round(peak_episode_force, 2),
        "downweighted_frames": downweighted,
        "total_frames": num_frames,
        "verdict": verdict_text or default_verdict,
    }
    if extra_audit:
        audit.update(extra_audit)

    return Episode(
        key=key,
        label=label,
        model_key=model_key,
        robot_type=robot_type,
        task=task,
        fps=fps,
        model=model,
        times=times,
        xpos=xpos_arr,
        xquat=np.stack(xquat_list, axis=0),
        obs_state=np.stack(obs_list, axis=0),
        state_names=state_names,
        actions=np.stack(act_list, axis=0),
        action_names=action_names,
        action_substeps=np.stack(sub_list, axis=0) if (has_substeps and sub_list) else None,
        contacts=contacts_list,
        telemetry=telemetry_list,
        series_defs=series_defs,
        overview_rgb=ov_frames,
        wrist_rgb=wr_frames,
        overview_cam_label=ov_label,
        wrist_cam_label=wr_label,
        audit=audit,
        forged_assets=forged_assets or {},
        use_shrinkwrap_mesh=use_shrinkwrap_mesh,
    )


def _load_mujoco_model(npz_path: Path, explicit_model: Path | None) -> tuple[mj.MjModel, str]:
    if explicit_model is not None:
        if not explicit_model.is_file():
            raise FileNotFoundError(f"Model file not found: {explicit_model}")
        p = explicit_model.resolve()
        mdl = mj.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mj.MjModel.from_xml_path(str(p))
        return mdl, p.stem

    for ext in (".mjb", ".xml"):
        same_stem = npz_path.with_suffix(ext)
        if same_stem.is_file():
            mdl = (
                mj.MjModel.from_binary_path(str(same_stem))
                if ext == ".mjb"
                else mj.MjModel.from_xml_path(str(same_stem))
            )
            return mdl, same_stem.stem

    for pat, is_bin in (("*.mjb", True), ("*.xml", False)):
        siblings = sorted(npz_path.parent.glob(pat))
        if siblings:
            s = siblings[0]
            return (mj.MjModel.from_binary_path(str(s)) if is_bin else mj.MjModel.from_xml_path(str(s))), s.stem

    raise ValueError(
        f"Cannot infer MuJoCo model for `{npz_path}`. Pass `--model path/to/scene.xml` or save a sibling `.xml`/`.mjb`."
    )


def _load_standard_npz(
    target: Path,
    model_xml: Path | None,
    max_frames: int | None,
    render_cameras: bool,
) -> list[Episode]:
    if target.is_dir():
        npzs = sorted(target.glob("*.npz"))
        if not npzs:
            raise ValueError(f"No `.npz` files found in {target}")
        out_eps: list[Episode] = []
        for p in npzs:
            out_eps.extend(_load_standard_npz(p, model_xml, max_frames, render_cameras))
        return out_eps

    model, model_stem = _load_mujoco_model(target, model_xml)

    with np.load(target, allow_pickle=False) as z:
        files = set(z.files)
        fps = int(z["fps"]) if "fps" in files else 25
        task = str(z["task"]) if "task" in files else f"MuJoCo episode `{target.stem}`"
        robot_type = str(z["robot_type"]) if "robot_type" in files else "generic_mujoco"
        phases = [str(p) for p in z["phase"].tolist()] if "phase" in files else None
        controls = z["controls"].copy() if "controls" in files else None
        stride = int(z["steps_per_frame"]) if "steps_per_frame" in files else 1
        meta = json.loads(str(z["anvil_meta"])) if "anvil_meta" in files else {}

        extra_audit: dict[str, object] = {}
        if "state" in files and "final_state" in files and controls is not None and len(controls) == len(z["state"]) * stride:
            rep = verify_causal_replay(model, z["state"][0], controls, z["final_state"])
            extra_audit["causal_replay_max_qpos_err"] = rep["max_abs_qpos_err"]

        if "state" in files:
            states = z["state"].copy()
            idxs = _compute_indices(len(states), max_frames)

            def apply_fn(m: mj.MjModel, d: mj.MjData, k: int):
                oi = int(idxs[k])
                mj.mj_setState(m, d, states[oi], INTEGRATION_STATE)
                ph = phases[oi] if phases is not None else "EXECUTE/step"
                u_slice = controls[oi * stride : (oi + 1) * stride] if controls is not None else None
                act0 = u_slice[0] if (u_slice is not None and len(u_slice)) else None
                if act0 is not None and m.nu == len(act0):
                    d.ctrl[:] = act0
                t_s = float(states[oi, 0] - states[0, 0])
                return t_s, ph, act0, u_slice if (u_slice is not None and len(idxs) == len(states)) else None

            n_total = len(idxs)
        elif "qpos" in files:
            qpos_arr = z["qpos"].copy()
            qvel_arr = z["qvel"].copy() if "qvel" in files else None
            times_arr = z["time"].copy() if "time" in files else None
            idxs = _compute_indices(len(qpos_arr), max_frames)

            def apply_fn(m: mj.MjModel, d: mj.MjData, k: int):
                oi = int(idxs[k])
                nq_c = min(m.nq, qpos_arr.shape[1])
                d.qpos[:nq_c] = qpos_arr[oi, :nq_c]
                if qvel_arr is not None:
                    nv_c = min(m.nv, qvel_arr.shape[1])
                    d.qvel[:nv_c] = qvel_arr[oi, :nv_c]
                ph = phases[oi] if phases is not None else "EXECUTE/step"
                u_slice = controls[oi * stride : (oi + 1) * stride] if controls is not None else None
                act0 = u_slice[0] if (u_slice is not None and len(u_slice)) else None
                if act0 is not None and m.nu == len(act0):
                    d.ctrl[:] = act0
                t_s = float(times_arr[oi]) if times_arr is not None else float(oi / fps)
                return t_s, ph, act0, u_slice if (u_slice is not None and len(idxs) == len(qpos_arr)) else None

            n_total = len(idxs)
        else:
            raise ValueError(f"Unsupported `.npz` keys in {target}: {sorted(files)}")

    forged_map: dict[str, tuple[Path, dict]] = {}
    for bname, adir_str in meta.get("forged_assets", {}).items():
        adir = Path(adir_str)
        rep_path = adir / "forge_report.json"
        if rep_path.is_file():
            forged_map[bname] = (adir, json.loads(rep_path.read_text(encoding="utf-8")))

    passed_flag = bool(meta.get("passed", True))
    ep_key = str(meta.get("key") or target.stem)
    ep_label = str(meta.get("label") or f"{robot_type} · {target.stem}")
    verdict = str(meta.get("verdict") or "") or None
    anom_start = int(n_total * 0.55) if not passed_flag else None

    return [
        extract_episode_from_callback(
            key=ep_key,
            label=ep_label,
            model_key=f"mdl_{model_stem}" if not forged_map else "forged_scene",
            robot_type=robot_type,
            task=task,
            model=model,
            num_frames=n_total,
            fps=fps,
            apply_frame_fn=apply_fn,
            episode_passed=passed_flag,
            verdict_text=verdict,
            force_anomaly_from_frame=anom_start,
            extra_audit=extra_audit,
            forged_assets=forged_map,
            use_shrinkwrap_mesh=bool(meta.get("use_shrinkwrap_mesh", False)),
            render_cameras=render_cameras,
        )
    ]


def load_episodes(
    target: Path,
    *,
    model_xml: Path | None = None,
    max_frames: int | None = 220,
    render_cameras: bool = True,
) -> list[Episode]:
    """Dispatch `target` through registered custom adapters or the built-in `.npz` loader."""
    resolved = Path(target).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Episode target does not exist: {resolved}")

    for _, pred, loader in _ADAPTER_REGISTRY:
        if pred(resolved, model_xml):
            return loader(resolved, model_xml, max_frames, render_cameras)

    return _load_standard_npz(resolved, model_xml, max_frames, render_cameras)
