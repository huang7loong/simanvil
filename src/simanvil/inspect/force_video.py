"""Native MuJoCo 3D Contact-Force Vector Video & GIF Renderer (`simanvil.inspect.force_video`).

Injects analytical 3D `mjGEOM_ARROW` force vectors and contact-point spheres directly into
`mujoco.Renderer.scene` (`mjv_initGeom` + `mjv_connector`), paired with synchronized wrist-camera
feeds and normal/tangential force telemetry plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


def _load_font(size: int, bold: bool = False):
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def inject_3d_contact_arrows(
    model: mj.MjModel,
    data: mj.MjData,
    scene: mj.MjvScene,
    *,
    ee_geoms: set[int],
    obj_bodies: set[int],
    force_scale_m_per_n: float = 0.0042,
    max_arrow_len_m: float = 0.14,
    min_force_n: float = 0.15,
) -> dict:
    """Append 3D force vector arrows (`mjGEOM_ARROW`) + contact spheres into `scene`."""
    wrench = np.zeros(6, dtype=np.float64)
    ee_world_sum = np.zeros(3, dtype=np.float64)
    support_world_sum = np.zeros(3, dtype=np.float64)
    ee_norm_mag = 0.0
    ee_shear_mag = 0.0
    support_mag = 0.0
    drawn = 0

    ident_mat = np.eye(3, dtype=np.float64).ravel()

    for k in range(data.ncon):
        if scene.ngeom + 2 >= scene.maxgeom:
            break
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in ee_geoms) and (g2 in ee_geoms):
            continue
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        inv_ee = (g1 in ee_geoms) or (g2 in ee_geoms)
        inv_obj = (b1 in obj_bodies) or (b2 in obj_bodies)
        if not (inv_ee or inv_obj):
            continue

        mj.mj_contactForce(model, data, k, wrench)
        fnorm = float(np.linalg.norm(wrench[:3]))
        if fnorm < min_force_n or not np.isfinite(fnorm):
            continue

        # In MuJoCo, wrench[:3] is force exerted by geom1 on geom2 in contact frame.
        # Ensure arrow represents force acting ON the manipulated object body!
        sign = 1.0 if (b2 in obj_bodies) else -1.0
        f_world = sign * (con.frame.reshape(3, 3).T @ wrench[:3])

        if inv_ee:
            ee_world_sum += f_world
            ee_norm_mag += abs(float(f_world[2]))
            ee_shear_mag += float(np.hypot(f_world[0], f_world[1]))
            rgba = np.array([0.22, 0.78, 0.98, 0.96], dtype=np.float32)  # Cyan: finger -> object
        else:
            support_world_sum += f_world
            support_mag += fnorm
            rgba = np.array([0.98, 0.74, 0.15, 0.96], dtype=np.float32)  # Gold: table pivot -> object

        p0 = con.pos.copy()
        arrow_len = min(max_arrow_len_m, 0.018 + force_scale_m_per_n * fnorm)
        u_dir = f_world / max(fnorm, 1e-9)
        p1 = p0 + u_dir * arrow_len

        # Contact point sphere
        g_dot = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            g_dot,
            mj.mjtGeom.mjGEOM_SPHERE,
            np.array([0.0036, 0.0, 0.0], dtype=np.float64),
            p0,
            ident_mat,
            rgba,
        )
        scene.ngeom += 1

        # 3D Force Vector Arrow
        g_arr = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(g_arr, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), ident_mat, rgba)
        mj.mjv_connector(g_arr, mj.mjtGeom.mjGEOM_ARROW, 0.0034, p0, p1)
        scene.ngeom += 1
        drawn += 1

    return {
        "arrows_drawn": drawn,
        "ee_force_world": ee_world_sum,
        "support_force_world": support_world_sum,
        "ee_normal_z_n": ee_norm_mag,
        "ee_shear_xy_n": ee_shear_mag,
        "support_total_n": support_mag,
    }


def render_contact_mechanics_reel(
    model: mj.MjModel,
    step_states: list[tuple[float, str, float, mj.MjData]],
    *,
    target_body_name: str,
    ee_geoms: set[int],
    closeup_cam: mj.MjvCamera,
    wrist_cam_name: str,
    out_mp4: Path,
    out_gif: Path | None = None,
    title: str = "Non-Prehensile Edge-Pivot Tumble · 3D Contact Force Couple",
    fps: int = 25,
) -> dict:
    """Render a side-by-side 3D Force-Vector + Wrist Camera + Shear/Normal Telemetry MP4 & GIF."""
    if not step_states:
        raise ValueError("step_states must contain at least one frame.")

    target_bid = model.body(target_body_name).id
    obj_bodies = {target_bid}

    # Make target object slightly translucent so both top-edge finger arrows & bottom-edge pivot arrows shine through
    orig_alpha: dict[int, float] = {}
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == target_bid and int(model.geom_contype[gid]) > 0:
            orig_alpha[gid] = float(model.geom_rgba[gid, 3])
            model.geom_rgba[gid, 3] = 0.72

    W, H = 1280, 672
    main_w, main_h = 756, 588
    wrist_w, wrist_h = 468, 263

    orig_offw = int(model.vis.global_.offwidth)
    orig_offh = int(model.vis.global_.offheight)
    r_main: mj.Renderer | None = None
    r_wrist: mj.Renderer | None = None

    records: list[dict] = []
    main_imgs: list[np.ndarray] = []
    wrist_imgs: list[np.ndarray] = []

    try:
        model.vis.global_.offwidth = max(orig_offw, main_w)
        model.vis.global_.offheight = max(orig_offh, main_h)

        r_main = mj.Renderer(model, height=main_h, width=main_w)
        r_wrist = mj.Renderer(model, height=wrist_h, width=wrist_w)
        scene_opt = mj.MjvOption()
        scene_opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]

        # Pre-pass to collect exact force & rotation angle series across all rendered frames
        init_rot = Rotation.from_quat(step_states[0][3].xquat[target_bid][[1, 2, 3, 0]])

        for sim_t, phase_lbl, speed_tag, d_frame in step_states:
            r_main.update_scene(d_frame, camera=closeup_cam, scene_option=scene_opt)
            f_stats = inject_3d_contact_arrows(
                model,
                d_frame,
                r_main.scene,
                ee_geoms=ee_geoms,
                obj_bodies=obj_bodies,
            )
            main_imgs.append(r_main.render().copy())

            r_wrist.update_scene(d_frame, camera=wrist_cam_name, scene_option=scene_opt)
            inject_3d_contact_arrows(
                model,
                d_frame,
                r_wrist.scene,
                ee_geoms=ee_geoms,
                obj_bodies=obj_bodies,
                force_scale_m_per_n=0.0032,
            )
            wrist_imgs.append(r_wrist.render().copy())

            cur_rot = Rotation.from_quat(d_frame.xquat[target_bid][[1, 2, 3, 0]])
            rel_deg = float(np.degrees((cur_rot * init_rot.inv()).magnitude()))
            records.append(
                {
                    "t": sim_t,
                    "phase": phase_lbl,
                    "speed_tag": speed_tag,
                    "tilt_deg": rel_deg,
                    **f_stats,
                }
            )
    finally:
        if r_main is not None:
            r_main.close()
        if r_wrist is not None:
            r_wrist.close()
        model.vis.global_.offwidth = orig_offw
        model.vis.global_.offheight = orig_offh
        for gid, a in orig_alpha.items():
            model.geom_rgba[gid, 3] = a

    font_title = _load_font(16, bold=True)
    font_hdr = _load_font(13, bold=True)
    font_mono = _load_font(12, bold=False)
    font_sm = _load_font(11, bold=False)

    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_mp4), mode="w")
    stream = container.add_stream("libx264", rate=fps, options={"crf": "19", "preset": "fast"})
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"

    composed_frames: list[Image.Image] = []
    n_total = len(records)
    max_force = max(35.0, max(max(r["ee_normal_z_n"], r["support_total_n"]) for r in records) * 1.12)

    for i, rec in enumerate(records):
        canvas = Image.new("RGB", (W, H), (11, 15, 20))
        draw = ImageDraw.Draw(canvas)

        # Header Bar
        draw.rectangle([0, 0, W, 44], fill=(17, 23, 32))
        draw.line([0, 44, W, 44], fill=(38, 51, 69), width=1)
        draw.rounded_rectangle([16, 10, 128, 34], radius=4, fill=(18, 44, 62), outline=(56, 189, 248))
        draw.text((25, 14), "SIMANVIL v0.1", fill=(56, 189, 248), font=font_sm)
        draw.text((142, 13), title, fill=(238, 243, 249), font=font_title)

        spd_col = (251, 191, 36) if rec["speed_tag"] < 0.9 else (159, 176, 197)
        spd_txt = f"{rec['speed_tag']:.2g}x SLOW-MO" if rec["speed_tag"] < 0.9 else "1.0x REALTIME"
        draw.text((W - 265, 14), f"{spd_txt} · t={rec['t']:4.2f}s", fill=spd_col, font=font_mono)

        # Left Main 3D Force-Vector Viewport
        lx, ly = 18, 56
        draw.rounded_rectangle([lx - 2, ly - 2, lx + main_w + 2, ly + main_h + 2], radius=7, fill=(17, 23, 32), outline=(38, 51, 69))
        canvas.paste(Image.fromarray(main_imgs[i]), (lx, ly))

        # Overlay Legend & Live 3D Wrench Readout inside Left Viewport
        draw.rounded_rectangle([lx + 12, ly + 12, lx + 470, ly + 66], radius=6, fill=(11, 15, 20), outline=(38, 51, 69))
        ef = rec["ee_force_world"]
        sf = rec["support_force_world"]
        draw.ellipse([lx + 22, ly + 22, lx + 30, ly + 30], fill=(56, 189, 248))
        draw.text(
            (lx + 36, ly + 18),
            f"Finger -> Block : [{ef[0]:+5.1f}, {ef[1]:+4.1f}, {ef[2]:+5.1f}] N",
            fill=(56, 189, 248),
            font=font_mono,
        )
        draw.ellipse([lx + 22, ly + 44, lx + 30, ly + 52], fill=(251, 191, 36))
        draw.text(
            (lx + 36, ly + 40),
            f"Table Pivot Rxn : [{sf[0]:+5.1f}, {sf[1]:+4.1f}, {sf[2]:+5.1f}] N",
            fill=(251, 191, 36),
            font=font_mono,
        )

        # Bottom-left Phase & Tumble Angle Pill
        draw.rounded_rectangle([lx + 12, ly + main_h - 40, lx + 390, ly + main_h - 10], radius=5, fill=(11, 15, 20), outline=(52, 211, 153))
        draw.text(
            (lx + 22, ly + main_h - 32),
            f"Phase: {rec['phase']}  |  Tumble θ = {rec['tilt_deg']:5.1f}°",
            fill=(238, 243, 249),
            font=font_mono,
        )

        # Right Top: Synchronized Wrist Camera (with 3D Force Arrows!)
        rx, ry = 792, 56
        draw.rounded_rectangle([rx - 2, ry - 2, rx + wrist_w + 2, ry + wrist_h + 24], radius=7, fill=(17, 23, 32), outline=(38, 51, 69))
        draw.text((rx + 10, ry + 4), "WRIST EYE-IN-HAND CAM (WITH 3D FORCE VECTORS)", fill=(159, 176, 197), font=font_sm)
        canvas.paste(Image.fromarray(wrist_imgs[i]), (rx, ry + 22))

        # Right Middle: Normal vs. Tangential Friction Force Chart (N)
        c1_y0, c1_y1 = 354, 496
        draw.rounded_rectangle([rx - 2, c1_y0, rx + wrist_w + 2, c1_y1], radius=7, fill=(17, 23, 32), outline=(38, 51, 69))
        draw.text(
            (rx + 10, c1_y0 + 8),
            f"FORCE COUPLE (N): Normal {rec['ee_normal_z_n']:4.1f}N | Shear {rec['ee_shear_xy_n']:4.1f}N | Pivot {rec['support_total_n']:4.1f}N",
            fill=(203, 213, 225),
            font=font_sm,
        )
        px0, px1, py0, py1 = rx + 34, rx + wrist_w - 10, c1_y0 + 30, c1_y1 - 12
        denom = max(1, n_total - 1)
        pts_norm = [
            (px0 + (k / denom) * (px1 - px0), py1 - min(1.0, records[k]["ee_normal_z_n"] / max_force) * (py1 - py0))
            for k in range(n_total)
        ]
        pts_piv = [
            (px0 + (k / denom) * (px1 - px0), py1 - min(1.0, records[k]["support_total_n"] / max_force) * (py1 - py0))
            for k in range(n_total)
        ]
        pts_shear = [
            (px0 + (k / denom) * (px1 - px0), py1 - min(1.0, records[k]["ee_shear_xy_n"] / max_force) * (py1 - py0))
            for k in range(n_total)
        ]
        draw.line(pts_piv, fill=(251, 191, 36), width=2)
        draw.line(pts_norm, fill=(56, 189, 248), width=2)
        draw.line(pts_shear, fill=(248, 113, 113), width=2)
        cur_px = px0 + (i / denom) * (px1 - px0)
        draw.line([(cur_px, py0), (cur_px, py1)], fill=(248, 250, 252), width=2)
        draw.text((rx + 4, py0 - 2), f"{int(max_force)}N", fill=(108, 127, 150), font=font_sm)
        draw.text((rx + 8, py1 - 8), "0N", fill=(108, 127, 150), font=font_sm)

        # Right Bottom: Block Rotation Angle Curve (0 deg -> 90 deg)
        c2_y0, c2_y1 = 508, 644
        draw.rounded_rectangle([rx - 2, c2_y0, rx + wrist_w + 2, c2_y1], radius=7, fill=(17, 23, 32), outline=(38, 51, 69))
        draw.text((rx + 10, c2_y0 + 8), "BLOCK TUMBLE ORIENTATION θ (0° -> 90° EDGE PIVOT)", fill=(203, 213, 225), font=font_sm)
        qy0, qy1 = c2_y0 + 30, c2_y1 - 12
        pts_deg = [
            (px0 + (k / denom) * (px1 - px0), qy1 - min(1.0, records[k]["tilt_deg"] / 100.0) * (qy1 - qy0))
            for k in range(n_total)
        ]
        y_90 = qy1 - 0.90 * (qy1 - qy0)
        draw.line([(px0, y_90), (px1, y_90)], fill=(45, 85, 72), width=1)
        draw.line(pts_deg, fill=(52, 211, 153), width=3)
        draw.line([(cur_px, qy0), (cur_px, qy1)], fill=(248, 250, 252), width=2)
        draw.text((rx + 4, y_90 - 5), "90°", fill=(52, 211, 153), font=font_sm)
        draw.text((rx + 8, qy1 - 8), "0°", fill=(108, 127, 150), font=font_sm)

        vf = av.VideoFrame.from_image(canvas)
        for pkt in stream.encode(vf):
            container.mux(pkt)
        composed_frames.append(canvas)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    res = {"mp4": str(out_mp4), "mp4_kb": round(out_mp4.stat().st_size / 1024, 1), "frames": n_total}
    if out_gif is not None:
        out_gif = Path(out_gif).resolve()
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        sub = [im.resize((860, 452), Image.Resampling.LANCZOS) for im in composed_frames[::2]]
        pal = sub[len(sub) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        quant = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in sub]
        quant[0].save(out_gif, save_all=True, append_images=quant[1:], duration=85, loop=0, optimize=True)
        res["gif"] = str(out_gif)
        res["gif_kb"] = round(out_gif.stat().st_size / 1024, 1)

    return res
