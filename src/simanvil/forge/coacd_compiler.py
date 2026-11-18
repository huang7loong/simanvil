"""CoACD Approximate Convex Decomposition, MJCF Assembler & MjSpec Attachment (`simanvil.forge`).

Complies with standard MuJoCo asset layer conventions:
  - `group="2"` : Visual meshes (`contype="0" conaffinity="0"`)
  - `group="3"` : CoACD convex collision hulls (`condim="4"` torsional friction)
  - `group="4"` : Reference & affordance `<site>` elements
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import coacd
from lxml import etree
import mujoco as mj
import numpy as np
import trimesh

from simanvil.forge.smoke_audit import run_physics_smoke_audit


MATERIAL_PRESETS: dict[str, dict[str, object]] = {
    "aluminum_6061": {
        "density": 2700.0,
        "friction": [0.85, 0.008, 0.0002],
        "rgba": [0.22, 0.34, 0.49, 1.0],
    },
    "anodized_gold": {
        "density": 1850.0,
        "friction": [0.90, 0.010, 0.0003],
        "rgba": [0.88, 0.63, 0.16, 1.0],
    },
    "delrin_pom": {
        "density": 1410.0,
        "friction": [0.65, 0.006, 0.0002],
        "rgba": [0.85, 0.87, 0.89, 1.0],
    },
    "abs_plastic": {
        "density": 1050.0,
        "friction": [0.78, 0.007, 0.0002],
        "rgba": [0.25, 0.58, 0.82, 1.0],
    },
    "stainless_304": {
        "density": 7800.0,
        "friction": [0.75, 0.008, 0.0002],
        "rgba": [0.72, 0.75, 0.78, 1.0],
    },
}


@dataclass
class ForgeConfig:
    asset_name: str
    output_dir: Path
    input_mesh: Path | None = None
    preset: str | None = None
    material: str = "aluminum_6061"
    threshold: float = 0.02
    max_convex_hull: int = 24
    preprocess_mode: str = "auto"
    preprocess_resolution: int = 65
    mcts_max_depth: int = 3
    max_visual_faces: int = 25000
    seed: int = 0
    blender_bin: Path | None = None


def find_blender_binary(explicit: Path | None = None) -> Path:
    if explicit is not None and explicit.is_file():
        return explicit
    env_bin = os.environ.get("BLENDER_BIN")
    candidates: list[Path] = []
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(Path("/Applications/Blender.app/Contents/MacOS/Blender"))
    which_bin = shutil.which("blender") or shutil.which("Blender")
    if which_bin:
        candidates.append(Path(which_bin))
    # Also check sibling project caches on developer workstations
    projects_dir = Path(__file__).resolve().parents[4]
    for cached in projects_dir.glob("*/.cache/blender-replay/Blender.app/Contents/MacOS/Blender"):
        candidates.append(cached)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate headless Blender binary. Install Blender 4.x or set BLENDER_BIN=/path/to/Blender."
    )


def _hull_palette_rgba(index: int, alpha: int = 255) -> list[int]:
    hue = (index * 0.618033988749895) % 1.0
    sat = 0.72 if index % 2 == 0 else 0.60
    val = 0.94 if index % 3 != 2 else 0.82
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), alpha]


def export_colored_hulls_glb(hull_meshes: list[trimesh.Trimesh], out_glb: Path) -> None:
    scene = trimesh.Scene()
    for idx, hm in enumerate(hull_meshes):
        colored = hm.copy()
        rgba_f = [c / 255.0 for c in _hull_palette_rgba(idx, alpha=255)]
        colored.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=f"hull_mat_{idx:02d}",
                baseColorFactor=rgba_f,
                metallicFactor=0.15,
                roughnessFactor=0.45,
            )
        )
        scene.add_geometry(colored, node_name=f"hull_{idx:02d}", geom_name=f"hull_{idx:02d}")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_glb.write_bytes(scene.export(file_type="glb"))


def _build_mjcf_bundle(
    asset_name: str,
    visual_obj_rel: str,
    hull_rels: list[str],
    bounds_max: list[float],
    material_spec: dict[str, object],
    out_xml: Path,
) -> None:
    root = etree.Element("mujoco", model=f"{asset_name}_forged")
    etree.SubElement(root, "compiler", angle="radian")

    asset_el = etree.SubElement(root, "asset")
    etree.SubElement(asset_el, "mesh", name=f"{asset_name}_visual", file=visual_obj_rel)
    for idx, hrel in enumerate(hull_rels):
        etree.SubElement(asset_el, "mesh", name=f"{asset_name}_hull_{idx:02d}", file=hrel)

    worldbody = etree.SubElement(root, "worldbody")
    body = etree.SubElement(worldbody, "body", name=asset_name, pos="0 0 0")
    rgba_str = " ".join(f"{c:.4f}" for c in material_spec["rgba"])
    friction_str = " ".join(f"{f:.5f}" for f in material_spec["friction"])
    density_str = f"{float(material_spec['density']):.2f}"

    # Group 2: Visual mesh ONLY (contype=0, conaffinity=0)
    etree.SubElement(
        body,
        "geom",
        name=f"{asset_name}_visual_geom",
        type="mesh",
        mesh=f"{asset_name}_visual",
        group="2",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba=rgba_str,
    )
    # Group 3: CoACD Convex Hulls ONLY (condim=4 torsional friction)
    for idx in range(len(hull_rels)):
        etree.SubElement(
            body,
            "geom",
            name=f"{asset_name}_col_{idx:02d}",
            type="mesh",
            mesh=f"{asset_name}_hull_{idx:02d}",
            group="3",
            condim="4",
            density=density_str,
            friction=friction_str,
        )
    # Group 4: Standard reference & top affordance sites
    etree.SubElement(body, "site", name=f"{asset_name}_origin", pos="0 0 0", size="0.002", group="4")
    etree.SubElement(
        body,
        "site",
        name=f"{asset_name}_top",
        pos=f"0 0 {float(bounds_max[2]):.5f}",
        size="0.002",
        group="4",
    )

    tree = etree.ElementTree(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_xml), pretty_print=True, xml_declaration=True, encoding="utf-8")


def attach_forged_asset(
    spec: mj.MjSpec,
    asset_dir: Path,
    *,
    body_name: str | None = None,
    pos: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] | list[float] = (1.0, 0.0, 0.0, 0.0),
    dynamic: bool = True,
    mode: str = "coacd",
) -> mj.MjsBody:
    """Attach a `simanvil forge` asset directory directly onto any `mujoco.MjSpec` scene."""
    if mode not in {"coacd", "shrinkwrap"}:
        raise ValueError(f"Unsupported collision mode: {mode}")
    adir = Path(asset_dir).resolve()
    report = json.loads((adir / "forge_report.json").read_text(encoding="utf-8"))
    bname = body_name or report["asset_name"]
    mat_spec = MATERIAL_PRESETS.get(report["material"], MATERIAL_PRESETS["aluminum_6061"])

    body = spec.worldbody.add_body(name=bname, pos=list(pos), quat=list(quat))
    if dynamic:
        body.add_freejoint(name=f"{bname}_free")

    vis_mesh_name = f"{bname}_vis_mesh"
    ms_vis = spec.add_mesh(name=vis_mesh_name)
    ms_vis.file = str(adir / "collision_source.obj")
    body.add_geom(
        name=f"{bname}_visual",
        type=mj.mjtGeom.mjGEOM_MESH,
        meshname=vis_mesh_name,
        group=2,
        contype=0,
        conaffinity=0,
        mass=0.0,
        rgba=list(mat_spec["rgba"]),
    )

    hull_files = (
        sorted((adir / "hulls").glob("hull_*.obj"))
        if mode == "coacd"
        else [adir / "naive_shrinkwrap.obj"]
    )
    for idx, hf in enumerate(hull_files):
        cm_name = f"{bname}_col_mesh_{idx:02d}"
        ms_col = spec.add_mesh(name=cm_name)
        ms_col.file = str(hf)
        body.add_geom(
            name=f"{bname}_hull_{idx:02d}" if mode == "coacd" else f"{bname}_shrinkwrap",
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=cm_name,
            group=3,
            condim=4,
            density=float(report["density_kg_m3"]),
            friction=list(report["friction"]),
            rgba=[0.9, 0.3, 0.3, 0.35],
        )

    body.add_site(name=f"{bname}_origin", pos=[0.0, 0.0, 0.0], size=[0.002, 0.002, 0.002], group=4)
    return body


def forge_asset(cfg: ForgeConfig) -> dict:
    """Compile an external 3D mesh or built-in preset into a verified Z-up MuJoCo + GLB asset bundle."""
    t0 = time.perf_counter()
    blender_bin = find_blender_binary(cfg.blender_bin)
    worker_script = Path(__file__).with_name("blender_worker.py")

    out_dir = cfg.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_glb = out_dir / "visual.glb"
    collision_obj = out_dir / "collision_source.obj"
    meta_json = out_dir / "blender_meta.json"

    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(worker_script),
        "--",
        "--visual-glb",
        str(visual_glb),
        "--collision-obj",
        str(collision_obj),
        "--meta-json",
        str(meta_json),
        "--max-visual-faces",
        str(cfg.max_visual_faces),
    ]
    if cfg.preset is not None:
        cmd.extend(["--preset", cfg.preset])
    elif cfg.input_mesh is not None:
        cmd.extend(["--input", str(cfg.input_mesh.resolve())])
    else:
        raise ValueError("Either ForgeConfig.preset or ForgeConfig.input_mesh must be specified.")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Headless Blender failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    blender_meta = json.loads(meta_json.read_text(encoding="utf-8"))

    glb_mesh = trimesh.load(visual_glb, force="mesh")
    src_mesh = trimesh.load(collision_obj, force="mesh", process=False)
    bounds_delta = float(np.max(np.abs(glb_mesh.bounds - src_mesh.bounds)))
    if bounds_delta > 2e-4:
        raise RuntimeError(
            f"Z-up coordinate mismatch between visual.glb and collision_source.obj: delta={bounds_delta:.6f}m"
        )

    coacd.set_log_level("error")
    c_mesh = coacd.Mesh(src_mesh.vertices, src_mesh.faces)
    parts = coacd.run_coacd(
        c_mesh,
        threshold=cfg.threshold,
        max_convex_hull=cfg.max_convex_hull,
        preprocess_mode=cfg.preprocess_mode,
        preprocess_resolution=cfg.preprocess_resolution,
        mcts_max_depth=cfg.mcts_max_depth,
        seed=cfg.seed,
    )

    hulls_dir = out_dir / "hulls"
    if hulls_dir.exists():
        shutil.rmtree(hulls_dir)
    hulls_dir.mkdir(parents=True, exist_ok=True)

    hull_meshes: list[trimesh.Trimesh] = []
    hull_paths: list[Path] = []
    hull_rels: list[str] = []
    hull_stats: list[dict] = []

    for idx, (verts, faces) in enumerate(parts):
        hm = trimesh.Trimesh(vertices=verts, faces=faces, process=True).convex_hull
        hpath = hulls_dir / f"hull_{idx:02d}.obj"
        hm.export(hpath)
        hull_meshes.append(hm)
        hull_paths.append(hpath)
        hull_rels.append(f"hulls/hull_{idx:02d}.obj")
        hull_stats.append(
            {
                "id": f"hull_{idx:02d}",
                "vertices": int(len(hm.vertices)),
                "faces": int(len(hm.faces)),
                "volume_cm3": round(float(hm.volume) * 1e6, 3),
                "color_rgba": _hull_palette_rgba(idx, alpha=255),
            }
        )

    hulls_debug_glb = out_dir / "hulls_debug.glb"
    export_colored_hulls_glb(hull_meshes, hulls_debug_glb)

    naive_hull = src_mesh.convex_hull
    naive_obj = out_dir / "naive_shrinkwrap.obj"
    naive_glb = out_dir / "naive_shrinkwrap.glb"
    naive_hull.export(naive_obj)
    naive_vis = naive_hull.copy()
    naive_vis.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="shrinkwrap_danger",
            baseColorFactor=[0.95, 0.24, 0.24, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.4,
        )
    )
    naive_glb.write_bytes(trimesh.Scene(naive_vis).export(file_type="glb"))

    mat_spec = MATERIAL_PRESETS.get(cfg.material, MATERIAL_PRESETS["aluminum_6061"])
    asset_xml = out_dir / "asset.xml"
    _build_mjcf_bundle(
        asset_name=cfg.asset_name,
        visual_obj_rel="collision_source.obj",
        hull_rels=hull_rels,
        bounds_max=blender_meta["bounds_max_m"],
        material_spec=mat_spec,
        out_xml=asset_xml,
    )

    probe_xy = (0.0, -0.055) if cfg.preset == "kitting_tray" else (0.0, 0.0)
    smoke = run_physics_smoke_audit(
        asset_name=cfg.asset_name,
        collision_obj=collision_obj,
        hull_paths=hull_paths,
        bounds_min=blender_meta["bounds_min_m"],
        bounds_max=blender_meta["bounds_max_m"],
        material_spec=mat_spec,
        probe_xy=probe_xy,
    )
    if not smoke["passed"]:
        raise RuntimeError(f"MuJoCo smoke audit failed for {cfg.asset_name}: {smoke}")

    report = {
        "schema": "simanvil.forge_report.v0.1",
        "asset_name": cfg.asset_name,
        "material": cfg.material,
        "density_kg_m3": mat_spec["density"],
        "friction": mat_spec["friction"],
        "mjcf_groups": {"visual": 2, "collision": 3, "sites": 4},
        "coacd_params": {
            "threshold": cfg.threshold,
            "max_convex_hull": cfg.max_convex_hull,
            "preprocess_mode": cfg.preprocess_mode,
            "preprocess_resolution": cfg.preprocess_resolution,
            "mcts_max_depth": cfg.mcts_max_depth,
        },
        "z_up_alignment_delta_m": round(bounds_delta, 8),
        "hull_count": len(hull_meshes),
        "total_hull_vertices": sum(h["vertices"] for h in hull_stats),
        "total_hull_faces": sum(h["faces"] for h in hull_stats),
        "naive_shrinkwrap_volume_cm3": round(float(naive_hull.volume) * 1e6, 2),
        "coacd_total_volume_cm3": round(sum(h["volume_cm3"] for h in hull_stats), 2),
        "occluded_void_volume_cm3": round(
            float(naive_hull.volume) * 1e6 - sum(h["volume_cm3"] for h in hull_stats), 2
        ),
        "blender": blender_meta,
        "smoke_test": smoke,
        "hulls": hull_stats,
        "files": {
            "visual_glb": str(visual_glb),
            "hulls_debug_glb": str(hulls_debug_glb),
            "naive_shrinkwrap_glb": str(naive_glb),
            "collision_source_obj": str(collision_obj),
            "asset_xml": str(asset_xml),
        },
        "compile_wall_time_s": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "forge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
