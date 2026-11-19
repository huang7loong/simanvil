"""Headless Blender 4.x Asset Normalizer & Z-Up Dual-Exporter (`simanvil.forge`).

Executed inside Blender (`blender --background --python blender_worker.py -- ...`).
Guarantees strict 1:1 Z-up geometric alignment between:
  1. `visual.glb`           (Three.js / PBR mesh, `export_yup=False`, Z-up)
  2. `collision_source.obj` (MuJoCo / CoACD triangulated mesh, `forward_axis='Y', up_axis='Z'`)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


def make_pbr_material(
    name: str,
    base_color: tuple[float, float, float, float],
    metallic: float = 0.75,
    roughness: float = 0.32,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = base_color
    return mat


def add_box(
    name: str,
    size_xyz: tuple[float, float, float],
    center_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size_xyz[0], size_xyz[1], size_xyz[2])
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def boolean_op(target: bpy.types.Object, cutter: bpy.types.Object, operation: str = "DIFFERENCE") -> None:
    bpy.context.view_layer.objects.active = target
    mod = target.modifiers.new(name=f"bool_{cutter.name}", type="BOOLEAN")
    mod.operation = operation
    mod.solver = "EXACT"
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def build_preset_kitting_tray() -> tuple[bpy.types.Object, dict]:
    """Dual-pocket CNC anodized kitting tray with an internal locating boss in Pocket R."""
    tray = add_box("kitting_tray", (0.156, 0.236, 0.038), (0.0, 0.0, 0.019))
    boolean_op(tray, add_box("pocket_l", (0.128, 0.092, 0.040), (0.0, -0.055, 0.028)), "DIFFERENCE")
    boolean_op(tray, add_box("pocket_r", (0.128, 0.092, 0.040), (0.0, 0.055, 0.028)), "DIFFERENCE")
    boolean_op(tray, add_box("grip_neg", (0.072, 0.014, 0.014), (0.0, -0.118, 0.033)), "DIFFERENCE")
    boolean_op(tray, add_box("grip_pos", (0.072, 0.014, 0.014), (0.0, 0.118, 0.033)), "DIFFERENCE")
    boolean_op(tray, add_box("locating_boss", (0.014, 0.014, 0.016), (0.0, 0.055, 0.014)), "UNION")

    bpy.context.view_layer.objects.active = tray
    bev = tray.modifiers.new(name="machined_chamfer", type="BEVEL")
    bev.width = 0.0012
    bev.segments = 2
    bev.limit_method = "ANGLE"
    bev.angle_limit = 0.65
    bpy.ops.object.modifier_apply(modifier=bev.name)

    mat_body = make_pbr_material("AnodizedSlateBlue", (0.16, 0.27, 0.42, 1.0), metallic=0.78, roughness=0.30)
    mat_floor = make_pbr_material("MachinedAluminum", (0.68, 0.73, 0.79, 1.0), metallic=0.85, roughness=0.24)
    tray.data.materials.append(mat_body)
    tray.data.materials.append(mat_floor)

    for poly in tray.data.polygons:
        cz = poly.center.z
        cx, cy = abs(poly.center.x), abs(poly.center.y)
        poly.material_index = 1 if (0.007 <= cz <= 0.023 and cx < 0.063 and 0.010 < cy < 0.100) else 0

    return tray, {
        "preset": "kitting_tray",
        "floor_thickness_m": 0.008,
        "rim_height_m": 0.038,
        "pocket_depth_m": 0.030,
        "left_pocket_xy_m": [0.0, -0.055],
        "right_pocket_xy_m": [0.0, 0.055],
        "boss_size_xyz_m": [0.014, 0.014, 0.014],
    }


def build_preset_u_bracket() -> tuple[bpy.types.Object, dict]:
    """Machined gold-anodized U-channel bridge bracket."""
    bracket = add_box("u_bracket", (0.048, 0.034, 0.032), (0.0, 0.0, 0.016))
    boolean_op(bracket, add_box("u_tunnel", (0.026, 0.044, 0.022), (0.0, 0.0, 0.008)), "DIFFERENCE")
    boolean_op(bracket, add_box("groove_l", (0.028, 0.003, 0.006), (0.0, -0.017, 0.025)), "DIFFERENCE")
    boolean_op(bracket, add_box("groove_r", (0.028, 0.003, 0.006), (0.0, 0.017, 0.025)), "DIFFERENCE")

    bpy.context.view_layer.objects.active = bracket
    bev = bracket.modifiers.new(name="bracket_chamfer", type="BEVEL")
    bev.width = 0.001
    bev.segments = 2
    bev.limit_method = "ANGLE"
    bev.angle_limit = 0.65
    bpy.ops.object.modifier_apply(modifier=bev.name)

    bracket.data.materials.append(make_pbr_material("AnodizedGold", (0.88, 0.63, 0.16, 1.0), metallic=0.82, roughness=0.26))
    return bracket, {
        "preset": "u_bracket",
        "outer_xyz_m": [0.048, 0.034, 0.032],
        "tunnel_width_x_m": 0.026,
        "tunnel_height_z_m": 0.019,
    }


def import_external_mesh(input_path: Path) -> bpy.types.Object:
    suffix = input_path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(input_path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(input_path), forward_axis="Y", up_axis="Z")
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(input_path), forward_axis="Y", up_axis="Z")
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(input_path), forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"Unsupported mesh extension: {suffix}")

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects found in {input_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    merged = bpy.context.active_object
    merged.name = input_path.stem
    if not merged.data.materials:
        merged.data.materials.append(
            make_pbr_material("DefaultPolymer", (0.72, 0.75, 0.78, 1.0), metallic=0.15, roughness=0.45)
        )
    return merged


def normalize_and_export(
    obj: bpy.types.Object,
    visual_glb: Path,
    collision_obj: Path,
    max_visual_faces: int = 25000,
) -> dict:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    raw_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    pre_min = [min(v[i] for v in raw_verts) for i in range(3)]
    pre_max = [max(v[i] for v in raw_verts) for i in range(3)]

    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    cov_world = Vector(obj.location)
    cov_shift = [float(cov_world.x), float(cov_world.y), float(cov_world.z)]
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    initial_faces = len(obj.data.polygons)
    if max_visual_faces > 0 and initial_faces > max_visual_faces:
        dec = obj.modifiers.new(name="web_decimate", type="DECIMATE")
        dec.ratio = float(max_visual_faces) / float(initial_faces)
        bpy.ops.object.modifier_apply(modifier=dec.name)

    for poly in obj.data.polygons:
        poly.use_smooth = False

    norm_verts = [v.co for v in obj.data.vertices]
    bounds_min = [float(min(v[i] for v in norm_verts)) for i in range(3)]
    bounds_max = [float(max(v[i] for v in norm_verts)) for i in range(3)]

    visual_glb.parent.mkdir(parents=True, exist_ok=True)
    collision_obj.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(visual_glb),
        export_format="GLB",
        export_yup=False,
        use_selection=True,
        export_apply=True,
    )
    bpy.ops.wm.obj_export(
        filepath=str(collision_obj),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_triangulated_mesh=True,
        export_materials=False,
    )

    return {
        "object_name": obj.name,
        "vertex_count": len(obj.data.vertices),
        "polygon_count": len(obj.data.polygons),
        "initial_polygon_count": initial_faces,
        "pre_shift_bounds_min_m": pre_min,
        "pre_shift_bounds_max_m": pre_max,
        "center_of_volume_shift_m": cov_shift,
        "bounds_min_m": bounds_min,
        "bounds_max_m": bounds_max,
        "extents_m": [bounds_max[i] - bounds_min[i] for i in range(3)],
        "visual_glb": visual_glb.name,
        "collision_source_obj": collision_obj.name,
        "coordinate_frame": "Z-up right-handed (X-right, Y-forward, Z-up), origin at center of volume",
    }


def parse_cli() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Headless Blender Dual-Exporter for SimAnvil")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--preset", choices=["kitting_tray", "u_bracket"])
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--collision-obj", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--max-visual-faces", type=int, default=25000)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_cli()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cad_features = {}
    if args.preset == "kitting_tray":
        obj, cad_features = build_preset_kitting_tray()
    elif args.preset == "u_bracket":
        obj, cad_features = build_preset_u_bracket()
    else:
        obj = import_external_mesh(args.input.resolve())

    meta = normalize_and_export(
        obj=obj,
        visual_glb=args.visual_glb.resolve(),
        collision_obj=args.collision_obj.resolve(),
        max_visual_faces=args.max_visual_faces,
    )
    if cad_features:
        meta["cad_features"] = cad_features

    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
