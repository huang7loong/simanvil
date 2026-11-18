---
name: simanvil-forge
description: >-
  Compiles concave or scanned 3D meshes (.glb, .gltf, .obj, .stl, .ply) into
  physically calibrated MuJoCo assets (MJCF + PBR visual.glb + CoACD convex
  collision hulls) using Headless Blender Z-up normalization and 200-step
  MuJoCo drop/cavity smoke audits. Use when importing external 3D objects,
  bins, trays, brackets, or containers into MuJoCo so cavities are not sealed
  by single-hull shrink-wrapping.
---

# SimAnvil Forge (`simanvil forge`)

Use `simanvil forge` (or `simanvil.forge_asset` / `simanvil.attach_forged_asset`) whenever a MuJoCo scene requires **concave geometry** (trays, bowls, cups, slotted brackets, sockets) where a naive `<geom type="mesh"/>` would seal off openings with a single outer convex hull.

## Core Invariants

1. **Single-Source Z-Up Alignment (`< 1 µm` bounds delta)**:
   - Headless Blender moves the mesh origin to `ORIGIN_CENTER_OF_VOLUME` at `(0, 0, 0)` and dual-exports:
     - `visual.glb` with `export_yup=False` (native Z-up PBR for Three.js);
     - `collision_source.obj` with `forward_axis='Y', up_axis='Z'` (for CoACD & MuJoCo).
2. **Standard MuJoCo Layer Separation**:
   - `group="2"`: Visual mesh (`contype="0" conaffinity="0"`);
   - `group="3"`: CoACD convex collision hulls (`condim="4"` torsional friction);
   - `group="4"`: Origin and top reference `<site>` elements.
3. **Parameter Selection Guidelines**:
   - Fine slots / thin walls (`< 8 mm` cavities): `--threshold 0.015 --resolution 80 --max-hulls 28`
   - Standard bins / brackets: `--threshold 0.02 --resolution 65 --max-hulls 24`
   - Batch GPU (`mujoco_warp`) scenes: cap `--max-hulls 16` to preserve contact throughput.
   - **Articulated objects** (drawers, hinged lids): split semantic moving parts into separate meshes BEFORE running `simanvil forge` on each link; never merge across a joint seam.

## CLI & `MjSpec` Attachment

```bash
simanvil forge \
  --name part_tray \
  --input path/to/raw_part.glb \
  --material aluminum_6061 \
  --threshold 0.02 \
  --output outputs/assets/part_tray
```

Mount onto any `mujoco.MjSpec` scene in Python:

```python
import mujoco as mj
from pathlib import Path
from simanvil import attach_forged_asset

spec = mj.MjSpec.from_file("robot_scene.xml")
attach_forged_asset(
    spec,
    Path("outputs/assets/part_tray"),
    pos=(0.50, 0.0, 0.04),
    dynamic=False,  # False for bolted fixtures, True for freejoint parts
)
model = spec.compile()
```

Always inspect `outputs/assets/<name>/forge_report.json` and confirm `smoke_test.passed == true` and `z_up_alignment_delta_m < 1e-4`.
