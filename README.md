# `simanvil`

**Deterministic MuJoCo Asset Forging, Interactive Contact-Force Inspection, and Quality-Weighted VLA Dataset Compilation.**

`simanvil` pairs a **zero-LLM deterministic physics & data core** (`src/simanvil/`) with **Agent Skills** (`skills/`) so both human robotics engineers and AI coding agents can compile concave 3D assets, audit contact mechanics without Reward Hacking, and export quality-weighted **LeRobot v3** datasets.

---

## Architecture & Core Subcommands

| Stage | CLI / Python API | What It Does |
| :--- | :--- | :--- |
| **1. Forge** | `simanvil forge`<br>`simanvil.forge_asset` | Normalizes `.glb/.obj/.stl/.ply` meshes in Headless Blender 4.x (`ORIGIN_CENTER_OF_VOLUME`, strict **Z-up** `< 1 µm` alignment), runs **CoACD** convex decomposition (`group="2"` visual, `group="3" condim="4"` collision, `group="4"` sites), and executes a 200-step MuJoCo stability & cavity-clearance smoke audit. |
| **2. Trace** | `simanvil.AnvilRecorder`<br>`simanvil.verify_causal_replay` | Records lightweight `(state, qpos, qvel, controls, final_state, phase)` `.npz` episodes in 4 lines of Python and verifies pure-actuator causal replay (`Δqpos < 1e-7`). |
| **3. Inspect** | `simanvil inspect`<br>`simanvil.export_inspection_bundle` | Compiles `.npz` trajectories into a zero-backend static HTML 3D Session Viewer featuring Three.js Z-up sync (`[w,x,y,z] → [x,y,z,w]`), PBR / Ghost+Hulls / Pure Hulls toggles, live 3D `mj_contactForce` arrows, and all-intra (`GOP=1`) dual camera streams. |
| **4. Assay** | `simanvil assay`<br>`simanvil.assay_to_lerobot_v3` | Detects slip/retry/fault anomalies, clamps stalled `task_progress` slopes to `0.0`, assigns `0.15×` BC `sample_weight` to anomalous frames, and exports PyAV-readback-verified **LeRobot v3** datasets. |

---

## Installation

```bash
# Core package (Forge + Trace + Inspect — lightweight, no PyTorch required)
pip install -e .

# Optional: LeRobot v3 dataset assaying & PyAV tensor readback
pip install -e ".[lerobot,dev]"
```

*Note: `simanvil forge` requires Blender 4.x installed (`/Applications/Blender.app` on macOS, `blender` on `PATH`, or `BLENDER_BIN=/path/to/Blender`).*

---

## 30-Second Quickstart

```bash
# 1. Run self-contained CoACD vs Shrink-Wrap benchmark & build HTML Session Viewer
python examples/quickstart_kitting.py

# 2. Launch local preview server
simanvil serve --dir outputs/quickstart/session --port 8895

# 3. Forge any external concave 3D mesh (.glb / .obj / .stl)
simanvil forge \
  --name part_bin \
  --input path/to/raw_bin.glb \
  --material aluminum_6061 \
  --threshold 0.02 \
  --output outputs/assets/part_bin

# 4. Inspect any recorded MuJoCo episode (+ optional LeRobot v3 dataset export)
simanvil inspect my_episode.npz --model scene.xml \
  --output outputs/session \
  --assay-out outputs/lerobot_v3
```

---

## Agent Skills Integration

`simanvil` ships a ready-to-mount [`plugin.json`](plugin.json) and three specialized agent skills under [`skills/`](skills/):

- [`simanvil-forge`](skills/simanvil-forge/SKILL.md): Parameter selection, articulated part splitting, and `MjSpec` mounting rules;
- [`simanvil-inspect`](skills/simanvil-inspect/SKILL.md): Anti-Reward-Hacking visual & contact-force audit checklist;
- [`simanvil-assay`](skills/simanvil-assay/SKILL.md): 4-tier hindsight relabeling and weighted BC / Flow-Matching loss integration.

---

## Roadmap

- [x] **v0.1**: Single-source Blender Z-up + CoACD `forge`, causal `AnvilRecorder`, multi-model HTML `inspect` workbench, weighted `LeRobot v3` `assay`, and Agent `skills/`
- [ ] **v0.2**: Multi-part articulated `.glb` auto-splitting (`hinge` / `slide` joint synthesis), MuJoCo 4 Filament PBR (`mjrf`) optional renderer, and `.mcap` real-robot log adapter
- [ ] **v0.3**: Headless contact-sheet visual critic CLI (`simanvil check-viewer`) and `mujoco_warp` batch domain randomization hooks
- [ ] **v0.4**: Single-image tabletop reconstruction workflow and real-robot two-stage Dry-Run safety verifier
