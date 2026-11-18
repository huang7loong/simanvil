---
name: simanvil-inspect
description: >-
  Records MuJoCo manipulation rollouts causally (`AnvilRecorder`) and compiles
  any `.npz` trajectory into a zero-backend interactive HTML 3D Session Viewer
  with Z-up Three.js sync, CoACD convex-hull overlays, 3D `mj_contactForce`
  arrows, and frame-exact dual camera streams. Use when auditing MuJoCo physics
  episodes, diagnosing grasp/IK/contact failures, or checking for simulation
  Reward Hacking.
---

# SimAnvil Inspect (`simanvil inspect`)

Use `simanvil inspect` to turn recorded MuJoCo trajectories into a self-contained static web workbench (`index.html` + `session_data.json` + GLBs + all-intra `GOP=1` H.264 camera streams) for human and headless-browser visual auditing.

## Recording Causal Trajectories (`AnvilRecorder`)

Never save RGB frames or contact arrays during physics stepping. Save lightweight `(state, qpos, qvel, controls, final_state, steps_per_frame, phase)` traces so camera views and 3D contact forces can be deterministically reconstructed offline:

```python
from pathlib import Path
from simanvil import AnvilRecorder

rec = AnvilRecorder(
    model,
    data,
    task="Pick red block and seat into tray pocket",
    robot_type="franka_fr3",
    fps=25,
    forged_assets={"kitting_tray": Path("outputs/assets/kitting_tray")},
)
for step in range(400):
    rec.step(ctrl, phase="S2_INSERT/seat_in_pocket")
rec.save(Path("outputs/runs/ep_01.npz"), passed=True)
```

## Building & Auditing the HTML Session Viewer

```bash
simanvil inspect outputs/runs/ep_01.npz --output outputs/session --serve --port 8895
```

### Agent Anti-Reward-Hacking Audit Checklist

1. **Causal Replay Check**: Confirm `causal_replay_max_qpos_err < 1e-7` in the JSON output (proves pure actuator commands reproduce the trajectory without state teleportation or assisted-grasp welding).
2. **Convex Hull Overlay (`Mode 2` / `Mode 3`)**: Toggle `2 · 半透 + 凸包` in the viewer to verify parts genuinely enter cavities rather than resting on invisible shrink-wrap membranes or penetrating thin walls.
3. **3D Contact Force Vectors**: Check peak `mj_contactForce` arrows (`Peak < 35 N` for gentle compliant grasps/seating; spikes `> 80 N` indicate unmodeled collisions, missing gravity compensation, or kinematic jamming).
4. **Browser Console Health**: When auditing via browser subagent or Playwright, assert `0` console errors and verify `window.__SIMANVIL_VIEWER_STATE__.ready === true`.
