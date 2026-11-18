"""Example external adapter: Registering AI2/MuJoCo `ToyBlocks` directories with `simanvil`.

Demonstrates how external benchmarks plug into `simanvil` via `register_episode_adapter`
without adding benchmark-specific code to `src/simanvil/` core.

Usage:
    python examples/inspect_toyblocks_dir.py \
      --robot-xml /path/to/franka_droid/model.xml \
      --toyblocks-scripts /path/to/robot-zoo/scripts \
      /path/to/heldout/recovery-central-3 \
      --output outputs/toyblocks_session
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from simanvil import export_inspection_bundle, load_episodes, register_episode_adapter, verify_causal_replay
from simanvil.trace.adapters import extract_episode_from_callback
from simanvil.trace.schema import INTEGRATION_STATE


def register_toyblocks_adapter(robot_xml: Path, scripts_dir: Path) -> None:
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from toyblocks_world import FPS as TB_FPS, World

    def _is_toyblocks_dir(target: Path, model_xml: Path | None) -> bool:
        return model_xml is None and target.is_dir() and (target / "result.json").is_file()

    def _load_toyblocks(folder: Path, _xml: Path | None, max_frames: int | None, render_cams: bool):
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        world = World(robot_xml, result["blocks"])
        model = world.m
        slug = f"{folder.parent.name}_{folder.name}".replace("-", "_")
        episodes = []

        for npz_name, is_fault in [("failed-prefix.npz", True), ("episode.npz", not result.get("success", True))]:
            npz_path = folder / npz_name
            if not npz_path.is_file():
                continue
            with np.load(npz_path, allow_pickle=False) as z:
                states = z["state"].copy()
                controls = z["controls"].copy()
                phases = [str(p) for p in z["phase"].tolist()]
                final_state = z["final_state"].copy() if "final_state" in z.files else None
                stride = int(z["steps_per_frame"])

            rep_err = 0.0
            if final_state is not None and len(controls) == len(states) * stride:
                rep_err = verify_causal_replay(model, states[0], controls, final_state)["max_abs_qpos_err"]

            idxs = (
                np.arange(len(states))
                if (not max_frames or len(states) <= max_frames)
                else np.unique(np.linspace(0, len(states) - 1, max_frames).round().astype(np.int64))
            )

            def apply_fn(m: mj.MjModel, d: mj.MjData, k: int, _s=states, _c=controls, _p=phases, _ix=idxs, _st=stride):
                oi = int(_ix[k])
                mj.mj_setState(m, d, _s[oi], INTEGRATION_STATE)
                u_slice = _c[oi * _st : (oi + 1) * _st]
                act0 = u_slice[0] if len(u_slice) else np.zeros(m.nu, dtype=np.float32)
                d.ctrl[:] = act0
                return float(_s[oi, 0] - _s[0, 0]), _p[oi], act0, u_slice if len(_ix) == len(_s) else None

            passed = (npz_name == "episode.npz") and bool(result.get("success", True))
            episodes.append(
                extract_episode_from_callback(
                    key=f"{slug}_{npz_path.stem.replace('-', '_')}",
                    label=f"{folder.name} ({'PASS' if passed else '0.15× ANOMALY'})",
                    model_key=f"tb_{slug}",
                    robot_type="franka_fr3_robotiq_2f85",
                    task=f"ToyBlocks `{result.get('case', folder.name)}`",
                    model=model,
                    num_frames=len(idxs),
                    fps=TB_FPS,
                    apply_frame_fn=apply_fn,
                    episode_passed=passed,
                    force_anomaly_from_frame=int(len(idxs) * 0.6) if is_fault else None,
                    extra_audit={"causal_replay_max_qpos_err": rep_err},
                    render_cameras=render_cams,
                )
            )
        return episodes

    register_episode_adapter("toyblocks", _is_toyblocks_dir, _load_toyblocks)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect ToyBlocks runs via SimAnvil adapter")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--toyblocks-scripts", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/toyblocks_session"))
    p.add_argument("--max-frames", type=int, default=220)
    args = p.parse_args()

    register_toyblocks_adapter(args.robot_xml.resolve(), args.toyblocks_scripts.resolve())
    eps = []
    for r in args.runs:
        eps.extend(load_episodes(r, max_frames=args.max_frames))
    print(json.dumps(export_inspection_bundle(args.output, eps), indent=2))


if __name__ == "__main__":
    main()
