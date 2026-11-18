"""Quality-Weighted LeRobot v3 Dataset Exporter with Atomic PyAV Readback Verification (`simanvil.assay`)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from simanvil.trace.schema import Episode


os.environ.setdefault("HF_HUB_OFFLINE", "1")

CAM_FRONT = "observation.images.front"
CAM_WRIST = "observation.images.wrist"


def assay_to_lerobot_v3(
    episodes: Sequence["Episode"],
    output_dir: Path,
    *,
    repo_id: str = "local/simanvil_dataset",
    anomaly_policy: str = "downweight",
    overwrite: bool = True,
) -> dict:
    """Compile `Episode` sequences into a self-contained, readback-verified LeRobot v3 dataset."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot export requires optional dependencies. Install with: pip install 'simanvil[lerobot]'"
        ) from exc

    if anomaly_policy not in {"downweight", "zero-anomaly-frames", "drop-failed-episodes"}:
        raise ValueError(f"Unsupported anomaly_policy: {anomaly_policy}")

    selected = [
        ep for ep in episodes if (anomaly_policy != "drop-failed-episodes" or ep.audit.get("passed", True))
    ]
    if not selected:
        raise ValueError("No episodes remaining after applying anomaly_policy.")

    first = selected[0]
    fps = int(first.fps)
    robot_type = first.robot_type
    state_dim = int(first.obs_state.shape[1])
    act_dim = int(first.actions.shape[1])

    for ep in selected:
        if ep.obs_state.shape[1] != state_dim or ep.actions.shape[1] != act_dim:
            raise ValueError(f"Incompatible state/action dimensions across `{first.key}` and `{ep.key}`.")
        if not ep.overview_rgb or not ep.wrist_rgb:
            raise ValueError(f"Episode `{ep.key}` has no rendered camera RGB frames.")

    img_h, img_w, _ = first.overview_rgb[0].shape
    include_substeps = (first.action_substeps is not None) and all(
        ep.action_substeps is not None and ep.action_substeps.shape[1] == first.action_substeps.shape[1]
        for ep in selected
    )

    cam_feat = {"dtype": "video", "shape": (img_h, img_w, 3), "names": ["height", "width", "channels"]}
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": list(first.state_names)},
        "observation.contact_forces": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["ee_pad_force_n", "obj_env_force_n", "peak_contact_force_n"],
        },
        "action": {"dtype": "float32", "shape": (act_dim,), "names": list(first.action_names)},
        "sample_weight": {"dtype": "float32", "shape": (1,), "names": ["bc_loss_weight"]},
        "task_progress": {"dtype": "float32", "shape": (1,), "names": ["progress_0_to_1"]},
        "g2_stage_id": {"dtype": "int64", "shape": (1,), "names": None},
        "g3_primitive_id": {"dtype": "int64", "shape": (1,), "names": None},
        "anomaly_flag": {"dtype": "int64", "shape": (1,), "names": None},
        CAM_FRONT: cam_feat.copy(),
        CAM_WRIST: cam_feat.copy(),
    }
    if include_substeps and first.action_substeps is not None:
        features["action.substeps"] = {
            "dtype": "float32",
            "shape": (int(first.action_substeps.shape[1]),),
            "names": None,
        }

    out = Path(output_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {out}")

    g2_vocab: dict[str, int] = {}
    g3_vocab: dict[str, int] = {}
    ep_summaries: list[dict] = []
    ref_anchors: list[dict[int, dict]] = []

    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".simanvil-assay-") as tmp:
        staging_root = Path(tmp) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=staging_root,
            robot_type=robot_type,
            video_backend="pyav",
            vcodec="h264",
            encoder_threads=2,
            streaming_encoding=True,
        )

        total_frames = 0
        clean_frames = 0
        downweighted_frames = 0
        effective_weight_sum = 0.0

        try:
            for ep in selected:
                n_f = len(ep.times)
                anchors: dict[int, dict] = {}
                check_idxs = sorted({0, n_f // 2, n_f - 1})
                ep_clean = 0
                ep_down = 0

                for i in range(n_f):
                    tel = ep.telemetry[i]
                    g2_id = g2_vocab.setdefault(str(tel["g2_stage"]), len(g2_vocab))
                    g3_id = g3_vocab.setdefault(str(tel["g3_primitive"]), len(g3_vocab))
                    anom = 1 if bool(tel.get("anomaly", False)) else 0
                    raw_w = float(tel.get("bc_loss_weight", 1.0))
                    eff_w = 0.0 if (anom and anomaly_policy == "zero-anomaly-frames") else raw_w
                    prog_01 = float(tel.get("task_progress_pct", 0.0)) / 100.0

                    peak_c = max((float(c["mag"]) for c in ep.contacts[i]), default=0.0)
                    c_vec = np.array(
                        [float(tel.get("pad_force_n", 0.0)), float(tel.get("part_tray_force_n", 0.0)), peak_c],
                        dtype=np.float32,
                    )

                    row: dict[str, object] = {
                        "observation.state": ep.obs_state[i].astype(np.float32),
                        "observation.contact_forces": c_vec,
                        "action": ep.actions[i].astype(np.float32),
                        "sample_weight": np.array([eff_w], dtype=np.float32),
                        "task_progress": np.array([prog_01], dtype=np.float32),
                        "g2_stage_id": np.array([g2_id], dtype=np.int64),
                        "g3_primitive_id": np.array([g3_id], dtype=np.int64),
                        "anomaly_flag": np.array([anom], dtype=np.int64),
                        CAM_FRONT: ep.overview_rgb[i],
                        CAM_WRIST: ep.wrist_rgb[i],
                        "task": ep.task,
                    }
                    if include_substeps and ep.action_substeps is not None:
                        row["action.substeps"] = ep.action_substeps[i].astype(np.float32)

                    dataset.add_frame(row)

                    if i in check_idxs:
                        anchors[i] = {
                            "state": ep.obs_state[i].astype(np.float32),
                            "action": ep.actions[i].astype(np.float32),
                            "weight": eff_w,
                            "progress": prog_01,
                            "anom": anom,
                            "front": ep.overview_rgb[i].copy(),
                            "wrist": ep.wrist_rgb[i].copy(),
                        }

                    total_frames += 1
                    effective_weight_sum += eff_w
                    if eff_w >= 0.999:
                        clean_frames += 1
                        ep_clean += 1
                    else:
                        downweighted_frames += 1
                        ep_down += 1

                dataset.save_episode()
                ref_anchors.append(anchors)
                ep_summaries.append(
                    {
                        "key": ep.key,
                        "label": ep.label,
                        "passed": bool(ep.audit.get("passed", True)),
                        "frames": n_f,
                        "clean_frames": ep_clean,
                        "downweighted_frames": ep_down,
                        "task": ep.task,
                    }
                )
        finally:
            dataset.finalize()

        loaded = LeRobotDataset(repo_id, root=staging_root, video_backend="pyav")
        if len(loaded) != total_frames:
            raise RuntimeError(f"Readback frame count mismatch: {len(loaded)} != {total_frames}")

        cursor = 0
        rgb_maes: list[float] = []
        verified_rows: list[int] = []
        for ep_idx, (ep, anchors) in enumerate(zip(selected, ref_anchors, strict=True)):
            for local_i, ref in anchors.items():
                global_i = cursor + local_i
                sample = loaded[global_i]
                assert int(sample["episode_index"]) == ep_idx
                assert int(sample["frame_index"]) == local_i
                np.testing.assert_allclose(sample["observation.state"].numpy(), ref["state"], atol=1e-6)
                np.testing.assert_allclose(sample["action"].numpy(), ref["action"], atol=1e-6)
                np.testing.assert_allclose(float(sample["sample_weight"].item()), ref["weight"], atol=1e-6)
                np.testing.assert_allclose(float(sample["task_progress"].item()), ref["progress"], atol=1e-5)
                assert int(sample["anomaly_flag"].item()) == ref["anom"]
                for cam_key, ref_img in ((CAM_FRONT, ref["front"]), (CAM_WRIST, ref["wrist"])):
                    decoded = sample[cam_key].permute(1, 2, 0).numpy()
                    mae = float(np.mean(np.abs(decoded - ref_img / 255.0)))
                    if mae > 0.05:
                        raise ValueError(f"Decoded RGB MAE exceeded threshold at row {global_i} ({cam_key}): {mae:.4f}")
                    rgb_maes.append(mae)
                verified_rows.append(global_i)
            cursor += len(ep.times)

        manifest = {
            "schema": "simanvil.assay_manifest.v0.1",
            "repo_id": repo_id,
            "robot_type": robot_type,
            "fps": fps,
            "anomaly_policy": anomaly_policy,
            "episodes_exported": len(selected),
            "total_frames": total_frames,
            "clean_frames_1x": clean_frames,
            "downweighted_frames": downweighted_frames,
            "effective_weighted_frames": round(effective_weight_sum, 2),
            "verified_readback_rows": verified_rows,
            "max_decoded_rgb_mae_0_to_1": round(max(rgb_maes) if rgb_maes else 0.0, 5),
            "g2_stage_vocab": g2_vocab,
            "g3_primitive_vocab": g3_vocab,
            "episodes": ep_summaries,
            "training_loss_hint": "weighted_bc_loss = (per_sample_loss * batch['sample_weight']).sum() / batch['sample_weight'].sum().clamp_min(1e-6)",
        }
        (staging_root / "meta" / "simanvil_assay.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(staging_root), str(out))

    manifest["dataset_dir"] = str(out)
    return manifest
