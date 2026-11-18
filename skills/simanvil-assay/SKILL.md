---
name: simanvil-assay
description: >-
  Performs 4-tier hindsight relabeling (G1-G3), detects slip/retry/fault
  anomalies, clamps stalled task progress slopes to zero, downweights anomalous
  frames (`0.15x` BC loss weight), and compiles MuJoCo episodes into
  readback-verified LeRobot v3 datasets (`simanvil assay`). Use when preparing
  clean or quality-weighted VLA / imitation-learning training datasets from
  MuJoCo rollouts.
---

# SimAnvil Assay (`simanvil assay`)

Use `simanvil assay` (requires `pip install 'simanvil[lerobot]'`) to purify raw MuJoCo trajectories into self-contained **LeRobot v3** datasets (`Parquet` + streaming `PyAV H.264` dual cameras) with explicit frame-level quality weights.

## Phase Tagging & Anomaly Policies

Format per-frame `phase` strings as `"G2_STAGE/g3_contact_primitive"` (e.g., `"S1_PICK/close_pads"`, `"S2_INSERT/retry_slip"`).
Frames matching fault/retry tokens (`fail`, `fault`, `error`, `abort`, `retry`, `slip`, `blocked`), caller-marked failure tails (`passed=False`), or excessive environmental impact spikes (`> 95 N`) are flagged with `anomaly_flag = 1` and `task_progress` slope clamped to `0.0`.

Choose `--anomaly-policy` based on downstream policy architecture:
- `downweight` *(default)*: Keeps all frames in sequence, setting `sample_weight = 0.15` on anomalous frames and `1.00` on clean frames. Ideal for action-chunked VLA / Diffusion / Flow-Matching models where dropping mid-episode frames breaks temporal continuity.
- `zero-anomaly-frames`: Preserves sequence continuity with `sample_weight = 0.00` on anomalous frames.
- `drop-failed-episodes`: Excludes any episode where `audit.passed == False`.

## CLI & Weighted Loss Integration

```bash
simanvil assay outputs/runs/*.npz \
  --anomaly-policy downweight \
  --repo-id local/my_vla_dataset \
  --output outputs/lerobot_v3
```

Every export automatically re-opens the finalized dataset with `PyAV` and verifies `observation.state`, `action`, `sample_weight`, `task_progress`, and decoded RGB pixel error (`MAE < 0.05`), writing summary statistics to `meta/simanvil_assay.json`.

In PyTorch training loops:

```python
per_frame_loss = criterion(pred_action, batch["action"])  # (B, ...)
weights = batch["sample_weight"].view(-1, *([1] * (per_frame_loss.ndim - 1)))
loss = (per_frame_loss * weights).sum() / weights.sum().clamp_min(1e-6)
```
