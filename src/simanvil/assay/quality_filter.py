"""Frame-level anomaly detection, progress slope clamping, and BC loss downweighting (`simanvil.assay`)."""

from __future__ import annotations


ANOMALY_BC_WEIGHT = 0.15
ANOMALY_PHASE_TOKENS = ("fail", "fault", "error", "abort", "retry", "slip", "blocked", "stall")


def split_phase_label(raw_label: str) -> tuple[str, str]:
    text = str(raw_label).strip() or "EXECUTE/step"
    for sep in ("/", ":"):
        if sep in text:
            g2, g3 = text.split(sep, 1)
            return g2.strip().upper(), g3.strip()
    return text.upper(), text


def compute_episode_telemetry(
    *,
    phases: list[str],
    ee_forces_n: list[float],
    env_forces_n: list[float],
    extra_series: dict[str, list[float]] | None = None,
    force_anomaly_from_frame: int | None = None,
    impact_spike_threshold_n: float = 95.0,
    anomaly_weight: float = ANOMALY_BC_WEIGHT,
) -> list[dict]:
    """Compute G2/G3 labels, monotonic clamped progress, and BC `sample_weight` per frame."""
    n_frames = len(phases)
    telemetry: list[dict] = []
    clamped_progress = 0.0

    for idx in range(n_frames):
        raw_phase = phases[idx]
        g2_stage, g3_prim = split_phase_label(raw_phase)
        nom_prog = idx / float(max(1, n_frames - 1))

        low_phase = raw_phase.lower()
        is_anomaly = False
        if force_anomaly_from_frame is not None and idx >= force_anomaly_from_frame:
            is_anomaly = True
        elif any(tok in low_phase for tok in ANOMALY_PHASE_TOKENS):
            is_anomaly = True
        elif env_forces_n[idx] > impact_spike_threshold_n:
            is_anomaly = True

        if is_anomaly:
            bc_weight = float(anomaly_weight)
        else:
            bc_weight = 1.0
            clamped_progress = max(clamped_progress, nom_prog)

        row: dict[str, object] = {
            "g2_stage": g2_stage,
            "g3_primitive": g3_prim,
            "pad_force_n": round(float(ee_forces_n[idx]), 2),
            "part_tray_force_n": round(float(env_forces_n[idx]), 2),
            "task_progress_pct": round(clamped_progress * 100.0, 1),
            "bc_loss_weight": bc_weight,
            "anomaly": is_anomaly,
        }
        if extra_series:
            for s_key, vals in extra_series.items():
                row[s_key] = round(float(vals[idx]), 2)
        telemetry.append(row)

    return telemetry
