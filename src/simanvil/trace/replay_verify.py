"""Deterministic pure-actuator causal replay verification (`simanvil.trace`)."""

from __future__ import annotations

import mujoco as mj
import numpy as np

from simanvil.trace.schema import INTEGRATION_STATE


def verify_causal_replay(
    model: mj.MjModel,
    init_state: np.ndarray,
    controls: np.ndarray,
    final_state: np.ndarray,
) -> dict[str, float]:
    """Re-integrate `controls` from `init_state` and measure endpoint drift vs `final_state`."""
    d_sim = mj.MjData(model)
    mj.mj_setState(model, d_sim, init_state, INTEGRATION_STATE)
    for u in controls:
        d_sim.ctrl[:] = u
        mj.mj_step(model, d_sim)

    d_exp = mj.MjData(model)
    mj.mj_setState(model, d_exp, final_state, INTEGRATION_STATE)

    return {
        "max_abs_qpos_err": float(np.max(np.abs(d_sim.qpos - d_exp.qpos))),
        "max_abs_qvel_err": float(np.max(np.abs(d_sim.qvel - d_exp.qvel))),
    }
