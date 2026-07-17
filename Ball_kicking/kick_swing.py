from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def swing_theta(
    env: ManagerBasedRLEnv,
    theta_max_deg: float = 60.0,
    swing_time: float = 0.8,
    period: float = 0.8,
) -> torch.Tensor:
    """Triangular swing-phase angle theta(t): 0 -> theta_max -> 0 over
    swing_time, then held at 0 until the period boundary (rest phase if
    period > swing_time).

    SINGLE SOURCE OF TRUTH. Used identically by:
      - kick_rewards.foot_swing_knee_extend  (reward target)
      - kick_observations.swing_phase        (sin/cos observation)

    If you change theta_max_deg / swing_time / period, change it ONCE here.
    The RewTerm and ObsTerm params in kick_env_cfg.py must still pass
    matching values to both — this only prevents the *formula* from
    drifting, not a config mismatch. Verify both param dicts match after
    any edit here.

    Returns:
        Tensor of shape (num_envs,), radians.
    """
    theta_max = math.radians(theta_max_deg)
    t = env.episode_length_buf.float() * env.step_dt
    t_cycle = torch.remainder(t, period)
    p = torch.clamp(t_cycle / swing_time, max=1.0)  # 0..1 during swing, 1 during rest
    tri = 1.0 - torch.abs(2.0 * p - 1.0)  # 0 -> 1 -> 0, stays 0 in rest
    return theta_max * tri
