"""Curriculum functions for H1 juggling task."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def juggling_stage_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    success_threshold: float = 0.8,
    radius_step: float = 0.05,
    max_radius: float = 0.5,
) -> None:
    """
    Dynamically adjusts the ball drop radius (Stage 2) and activates alternate foot
    penalty (Stage 3) based on environment success rate.
    """
    # 1. Initialize custom curriculum state buffers on first call
    if not hasattr(env, "ball_radius"):
        env.ball_radius = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        # Stages: 1 (Fixed), 2 (Radius expanding), 3 (Alternate foot active)
        env.juggling_stage = torch.ones(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env.success_ema = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    if len(env_ids) == 0:
        return

    # 2. Evaluate Success (Survival)
    # If the environment reached the timeout without triggering "floor is lava", it's a success.
    ep_lengths = env.episode_length_buf[env_ids]
    is_success = (ep_lengths >= env.max_episode_length - 1).float()

    # Update Exponential Moving Average of success rate (alpha = 0.2)
    env.success_ema[env_ids] = 0.8 * env.success_ema[env_ids] + 0.2 * is_success

    # 3. Stage Progression Logic
    ready_to_advance = env.success_ema[env_ids] >= success_threshold

    # --- Stage 2: Expand Radius ---
    # Envs that hit 80% success but haven't reached 0.5m radius yet
    can_increase_radius = ready_to_advance & (env.ball_radius[env_ids] < max_radius)
    env.ball_radius[env_ids[can_increase_radius]] += radius_step

    # Reset success EMA for those that leveled up so they must prove themselves at the new difficulty
    env.success_ema[env_ids[can_increase_radius]] = 0.0

    # Update stage flag
    is_stage_2 = (env.ball_radius[env_ids] > 0.0) & (env.juggling_stage[env_ids] < 2)
    env.juggling_stage[env_ids[is_stage_2]] = 2

    # --- Stage 3: Consecutive Touches (Alternate Foot) ---
    # Envs that hit 80% success and have already mastered the 0.5m radius
    ready_for_stage_3 = ready_to_advance & (env.ball_radius[env_ids] >= max_radius)
    env.juggling_stage[env_ids[ready_for_stage_3]] = 3
