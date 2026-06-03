# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for H1 juggling environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


##
# Ball reset
##


def reset_ball_position_and_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the ball's pose and velocity based on the current curriculum stage.

    Stage behaviour:
        1  — No ball: placed underground at z = -10 m (out of sim).
        2  — Fixed at 0.3 m, 0.15 m in front of the robot.
        3  — Free Z only: spawned at 0.3 m, 0.15 m in front.
        4  — Dropped from 0.6 m, 0.15 m in front.
        5  — Dropped from 1.0 m, 0.15 m in front.
        6  — Dropped from 1.0 m, distance increments from 0.15 → 0.6 m.
        7  — Dropped from 1.0 m, random distance in [0.15, 0.6] m.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    stage: int = getattr(env, "curriculum_stage", 1)
    num_resets: int = len(env_ids)

    # Identity quaternion (w, x, y, z)
    ball_quat = torch.zeros((num_resets, 4), device=env.device)
    ball_quat[:, 0] = 1.0

    ball_pos = torch.zeros((num_resets, 3), device=env.device)
    ball_vel = torch.zeros((num_resets, 6), device=env.device)

    if stage == 1:
        # No ball in Stage 1 — hide underground
        ball_pos[:, 2] = -10.0

    else:
        robot_pos = robot.data.root_pos_w[env_ids]  # (num_resets, 3)

        # --- Spawn height ---
        if stage == 4:
            spawn_height = 0.6
        elif stage >= 5:
            spawn_height = 1.0
        else:
            # Stages 2 & 3: ball starts at 0.3 m (fixed / Z-free)
            spawn_height = 0.3

        # --- Spawn distance ---
        if stage == 6:
            # Distance increments each successful set; clamped to [0.15, 0.6]
            dist_val: float = float(getattr(env, "stage6_spawn_dist", 0.15))
            distances = torch.full((num_resets,), dist_val, device=env.device)
        elif stage == 7:
            # Random distance in [0.15, 0.6] m
            distances = torch.rand((num_resets,), device=env.device) * 0.45 + 0.15
        else:
            distances = torch.full((num_resets,), 0.15, device=env.device)

        # Spawn in front of robot along its +X (world frame assumption)
        ball_pos[:, 0] = robot_pos[:, 0] + distances
        ball_pos[:, 1] = robot_pos[:, 1]
        ball_pos[:, 2] = spawn_height

        # Stages 2: ball is constraint-fixed — zero velocity is correct.
        # Stages 3+: ball has free motion; reset velocity to zero so it
        #            falls naturally under gravity from spawn_height.

    ball.write_root_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)


##
# Tracking variable resets
##


def reset_juggling_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset all per-episode juggling tracking buffers on environment reset.

    Buffers expected on ``env`` (initialised by the custom env class):

    * ``last_contact_foot``       (num_envs, 2)  — one-hot last foot that hit ball
    * ``juggle_streak_buf``       (num_envs,)    — consecutive alt-foot streak count
    * ``ball_apex_height``        (num_envs,)    — highest ball z reached this episode
    * ``ball_prev_vel_z``         (num_envs,)    — previous step ball z-velocity (apex detect)
    * ``ball_ground_contact_time``(num_envs,)    — sim time when ball hit ground (-1 = none)
    * ``leg_raise_timer``         (num_envs, 2)  — how long each ankle has been in height band
    * ``contact_count``           (num_envs,)    — total ball–foot contacts this episode
    """
    if hasattr(env, "last_contact_foot"):
        env.last_contact_foot[env_ids] = 0.0

    if hasattr(env, "juggle_streak_buf"):
        env.juggle_streak_buf[env_ids] = 0

    if hasattr(env, "ball_apex_height"):
        env.ball_apex_height[env_ids] = 0.0

    if hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z[env_ids] = 0.0

    if hasattr(env, "ball_ground_contact_time"):
        # -1.0 signals "no ground contact recorded yet"
        env.ball_ground_contact_time[env_ids] = -1.0

    if hasattr(env, "leg_raise_timer"):
        env.leg_raise_timer[env_ids] = 0.0

    if hasattr(env, "contact_count"):
        env.contact_count[env_ids] = 0


##
# Stage 6 distance increment
##


def increment_stage6_distance(env: ManagerBasedRLEnv) -> None:
    """Increment the Stage 6 ball spawn distance after a successful iteration set.

    Called by the curriculum manager when the rolling success rate over the
    last 10 iterations exceeds 80 %.  Distance increments by 0.03 m and is
    clamped to a maximum of 0.6 m.
    """
    current: float = float(getattr(env, "stage6_spawn_dist", 0.15))
    env.stage6_spawn_dist = min(current + 0.03, 0.6)