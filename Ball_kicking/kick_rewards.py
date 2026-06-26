from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# STAGE 1 PENALTIES
def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns 1.0 on the step the episode terminates early (fall or out of bounds).
    In env_cfg.py, this will be multiplied by weight=-10.0.
    """
    return env.reset_terminated.float()


def ball_robot_dist_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 3.0,
) -> torch.Tensor:
    """
    Penalty when robot goes away from the ball (distance > 3m).
    Penalty = exp(dist^2). Clamped safely to avoid exploding gradients.
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Calculate 2D Euclidean distance (X, Y only) between robot root and ball root
    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )

    penalty = torch.where(
        dist > threshold,
        torch.exp(torch.square((dist / threshold) - 1)),
        torch.zeros_like(dist),
    )
    return penalty


# STAGE 1 REWARDS
def gated_ball_contact_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=".*ankle_link"
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_height: float = 0.3,
    max_height: float = 0.35,
    min_vel_z: float = 0.5,
) -> torch.Tensor:
    """
    Reward for kicking the ball. Fires only when:
    1. Foot contact sensor detects force > 1.0N
    2. Ball height is between 0.3m - 0.35m
    3. Ball vertical velocity > min_vel_z (0.5 m/s)
    """
    foot_sensor: ContactSensor = env.scene[foot_sensor_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # Detect foot contact
    foot_forces = foot_sensor.data.net_forces_w[:, :2, :]
    foot_force_mag = torch.norm(foot_forces, dim=-1)
    contact_this_step = foot_force_mag.sum(dim=1) > 1.0

    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    # Conditions
    height_valid = (ball_pos_z >= min_height) & (ball_pos_z <= max_height)
    vel_valid = ball_vel_z > min_vel_z

    any_valid_contact = contact_this_step & height_valid & vel_valid

    # --- Update tracking buffer for observation space (last_contact_foot) ---
    left_contact = foot_force_mag[:, 0] > 1.0
    right_contact = foot_force_mag[:, 1] > 1.0
    current_contact_foot = torch.stack(
        [left_contact.float(), right_contact.float()], dim=1
    )

    if not hasattr(env, "last_contact_foot"):
        env.last_contact_foot = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )

    env.last_contact_foot = torch.where(
        any_valid_contact.unsqueeze(-1).expand_as(current_contact_foot),
        current_contact_foot,
        env.last_contact_foot,
    )

    return any_valid_contact.float()


def apex_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Reward given exactly at the ball's apex if apex is within [0, 2] meters.
    Formula: exp(-(1 - h)^2)
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)

    # Apex is detected when Z velocity goes from positive to negative
    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_prev_vel_z = ball_vel_z.clone()

    # Apex reward formula -> exp(-(1-h)^2)
    reward_val = torch.exp(-torch.square(1.0 - ball_pos_z))

    # Gate: only reward if apex is within 2m
    within_bounds = (ball_pos_z >= 0.0) & (ball_pos_z <= 2.0)

    return (at_apex & within_bounds).float() * reward_val


# STAGE 1 TRACKING
def track_max_ball_vel_z(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Tracks the maximum vertical velocity of the ball across the episode.
    Returns the running max so it can be logged by setting weight=0.0 in env_cfg.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    if not hasattr(env, "max_ball_vel_z"):
        env.max_ball_vel_z = torch.zeros(env.num_envs, device=env.device)

    # Track maximum velocity
    env.max_ball_vel_z = torch.maximum(env.max_ball_vel_z, ball_vel_z)

    return env.max_ball_vel_z


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_xy = robot.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel_xy), dim=1) / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel_z = robot.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(ang_vel_z) / std**2)


def dof_pos_limits(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    out_of_limits = -(
        robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 0]
    ).clamp(max=0.0)
    out_of_limits += (
        robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 1]
    ).clamp(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_hip(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"]
    ),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos), dim=1)


def joint_deviation_arms(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            ".*_shoulder_pitch",
            ".*_shoulder_roll",
            ".*_shoulder_yaw",
            ".*_elbow",
        ],
    ),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos - default_pos), dim=1)


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
