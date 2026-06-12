from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position expressed in the robot's base frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Relative position in world frame
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w

    # Rotate into robot base frame
    rel_pos_b = math_utils.quat_rotate_inverse(robot.data.root_quat_w, rel_pos_w)
    return rel_pos_b


def ball_linear_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball linear velocity expressed in the robot's base frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_lin_vel_b = math_utils.quat_rotate_inverse(
        robot.data.root_quat_w, ball.data.root_lin_vel_w
    )
    return ball_lin_vel_b


def ball_linear_velocity_world(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball linear velocity in world frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_lin_vel_w.clone()


def ball_height(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball height (z-position) in world frame.

    Useful as a scalar cue for the policy to gauge juggle apex.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w[:, 2].unsqueeze(-1)


def ball_position_world(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position in world frame (x, y, z).

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w.clone()

def feet_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_link", "right_ankle_link"]),
) -> torch.Tensor:
    """Left and right ankle positions expressed in the robot's base frame.

    Concatenates both feet: [left_ankle (3), right_ankle (3)].

    Returns:
        Tensor of shape (num_envs, 6).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # body_pos_w: (num_envs, num_bodies, 3)
    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]  # (num_envs, 2, 3)

    # Expand robot base quat for broadcast: (num_envs, 1, 4) -> rotate each foot
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, feet_pos_w.shape[1], -1)
    rel_pos_w = feet_pos_w - robot.data.root_pos_w.unsqueeze(1)

    # Rotate each foot position into base frame
    feet_pos_b = math_utils.quat_rotate_inverse(
        root_quat.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1)  # (num_envs, 6)

    return feet_pos_b


def feet_height_world(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_link", "right_ankle_link"]),
) -> torch.Tensor:
    """Left and right ankle heights (z) in world frame.

    Returns:
        Tensor of shape (num_envs, 2).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]  # (num_envs, 2, 3)
    return feet_pos_w[:, :, 2]  # (num_envs, 2)

def last_contact_foot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot encoding of the last foot to contact the ball.

    - [1, 0] → left foot last contacted the ball
    - [0, 1] → right foot last contacted the ball
    - [0, 0] → no contact yet this episode

    Assumes ``env.last_contact_foot`` (shape: [num_envs, 2]) is initialised
    and updated by the event/step logic.

    Returns:
        Tensor of shape (num_envs, 2).
    """
    if not hasattr(env, "last_contact_foot"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return env.last_contact_foot.clone()


def juggle_streak(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current consecutive alternate-foot juggle streak per environment.

    Gives the policy a count signal for the Stage 7 streak bonus.
    Clipped to [0, 20] and normalised to [0, 1] before returning.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    if not hasattr(env, "juggle_streak_buf"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    streak = env.juggle_streak_buf.float().unsqueeze(-1)
    return (streak / 20.0).clamp(0.0, 1.0)