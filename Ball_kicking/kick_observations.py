from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import kick_swing

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position expressed in the robot's base frame. (num_envs, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
    rel_pos_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)
    return rel_pos_b


def ball_linear_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball linear velocity expressed in the robot's base frame. (num_envs, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ball_lin_vel_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, ball.data.root_lin_vel_w
    )
    return ball_lin_vel_b


def feet_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_link", "right_ankle_link"]
    ),
) -> torch.Tensor:
    """Ankle positions in base frame, concatenated [left(3), right(3)]. (num_envs, 6)."""
    robot: Articulation = env.scene[robot_cfg.name]
    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, feet_pos_w.shape[1], -1)
    rel_pos_w = feet_pos_w - robot.data.root_pos_w.unsqueeze(1)
    feet_pos_b = math_utils.quat_apply_inverse(
        root_quat.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    return feet_pos_b


def knees_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_knee_link", "right_knee_link"]
    ),
) -> torch.Tensor:
    """Knee positions in base frame [left(3), right(3)]. (num_envs, 6). Optional — redundant with joint_pos."""
    robot: Articulation = env.scene[robot_cfg.name]
    knee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, knee_pos_w.shape[1], -1)
    rel_pos_w = knee_pos_w - robot.data.root_pos_w.unsqueeze(1)
    knee_pos_b = math_utils.quat_apply_inverse(
        root_quat.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    return knee_pos_b


def swing_phase(
    env: ManagerBasedRLEnv,
    theta_max_deg: float = 60.0,
    swing_time: float = 0.8,
    period: float = 0.8,
) -> torch.Tensor:
    """[sin(theta), cos(theta)] of the swing-cycle angle. (num_envs, 2).

    theta is the SAME triangular 0 -> theta_max -> 0 wave used as the reward
    target in kick_rewards.foot_swing_knee_extend, computed via the shared
    kick_common.swing_theta helper — not a generic phase ramp.

    CRITICAL: theta_max_deg / swing_time / period passed here (via the
    ObsTerm params in kick_env_cfg.py) MUST exactly match the values passed
    to the foot_swing_knee_extend RewTerm. A mismatch desyncs what the
    policy observes from what it is rewarded for tracking, with no error
    thrown — verify both param dicts side by side after any edit.
    """
    theta = kick_swing.swing_theta(env, theta_max_deg, swing_time, period)
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def last_contact_foot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot of the last foot to contact the ball. [1,0]=left, [0,1]=right, [0,0]=none. (num_envs, 2)."""
    if not hasattr(env, "last_contact_foot"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return env.last_contact_foot.clone()
