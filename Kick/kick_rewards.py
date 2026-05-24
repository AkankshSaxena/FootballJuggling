"""Kick-specific reward functions for H1 kicking task."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def move_towards_ball(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_link"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Reward the robot for moving its foot closer to the ball.

    Returns shape: [N]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # Foot positions: [N, num_feet, 2] (xy only)
    foot_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :2]

    # Ball position: [N, 2] → unsqueeze to [N, 1, 2] for broadcasting
    ball_pos = ball.data.root_pos_w[:, :2].unsqueeze(1)

    # Distance from each foot to ball: [N, num_feet]
    dist = torch.norm(foot_pos - ball_pos, dim=-1)

    # Take minimum distance across feet: [N]
    min_dist = dist.min(dim=1)[0]

    # Reward: closer = higher. Clamp to avoid division explosion.
    return 1.0 / torch.clamp(min_dist, min=0.1)


def ball_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,  # points to foot_ball_contact_sensor
) -> torch.Tensor:
    """Reward any foot contact with the ball.

    Returns shape: [N] — 1.0 if contact, 0.0 if not.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Force history: [N, history_len, num_bodies, 3]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]

    # Max force magnitude across history and bodies: [N]
    max_force = forces.norm(dim=-1).max(dim=1)[0].max(dim=1)[0]

    # Binary contact: [N]
    has_contact = max_force > 1.0

    return has_contact.float()


def ball_upward_velocity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Reward upward ball velocity after being kicked.

    Returns shape: [N]
    """
    ball: RigidObject = env.scene[ball_cfg.name]

    # Z-velocity of ball root: [N]
    vel_z = ball.data.root_lin_vel_w[:, 2]

    # Only reward positive (upward) velocity
    return torch.clamp(vel_z, min=0.0)
