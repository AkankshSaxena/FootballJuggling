"""Kick-specific reward and termination functions for H1 juggling task."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import math
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- JUGGLING CONSTANTS ---
H_B = 1.2
H_F = 0.3
M_B = 0.45
G = 9.81


def juggle_impulse_gaussian(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float = 2.0,
) -> torch.Tensor:
    """Reward impulse J using a Gaussian distribution centered at target impulse."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    dt = env.step_dt

    # Net impulse = Force * dt
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    impulse = forces.norm(dim=-1).sum(dim=1) * dt

    # Target Gaussian center
    base_val = M_B * math.sqrt(2 * G * (H_B - H_F))
    target_impulse = 2.5 * base_val

    return torch.exp(-torch.square(impulse - target_impulse) / (std**2))


def apex_height_band(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Reward ball reaching target height band (0.8m to 1.6m) at apex."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_z = ball.data.root_pos_w[:, 2]
    ball_vz = ball.data.root_lin_vel_w[:, 2]

    # Detect apex: vertical velocity near zero
    at_apex = torch.abs(ball_vz) < 0.2
    # Check if height is within band
    in_band = (ball_z >= (H_B - 0.4)) & (ball_z <= (H_B + 0.4))

    return (at_apex & in_band).float()


def root_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal displacement (XY velocity) of the pelvis/base."""
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1)


def floor_is_lava(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Penalize when ball contacts the ground (Z < 0.15m)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return (ball.data.root_pos_w[:, 2] < 0.15).float()


def ball_ground_contact(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Termination condition: Returns True for envs where the ball hits the ground."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w[:, 2] < 0.15


def hand_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize hand/arm contact with the ball."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    has_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0
    return has_contact.float()


def alternate_foot_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the same foot contacting the ball consecutively (Stage 3)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]

    left_contact = forces[:, 0].norm(dim=-1) > 1.0
    right_contact = forces[:, 1].norm(dim=-1) > 1.0

    penalty = torch.zeros(env.num_envs, device=env.device)

    # Check against the custom buffer initialized in events.py
    if hasattr(env, "last_contact_foot"):
        same_left = left_contact & (env.last_contact_foot == 0)
        same_right = right_contact & (env.last_contact_foot == 1)
        penalty = (same_left | same_right).float()

        # Update buffer state
        env.last_contact_foot = torch.where(left_contact, 0, env.last_contact_foot)
        env.last_contact_foot = torch.where(right_contact, 1, env.last_contact_foot)

    current_stage = getattr(
        env, "juggling_stage", torch.ones(env.num_envs, device=env.device)
    )
    return penalty * (current_stage == 3).float()

def foot_tracking_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Reward the robot for moving its feet closer to the ball."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    foot_positions = robot.data.body_pos_w[:, robot_cfg.body_ids, :] 
    ball_position = ball.data.root_pos_w.unsqueeze(1) 
    
    distances = torch.norm(foot_positions - ball_position, dim=-1)
    min_distance = torch.min(distances, dim=1)[0]
    
    return torch.exp(-2.0 * min_distance)


def ball_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football"),
) -> torch.Tensor:
    """Reward for keeping the ball elevated: max(0, ball_z - floor_z)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_z = ball.data.root_pos_w[:, 2]
    
    return torch.clamp(ball_z, min=0.0)


def foot_ball_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Discrete reward (1 or 0) for the foot making contact with the ball."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Extract the net forces applied to the foot bodies
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    
    # Check if the maximum force vector length exceeds a small noise threshold
    has_contact = forces.norm(dim=-1).max(dim=1)[0] > 0.1
    
    return has_contact.float()