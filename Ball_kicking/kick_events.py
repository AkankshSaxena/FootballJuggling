from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_offset: float = 0.5,
    height_offset: float = 0.3,
):
    """
    Resets the ball position relative to the robot.
    Spawns it exactly 'distance_offset' meters in front of the robot's facing direction,
    at a fixed 'height_offset' above the ground.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Get robot positions and orientations for the environments being reset
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    # Create a local offset vector [distance, 0, 0] assuming robot's forward is +X
    local_offset = torch.zeros((len(env_ids), 3), device=env.device)
    local_offset[:, 0] = distance_offset

    # Rotate the local offset to match the robot's current world orientation
    world_offset = math_utils.quat_apply(robot_quat, local_offset)

    # Set new ball position: Robot XY + Offset XY, Fixed Z
    ball_pos = torch.zeros_like(robot_pos)
    ball_pos[:, 0] = robot_pos[:, 0] + world_offset[:, 0]
    ball_pos[:, 1] = robot_pos[:, 1] + world_offset[:, 1]
    ball_pos[:, 2] = height_offset

    # Zero out all linear and angular velocities for the ball
    ball_vel = torch.zeros((len(env_ids), 6), device=env.device)

    # Write the new state to the simulator
    ball.write_root_pose_to_sim(ball_pos, env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)
