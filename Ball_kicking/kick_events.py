from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_state(
    env,
    env_ids,
    ball_cfg=SceneEntityCfg("ball"),
    robot_cfg=SceneEntityCfg("robot"),
    distance_offset: float = 0.0,
    height_offset: float = 3.0,
):
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    local_offset = torch.zeros((len(env_ids), 3), device=env.device)
    local_offset[:, 0] = distance_offset
    world_offset = math_utils.quat_apply(robot_quat, local_offset)

    ball_pos = torch.zeros_like(robot_pos)
    ball_pos[:, 0] = robot_pos[:, 0] + world_offset[:, 0]
    ball_pos[:, 1] = robot_pos[:, 1] + world_offset[:, 1]
    ball_pos[:, 2] = height_offset

    # NEW: persist the world-frame anchor so it never has to be recomputed
    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env.ball_anchor_xy[env_ids] = ball_pos[:, :2]

    ball_vel = torch.zeros((len(env_ids), 6), device=env.device)
    ball_quat = torch.zeros((len(env_ids), 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)
    if hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z[env_ids] = 0.0
    if hasattr(env, "max_ball_vel_z"):
        env.max_ball_vel_z[env_ids] = 0.0
    if hasattr(env, "last_contact_foot"):
        env.last_contact_foot[env_ids] = 0.0
    if hasattr(env, "contact_count"):
        env.contact_count[env_ids] = 0.0
    if hasattr(env, "prev_kick_foot"):
        env.prev_kick_foot[env_ids] = -1
    if hasattr(env, "ball_ground_since"):
        env.ball_ground_since[env_ids] = float("inf")

    # Write the new state to the simulator
    ball_quat = torch.zeros((len(env_ids), 4), device=env.device)
    ball_quat[:, 0] = 1.0  # w=1 identity quaternion
    ball.write_root_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)


def constrain_ball_to_z_axis(
    env,
    env_ids,
    ball_cfg=SceneEntityCfg("ball"),
    min_height: float = 3.0,
) -> None:
    ball: RigidObject = env.scene[ball_cfg.name]
    all_ids = torch.arange(env.num_envs, device=env.device)

    ball_pos = ball.data.root_pos_w.clone()
    ball_lin_vel = ball.data.root_lin_vel_w.clone()

    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = ball_pos[:, :2].clone()  # fallback for first step

    ball_pos[:, :2] = env.ball_anchor_xy
    ball_lin_vel[:, :2] = 0.0

    at_floor = ball_pos[:, 2] < min_height
    ball_pos[:, 2] = torch.clamp(ball_pos[:, 2], min=min_height)
    ball_lin_vel[:, 2] = torch.where(
        at_floor & (ball_lin_vel[:, 2] < 0.0),
        torch.zeros_like(ball_lin_vel[:, 2]),
        ball_lin_vel[:, 2],
    )

    ball_quat = torch.zeros((env.num_envs, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=all_ids
    )
    ball.write_root_velocity_to_sim(
        torch.cat(
            [ball_lin_vel, torch.zeros((env.num_envs, 3), device=env.device)], dim=-1
        ),
        env_ids=all_ids,
    )
