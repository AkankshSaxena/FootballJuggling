from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _front_anchor_xy(
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    distance_offset: float,
    lateral_offset: float,
    device,
) -> torch.Tensor:
    """World-frame xy point offset from the robot in its own (heading-aligned) frame."""
    local_offset = torch.zeros((robot_pos.shape[0], 3), device=device)
    local_offset[:, 0] = distance_offset
    local_offset[:, 1] = -lateral_offset  # +lateral = robot's right (local -y)
    world_offset = math_utils.quat_apply(robot_quat, local_offset)
    return robot_pos[:, :2] + world_offset[:, :2]


def reset_ball_state(
    env,
    env_ids,
    ball_cfg=SceneEntityCfg("ball"),
    robot_cfg=SceneEntityCfg("robot"),
    distance_offset: float = 0.0,
    lateral_offset: float = 0.0,
    height_offset: float = 3.0,
):
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    anchor_xy = _front_anchor_xy(
        robot_pos, robot_quat, distance_offset, lateral_offset, env.device
    )

    ball_pos = torch.zeros_like(robot_pos)
    ball_pos[:, :2] = anchor_xy
    ball_pos[:, 2] = height_offset
    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env.ball_anchor_xy[env_ids] = anchor_xy
    env.ball_distance_offset = distance_offset
    env.ball_lateral_offset = lateral_offset

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
    if hasattr(env, "prev_ball_contact"):
        env.prev_ball_contact[env_ids] = False
    if hasattr(env, "last_kick_time"):
        env.last_kick_time[env_ids] = -1e9

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
    robot_cfg=SceneEntityCfg("robot"),
    min_height: float = 3.0,
    follow_robot: bool = True,
    distance_offset: float | None = None,
    lateral_offset: float | None = None,
    pin_x: bool = True,
    pin_y: bool = True,
) -> None:
    """Pin the ball's xy so only Z (fall/bounce) is free."""
    ball: RigidObject = env.scene[ball_cfg.name]
    all_ids = torch.arange(env.num_envs, device=env.device)

    ball_pos = ball.data.root_pos_w.clone()
    ball_lin_vel = ball.data.root_lin_vel_w.clone()

    if follow_robot:
        robot: Articulation = env.scene[robot_cfg.name]
        offset = (
            distance_offset
            if distance_offset is not None
            else getattr(env, "ball_distance_offset", 0.0)
        )
        lateral = (
            lateral_offset
            if lateral_offset is not None
            else getattr(env, "ball_lateral_offset", 0.0)
        )
        env.ball_anchor_xy = _front_anchor_xy(
            robot.data.root_pos_w, robot.data.root_quat_w, offset, lateral, env.device
        )
    elif not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = ball_pos[:, :2].clone()  # fallback for first step

    if pin_x:
        ball_pos[:, 0] = env.ball_anchor_xy[:, 0]
        ball_lin_vel[:, 0] = 0.0
    if pin_y:
        ball_pos[:, 1] = env.ball_anchor_xy[:, 1]
        ball_lin_vel[:, 1] = 0.0

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


def ball_gravity_scale(env, env_ids, k: float = 0.25, ball_cfg=SceneEntityCfg("ball")):
    """Upward force cancelling (1-k) of the ball's weight - g_eff = k*9.81."""
    ball = env.scene[ball_cfg.name]
    m = ball.data.default_mass.to(env.device)  # (N,1)
    f = torch.zeros((env.num_envs, 1, 3), device=env.device)
    f[:, 0, 2] = m[:, 0] * 9.81 * (1.0 - k)
    ball.set_external_force_and_torque(f, torch.zeros_like(f))
    env.extras.setdefault("log", {})["debug/g_eff"] = k * 9.81
