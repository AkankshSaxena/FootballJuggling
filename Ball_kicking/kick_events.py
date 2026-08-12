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
    distance_offset,
    lateral_offset,
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
    distance_offset: float = 0.4,
    lateral_offset: float = 0.15,
    height_offset: float = 0.3,
    randomize_side: bool = True,
):
    """Reset ball pose/velocity and (re)anchor it in front of the robot."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]
    n = len(env_ids)

    # 1. Determine lateral placement (Left vs Right)
    if randomize_side:
        # 50% chance for +1.0 (Left), 50% chance for -1.0 (Right)
        side_multiplier = torch.where(torch.rand(n, device=env.device) > 0.5, 1.0, -1.0)
    else:
        side_multiplier = torch.ones(n, device=env.device)

    lateral_offset_signed = lateral_offset * side_multiplier

    # 2. Define ball position in robot's local base frame
    ball_pos_b = torch.zeros((n, 3), device=env.device)
    ball_pos_b[:, 0] = distance_offset  # X: Forward
    ball_pos_b[:, 1] = lateral_offset_signed  # Y: Left/Right trajectory
    ball_pos_b[:, 2] = 0.0  # Z handled strictly in world frame

    # 3. Transform to World Frame (Crucial if the robot is rotated)
    yaw_quat = math_utils.yaw_quat(robot_quat)
    ball_pos_w = robot_pos + math_utils.quat_apply(yaw_quat, ball_pos_b)
    ball_pos_w[:, 2] = height_offset  # Set absolute spawn height

    # 4. Update Environment Positional Trackers
    anchor_xy = ball_pos_w[:, :2].clone()

    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env.ball_anchor_xy[env_ids] = anchor_xy

    if not hasattr(env, "ball_distance_offset"):
        env.ball_distance_offset = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "ball_lateral_offset"):
        env.ball_lateral_offset = torch.zeros(env.num_envs, device=env.device)

    env.ball_distance_offset[env_ids] = float(distance_offset)
    env.ball_lateral_offset[env_ids] = lateral_offset_signed

    # 5. Set Active Leg (0 = Right, 1 = Left)
    # If ball is on the left (Y > 0), active leg is strictly Left (1)
    if not hasattr(env, "active_leg"):
        env.active_leg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.active_leg[env_ids] = (lateral_offset_signed > 0.0).long()

    # 6. Reset Velocity and Orientation
    ball_vel = torch.zeros((n, 6), device=env.device)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0  # w=1 identity quaternion

    # 7. Reset Reward/Tracking Buffers
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

    # 8. Write to Simulator
    ball.write_root_pose_to_sim(
        torch.cat([ball_pos_w, ball_quat], dim=-1), env_ids=env_ids
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


def ball_gravity_scale(env, env_ids, k: float = 0.25, ball_cfg=SceneEntityCfg("ball")):
    """Upward force cancelling (1-k) of the ball's weight -> g_eff = k*9.81."""
    ball = env.scene[ball_cfg.name]
    m = ball.data.default_mass.to(env.device)  # (N,1)
    f = torch.zeros((env.num_envs, 1, 3), device=env.device)
    f[:, 0, 2] = m[:, 0] * 9.81 * (1.0 - k)
    ball.set_external_force_and_torque(f, torch.zeros_like(f))
    env.extras.setdefault("log", {})["debug/g_eff"] = k * 9.81
