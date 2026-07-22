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
    """World-frame xy point offset from the robot in its own (heading-aligned) frame.

    distance_offset -> robot local +x (forward).
    lateral_offset  -> robot's RIGHT, i.e. the right-foot side. The H1 root frame
        is x-forward / y-LEFT / z-up, so a positive lateral_offset maps to local
        -y. This rotates with the robot's heading. Returns (N, 2). Shared by spawn
        (reset_ball_state) and the optional follow-constraint so both use identical
        offset math.
    """
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
    distance_offset: float = 0.47,
    lateral_offset: float = 0.08,
    height_offset: float = 0.24,
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

    # Persist the world-frame anchor (used when the follow-constraint is OFF) and
    # the spawn offsets (reused by the follow-constraint when it is ON) so both are
    # driven from this single reset config.
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
    min_height: float = 0.24,
    follow_robot: bool = False,
    distance_offset: float | None = None,
    lateral_offset: float | None = None,
) -> None:
    """Pin the ball's xy so only Z (fall/bounce) is free.

    follow_robot=False (default): pin to the fixed spawn anchor -- the ball stays
    where it was reset even if the robot walks away.
    follow_robot=True: re-anchor `distance_offset` forward / `lateral_offset` to the
    right of the *current* robot pose every step, so the ball tracks the robot as it
    moves/rotates. When these are None they reuse the spawn offsets from
    reset_ball_state, so the reset config stays the single source of truth. Set
    follow_robot=False (or disable this whole event term) to turn following off in
    later stages.
    """
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
