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
    lateral_offset: float | torch.Tensor,
    device,
) -> torch.Tensor:
    """World-frame xy point offset from the robot in its own (heading-aligned) frame."""
    local_offset = torch.zeros((robot_pos.shape[0], 3), device=device)
    local_offset[:, 0] = distance_offset
    local_offset[:, 1] = -lateral_offset  # +lateral = robot's right (local -y)
    world_offset = math_utils.quat_apply(math_utils.yaw_quat(robot_quat), local_offset)
    return robot_pos[:, :2] + world_offset[:, :2]


def _stage1_env_ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _set_leg_and_ball_side(env, env_ids, rl, lateral_offset: float = 0.3):
    """0 = left (+heading Y), 1 = right (-heading Y)."""
    if not hasattr(env, "leg_cmd"):
        env.leg_cmd = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not isinstance(getattr(env, "ball_lateral_offset", None), torch.Tensor):
        env.ball_lateral_offset = torch.zeros(env.num_envs, device=env.device)
    env.leg_cmd[env_ids] = rl
    env.ball_lateral_offset[env_ids] = (2.0 * rl.float() - 1.0) * abs(lateral_offset)


def reset_leg_cmd(env, env_ids, lateral_offset: float = 0.3):
    env_ids = _stage1_env_ids(env, env_ids)
    rl = torch.randint(0, 2, (len(env_ids),), device=env.device)
    _set_leg_and_ball_side(env, env_ids, rl, lateral_offset)
    if not hasattr(env, "leg_cycle_index"):
        env.leg_cycle_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.leg_cycle_index[env_ids] = 0


def resample_leg_cmd(
    env,
    env_ids,
    period: float = 1.2,
    lateral_offset: float = 0.3,
    ball_cfg=SceneEntityCfg("ball"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """Stage 1.2: sample once at each full swing-plus-pause boundary."""
    env_ids = _stage1_env_ids(env, env_ids)
    t = env.episode_length_buf[env_ids].float() * env.step_dt
    cycle = torch.floor(t / period + 1e-6).long()
    changed = cycle > env.leg_cycle_index[env_ids]
    changed_ids = env_ids[changed]
    if len(changed_ids) == 0:
        return
    env.leg_cycle_index[changed_ids] = cycle[changed]
    rl = torch.randint(0, 2, (len(changed_ids),), device=env.device)
    _set_leg_and_ball_side(env, changed_ids, rl, lateral_offset)

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    anchor_xy = _front_anchor_xy(
        robot.data.root_pos_w[changed_ids],
        robot.data.root_quat_w[changed_ids],
        env.ball_distance_offset,
        env.ball_lateral_offset[changed_ids],
        env.device,
    )
    env.ball_anchor_xy[changed_ids] = anchor_xy
    ball_pos = ball.data.root_pos_w[changed_ids].clone()
    ball_pos[:, :2] = anchor_xy
    ball_quat = ball.data.root_quat_w[changed_ids]
    ball_vel = torch.cat([
        ball.data.root_lin_vel_w[changed_ids],
        ball.data.root_ang_vel_w[changed_ids],
    ], dim=-1).clone()
    ball_vel[:, :2] = 0.0
    ball.write_root_pose_to_sim(torch.cat([ball_pos, ball_quat], dim=-1), env_ids=changed_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids=changed_ids)


def reset_ball_state(
    env,
    env_ids,
    ball_cfg=SceneEntityCfg("ball"),
    robot_cfg=SceneEntityCfg("robot"),
    distance_offset: float = 0.0,
    lateral_offset: float = 0.0,
    height_offset: float = 3.0,
):
    env_ids = _stage1_env_ids(env, env_ids)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    if hasattr(env, "leg_cmd"):
        lateral = env.ball_lateral_offset[env_ids]
    else:
        if not isinstance(getattr(env, "ball_lateral_offset", None), torch.Tensor):
            env.ball_lateral_offset = torch.zeros(env.num_envs, device=env.device)
        env.ball_lateral_offset[env_ids] = lateral_offset
        lateral = env.ball_lateral_offset[env_ids]
    anchor_xy = _front_anchor_xy(
        robot_pos, robot_quat, distance_offset, lateral, env.device
    )

    ball_pos = torch.zeros_like(robot_pos)
    ball_pos[:, :2] = anchor_xy
    ball_pos[:, 2] = height_offset
    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env.ball_anchor_xy[env_ids] = anchor_xy
    env.ball_distance_offset = distance_offset

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



def constrain_ball_to_z_axis(
    env,
    env_ids,
    ball_cfg=SceneEntityCfg("ball"),
    robot_cfg=SceneEntityCfg("robot"),
    min_height: float = 3.0,
    follow_robot: bool = True,
    distance_offset: float | None = None,
    lateral_offset: float | torch.Tensor | None = None,
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
