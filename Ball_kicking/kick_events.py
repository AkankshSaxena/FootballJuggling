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
    """World-frame xy point offset from the robot in its own (heading-aligned) frame.

    distance_offset -> robot local +x (forward).
    lateral_offset  -> robot's RIGHT, i.e. the right-foot side, WHEN POSITIVE. The H1
        root frame is x-forward / y-LEFT / z-up, so a positive lateral_offset maps to
        local -y. This rotates with the robot's heading. Returns (N, 2). Shared by spawn
        (reset_ball_state) and the optional follow-constraint so both use identical
        offset math.

    distance_offset / lateral_offset may be a python float (uniform across env_ids) or
    a (N,) tensor (per-env values, e.g. randomized side sign) -- both broadcast fine
    into the local_offset column assignment below.
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
    distance_offset: float = 0.0,
    lateral_offset: float = 0.0,
    height_offset: float = 3.0,
    randomize_side: bool = False,
):
    """Reset ball pose/velocity and (re)anchor it in front of the robot.

    ACTIVE LEG: env.active_leg (0=right, 1=left) is set DIRECTLY here (no hysteresis)
    for every env in env_ids, derived from the sign of the spawn's robot-frame Y
    (== -lateral_offset_signed, by construction of _front_anchor_xy). This must be a
    direct assignment, not the hysteretic update used mid-episode in
    foot_swing_knee_extend -- a fresh spawn has no relationship to the previous
    episode's active leg, so hysteresis against stale state would bias it.

    randomize_side=True: the SIGN of lateral_offset is randomized per env in env_ids
    (magnitude unchanged). Same reset_ball_state call is reused for the Stage 1
    overhead-park spawn (height_offset=3.0-ish) and the Stage 2.1+ on-arc spawn
    (height_offset on the arc) -- only height_offset differs between those configs.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]
    n = len(env_ids)

    if randomize_side:
        sign = torch.where(
            torch.rand(n, device=env.device) < 0.5,
            torch.full((n,), -1.0, device=env.device),
            torch.full((n,), 1.0, device=env.device),
        )
        lateral_offset_signed = lateral_offset * sign
    else:
        lateral_offset_signed = torch.full(
            (n,), float(lateral_offset), device=env.device
        )

    anchor_xy = _front_anchor_xy(
        robot_pos, robot_quat, distance_offset, lateral_offset_signed, env.device
    )

    ball_pos = torch.zeros_like(robot_pos)
    ball_pos[:, :2] = anchor_xy
    ball_pos[:, 2] = height_offset

    # Persist the world-frame anchor (used when the follow-constraint is OFF) and the
    # per-env spawn offsets (reused by the follow-constraint when ON, and now
    # necessarily per-env tensors since side may be randomized) so both are driven
    # from this single reset config.
    if not hasattr(env, "ball_anchor_xy"):
        env.ball_anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env.ball_anchor_xy[env_ids] = anchor_xy

    if not hasattr(env, "ball_distance_offset"):
        env.ball_distance_offset = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "ball_lateral_offset"):
        env.ball_lateral_offset = torch.zeros(env.num_envs, device=env.device)
    env.ball_distance_offset[env_ids] = float(distance_offset)
    env.ball_lateral_offset[env_ids] = lateral_offset_signed

    # Active leg: direct assignment from spawn geometry (see docstring). Robot-frame
    # Y at spawn = -lateral_offset_signed -> Y>0 (left) iff lateral_offset_signed<0.
    if not hasattr(env, "active_leg"):
        env.active_leg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.active_leg[env_ids] = (lateral_offset_signed < 0).long()

    ball_vel = torch.zeros((n, 6), device=env.device)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0  # w=1 identity quaternion

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

    # Single write to the simulator (duplicate write removed -- original wrote the
    # same pose/velocity twice with identical values).
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
) -> None:
    """Pin the ball's xy so only Z (fall/bounce) is free.

    follow_robot=False (default): pin to the fixed spawn anchor -- the ball stays
    where it was reset even if the robot walks away.
    follow_robot=True: re-anchor `distance_offset` forward / `lateral_offset` to the
    right of the *current* robot pose every step, so the ball tracks the robot as it
    moves/rotates. When these are None they reuse the spawn offsets from
    reset_ball_state -- env.ball_distance_offset / env.ball_lateral_offset are now
    per-env (N,) tensors (side may be randomized per env), not shared scalars, so this
    correctly re-anchors each env to its own spawned side. Set follow_robot=False (or
    disable this whole event term) to turn following off in later stages.
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


def ball_gravity_scale(env, env_ids, k: float = 0.25, ball_cfg=SceneEntityCfg("ball")):
    """Upward force cancelling (1-k) of the ball's weight -> g_eff = k*9.81."""
    ball = env.scene[ball_cfg.name]
    m = ball.data.default_mass.to(env.device)  # (N,1)
    f = torch.zeros((env.num_envs, 1, 3), device=env.device)
    f[:, 0, 2] = m[:, 0] * 9.81 * (1.0 - k)
    ball.set_external_force_and_torque(f, torch.zeros_like(f))
    env.extras.setdefault("log", {})["debug/g_eff"] = k * 9.81
