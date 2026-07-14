from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Flat penalty on any non-timeout termination (fall / out-of-bounds)."""
    return env.reset_terminated.float()


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward zero base XY velocity (stationary juggle; no pull toward ball)."""
    robot: Articulation = env.scene[robot_cfg.name]
    v_xy = robot.data.root_lin_vel_w[:, :2]
    err = torch.sum(torch.square(v_xy), dim=1)
    return torch.exp(-err / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward zero yaw rate (straight-facing posture)."""
    robot: Articulation = env.scene[robot_cfg.name]
    err = torch.square(robot.data.root_ang_vel_w[:, 2])
    return torch.exp(-err / std**2)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize vertical base velocity (anti-bounce)."""
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot XY velocity while that foot is in ground contact."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = (
        cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


# STABILITY
def ball_robot_dist_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    kick_range: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward standing at kick_range from the ball (peaks at kick_range, not 0)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/robot_ball_dist"] = dist.mean().item()
    return torch.clamp(torch.exp(-torch.square(dist - kick_range) / std**2), max=0.60)


def one_foot_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward having >=1 foot on the ground (anti-hop; NOT a stability guarantee)."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = (
        cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    return (forces > force_threshold).any(dim=1).float()


def foot_trajectory_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_ankle_link"]
    ),
    period: float = 1.0,
    reach_x: float = 0.30,
    lift_z: float = 0.30,
    base_z: float = -0.90,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward the swing foot for tracking a phase-indexed forward-up arc.

    phi = (t mod period)/period; target in base/yaw frame is
    x = reach_x*s, z = base_z + lift_z*s with s = (1-cos(2*pi*phi))/2.
    Policy MUST observe sin/cos(2*pi*phi) or this target is unlearnable.
    Defaults (reach_x/lift_z/base_z) are guesses -- tune from track/ logs.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t, period) / period
    s = (1.0 - torch.cos(2.0 * math.pi * phase)) / 2.0

    x_target = reach_x * s
    z_target = base_z + lift_z * s

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_to_foot_w = foot_pos_w - robot.data.root_pos_w.unsqueeze(1)
    yaw_quat = (
        math_utils.yaw_quat(robot.data.root_quat_w)
        .unsqueeze(1)
        .expand(-1, foot_pos_w.shape[1], -1)
    )
    foot_pos_b = math_utils.quat_apply_inverse(yaw_quat, root_to_foot_w)

    foot_x = foot_pos_b[:, 0, 0]
    foot_z = foot_pos_b[:, 0, 2]
    err = torch.square(foot_x - x_target) + torch.square(foot_z - z_target)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/swing_phase"] = phase.mean().item()
    env.extras["log"]["debug/swing_x_target"] = x_target.mean().item()
    env.extras["log"]["debug/swing_foot_x"] = foot_x.mean().item()
    env.extras["log"]["debug/swing_z_target"] = z_target.mean().item()
    env.extras["log"]["debug/swing_foot_z"] = foot_z.mean().item()
    return torch.exp(-err / std**2)


# JUGGLING
def _filtered_contact_force_mag(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Max ball-filtered contact force for a SINGLE-BODY sensor (force_matrix_w)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.force_matrix_w_history  # (N, history, 1, 1, 3)
    return torch.norm(forces, dim=-1).sum(dim=-1).max(dim=1)[0].squeeze(-1)


def ball_foot_contact_reward(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.1,
    min_ball_vel_z: float = 1.0,
) -> torch.Tensor:
    """Sparse rising-edge kick reward, gated on ball moving upward past min_ball_vel_z."""
    left_force = _filtered_contact_force_mag(env, left_sensor_cfg)
    right_force = _filtered_contact_force_mag(env, right_sensor_cfg)

    left_contact = left_force > force_threshold
    right_contact = right_force > force_threshold
    any_contact = left_contact | right_contact

    if not hasattr(env, "prev_ball_contact"):
        env.prev_ball_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    new_contact = any_contact & (~env.prev_ball_contact)
    env.prev_ball_contact = any_contact.clone()

    current_contact_foot = torch.stack(
        [left_contact.float(), right_contact.float()], dim=1
    )
    if not hasattr(env, "last_contact_foot"):
        env.last_contact_foot = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )
    env.last_contact_foot = torch.where(
        any_contact.unsqueeze(-1).expand_as(current_contact_foot),
        current_contact_foot,
        env.last_contact_foot,
    )

    ball: RigidObject = env.scene["ball"]
    ball_going_up = ball.data.root_lin_vel_w[:, 2] > min_ball_vel_z
    scored = new_contact & ball_going_up

    if not hasattr(env, "contact_count"):
        env.contact_count = torch.zeros(env.num_envs, device=env.device)
    env.contact_count += scored.float()

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/new_ball_contacts"] = new_contact.float().sum().item()
    env.extras["log"]["debug/scored_kicks"] = scored.float().sum().item()
    return scored.float()


def ball_illegal_contact_penalty(
    env: ManagerBasedRLEnv,
    illegal_sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize ball contact on any non-foot body (pass one single-body sensor each)."""
    illegal_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in illegal_sensor_cfgs:
        force = _filtered_contact_force_mag(env, cfg)
        illegal_contact = illegal_contact | (force > force_threshold)
    return illegal_contact.float()


def apex_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    apex_min: float = 0.8,
    apex_max: float = 1.6,
) -> torch.Tensor:
    """Fire once per flight at the Z-velocity sign flip if apex is within band."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/ball_vel_xy"] = (
        torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1).mean().item()
    )

    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)
    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_prev_vel_z = ball_vel_z.clone()

    within_bounds = (ball_pos_z >= apex_min) & (ball_pos_z <= apex_max)
    return (at_apex & within_bounds).float()


# Logging
def log_tracking(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=[
            "left_ankle_link",
            "right_ankle_link",
            "left_knee_link",
            "right_knee_link",
        ],
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Log base-frame foot/knee/ball positions (for ball placement; no gradient)."""
    robot: Articulation = env.scene[robot_cfg.name]
    names = robot.data.body_names
    ids = robot_cfg.body_ids

    pos_w = robot.data.body_pos_w[:, ids, :]
    root_to = pos_w - robot.data.root_pos_w.unsqueeze(1)
    yaw = (
        math_utils.yaw_quat(robot.data.root_quat_w)
        .unsqueeze(1)
        .expand(-1, pos_w.shape[1], -1)
    )
    pos_b = math_utils.quat_apply_inverse(yaw, root_to)

    log = env.extras.setdefault("log", {})
    for i, bid in enumerate(ids):
        nm = names[bid]
        log[f"track/{nm}_x"] = pos_b[:, i, 0].mean().item()
        log[f"track/{nm}_z"] = pos_b[:, i, 2].mean().item()

    ball: RigidObject = env.scene[ball_cfg.name]
    rel = ball.data.root_pos_w - robot.data.root_pos_w
    ball_b = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(robot.data.root_quat_w), rel
    )
    log["track/ball_x"] = ball_b[:, 0].mean().item()
    log["track/ball_y"] = ball_b[:, 1].mean().item()
    log["track/ball_z"] = ball_b[:, 2].mean().item()
    return torch.zeros(env.num_envs, device=env.device)


def log_termination_causes(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Log per-cause termination fractions and torso height (no gradient)."""
    log = env.extras.setdefault("log", {})
    fell = env.termination_manager.get_term("robot_falls")
    timeout = env.termination_manager.get_term("time_out")
    oob = env.termination_manager.get_term("robot_out_of_bounds")
    n_term = env.termination_manager.terminated.sum().float().clamp(min=1.0)
    log["debug/term_frac_falls"] = (fell.float().sum() / n_term).item()
    log["debug/term_frac_timeout"] = (timeout.float().sum() / n_term).item()
    log["debug/term_frac_oob"] = (oob.float().sum() / n_term).item()

    h = env.scene["robot"].data.root_pos_w[:, 2]
    log["debug/torso_height_mean"] = h.mean().item()
    log["debug/torso_height_min"] = h.min().item()
    return torch.zeros(env.num_envs, device=env.device)
