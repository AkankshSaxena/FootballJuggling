from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import kick_swing

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def ball_robot_dist_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    kick_range: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward standing at kick_range from the ball (peaks at kick_range)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )
    log = env.extras.setdefault("log", {})
    log["debug/robot_ball_dist"] = dist.mean().item()
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


def foot_swing_knee_extend(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["right_hip_pitch_link", "right_ankle_link"],
        preserve_order=True,
    ),
    h: float = 0.55,  # hip->ankle, knee bent (spawn pose) — measure
    h_prime: float = 0.80,  # hip->ankle, knee straight — measure
    theta_max_deg: float = 60.0,
    swing_time: float = 0.8,  # FULL cycle: 0 -> 60 -> 0
    period: float = 0.8,  # set > swing_time to insert a rest (foot down) between kicks
    std: float = 0.15,
) -> torch.Tensor:
    """Two-phase kick-swing target, triangular in theta (up and back in swing_time).

    theta(t): triangle wave 0 -> theta_max -> 0 over swing_time, then held at 0
    until the next period boundary (rest phase if period > swing_time).

    Phase 1 (theta <= acos(h/h')): knee extends/re-bends, ankle drags at depth h.
        x = h * tan(theta),   z = -h
    Phase 2: straight-leg pendulum of length h'.
        x = h' * sin(theta),  z = -h' * cos(theta)
    y target = 0 throughout.

    NOTE: swing_x/z_actual logged here are HIP-relative — do not compare them
    with the ROOT-relative positions logged by log_kinematics.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # --- triangular phase variable ---
    t = env.episode_length_buf.float() * env.step_dt
    theta_max = math.radians(theta_max_deg)
    t_cycle = torch.remainder(t, period)
    p = torch.clamp(t_cycle / swing_time, max=1.0)  # 0..1 during swing, 1 during rest
    tri = 1.0 - torch.abs(2.0 * p - 1.0)  # 0 -> 1 -> 0, stays 0 in rest
    theta = kick_swing.swing_theta(env, theta_max_deg, swing_time, period)

    # --- piecewise target (hip-relative, yaw frame) ---
    theta_c = math.acos(h / h_prime)
    phase1 = theta <= theta_c

    x_target = torch.where(phase1, h * torch.tan(theta), h_prime * torch.sin(theta))
    z_target = torch.where(
        phase1, torch.full_like(theta, -h), -h_prime * torch.cos(theta)
    )

    # --- actual ankle position relative to hip pitch link, yaw frame ---
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    hip_pos_w = body_pos_w[:, 0, :]
    ankle_pos_w = body_pos_w[:, 1, :]

    rel_w = ankle_pos_w - hip_pos_w
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)
    rel_b = math_utils.quat_apply_inverse(yaw_quat, rel_w)

    err = (
        torch.square(rel_b[:, 0] - x_target)
        + torch.square(rel_b[:, 1])  # y = 0 constraint
        + torch.square(rel_b[:, 2] - z_target)
    )

    log = env.extras.setdefault("log", {})
    log["debug/swing_theta_deg"] = math.degrees(theta.mean().item())
    log["debug/swing_x_target"] = x_target.mean().item()
    log["debug/swing_x_actual"] = rel_b[:, 0].mean().item()
    log["debug/swing_z_target"] = z_target.mean().item()
    log["debug/swing_z_actual"] = rel_b[:, 2].mean().item()
    log["debug/swing_y_actual"] = rel_b[:, 1].mean().item()

    return torch.exp(-err / std**2)


# ---------------------------------------------------------------------------
# Juggling
# ---------------------------------------------------------------------------
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

    log = env.extras.setdefault("log", {})
    log["debug/new_ball_contacts"] = new_contact.float().sum().item()
    log["debug/scored_kicks"] = scored.float().sum().item()
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

    log = env.extras.setdefault("log", {})
    log["debug/ball_vel_xy"] = (
        torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1).mean().item()
    )

    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)
    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_prev_vel_z = ball_vel_z.clone()

    within_bounds = (ball_pos_z >= apex_min) & (ball_pos_z <= apex_max)
    return (at_apex & within_bounds).float()


# ---------------------------------------------------------------------------
# Logging (zero-weight term — writes to env.extras["log"], returns zeros)
# ---------------------------------------------------------------------------
def log_kinematics(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=[
            "left_ankle_link",
            "right_ankle_link",
            "left_knee_link",
            "right_knee_link",
            "right_hip_pitch_link",  # kicking-leg hip motor — verify exact name
        ],
        preserve_order=True,  # keep body_ids aligned to body_names order (see channel_names loop)
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Consolidated kinematics logging. Register with weight=0.0.

    Channels (all under debug/):
      ball_x/y/z            — ball pos, ROOT-relative, yaw-aligned base frame
      left_foot_x/y/z       — left ankle, root-relative yaw frame
      right_foot_x/y/z      — right ankle, root-relative yaw frame
      left_knee_x/y/z       — left knee, root-relative yaw frame
      right_knee_x/y/z      — right knee, root-relative yaw frame
      hip_z                 — kicking-leg hip pitch link, WORLD frame height
                              (monitors the "hip stays at h" assumption behind
                              the phase-1 swing target z = -h)

    Frame note: foot_swing_knee_extend logs HIP-relative actuals; these are
    ROOT-relative. Not directly comparable.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ids = robot_cfg.body_ids  # order follows body_names order above

    # Guard the "ids order == body_names order" assumption the channel labels and
    # the hip_z index=4 lookup below rely on. Requires preserve_order=True on the
    # SceneEntityCfg; without it find_bodies() returns URDF-sorted indices and the
    # panels get silently mislabeled. Runs once (first call) to avoid per-step cost.
    if not getattr(env, "_log_kinematics_order_checked", False):
        expected = [
            "left_ankle_link",
            "right_ankle_link",
            "left_knee_link",
            "right_knee_link",
            "right_hip_pitch_link",
        ]
        resolved = [robot.data.body_names[i] for i in ids]
        assert resolved == expected, (
            "log_kinematics body order mismatch — set preserve_order=True on the "
            f"robot_cfg. expected {expected}, got {resolved}"
        )
        env._log_kinematics_order_checked = True

    # --- transform all tracked bodies into root-relative yaw frame ---
    pos_w = robot.data.body_pos_w[:, ids, :]  # (N, 5, 3)
    root_to = pos_w - robot.data.root_pos_w.unsqueeze(1)
    yaw = (
        math_utils.yaw_quat(robot.data.root_quat_w)
        .unsqueeze(1)
        .expand(-1, pos_w.shape[1], -1)
    )
    pos_b = math_utils.quat_apply_inverse(yaw, root_to)  # (N, 5, 3)

    log = env.extras.setdefault("log", {})

    # index i matches body_names order in robot_cfg
    channel_names = ["left_foot", "right_foot", "left_knee", "right_knee"]
    for i, nm in enumerate(channel_names):
        log[f"debug/{nm}_x"] = pos_b[:, i, 0].mean().item()
        log[f"debug/{nm}_y"] = pos_b[:, i, 1].mean().item()
        log[f"debug/{nm}_z"] = pos_b[:, i, 2].mean().item()

    # hip motor: world-frame height (index 4 = right_hip_pitch_link)
    log["debug/hip_z"] = pos_w[:, 4, 2].mean().item()

    # --- ball, root-relative yaw frame ---
    ball: RigidObject = env.scene[ball_cfg.name]
    rel = ball.data.root_pos_w - robot.data.root_pos_w
    ball_b = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(robot.data.root_quat_w), rel
    )
    log["debug/ball_x"] = ball_b[:, 0].mean().item()
    log["debug/ball_y"] = ball_b[:, 1].mean().item()
    log["debug/ball_z"] = ball_b[:, 2].mean().item()

    return torch.zeros(env.num_envs, device=env.device)
