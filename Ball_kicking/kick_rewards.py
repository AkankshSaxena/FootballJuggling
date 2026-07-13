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
    return env.reset_terminated.float()


def track_lin_vel_xy_to_ball_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward tracking a target velocity that closes the gap to the ball over
    a 2s horizon: target_vel = (ball_pos - robot_pos) / 2.0.

    This intentionally scales with distance: far away, target velocity is
    large (robot covers ground quickly); close in, target velocity shrinks
    toward zero (robot slows/stops approaching). This is deliberate design,
    not a bug - do not change the divisor without replacing this mechanism
    with another way to get the same quick-approach/slow-down behavior.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    pos_diff = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    target_vel_xy = pos_diff / 2.0

    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]
    lin_vel_error = torch.sum(torch.square(target_vel_xy - robot_vel_xy), dim=1)

    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular velocity (yaw) to encourage straight-facing posture."""
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel_error = torch.square(robot.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def ball_robot_dist_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    kick_range: float = 0.4,
    std: float = 0.3,
) -> torch.Tensor:
    """Rewards standing at kicking range from the ball, not on top of it.

    Peaks at dist == kick_range and falls off in both directions. Old version
    peaked at dist == 0, which rewards standing on the ball (attempt #1
    reward-hacking behavior).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/robot_ball_dist"] = dist.mean().item()

    return torch.clamp(torch.exp(-torch.square(dist - kick_range) / std**2), max=0.99)


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Single-stance-gated air time reward. Only pays out when exactly one
    foot is in contact, so a synchronized two-foot hop earns nothing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(
            single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)
        ),
        dim=1,
    )[0]
    return torch.clamp(reward, max=threshold)


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot velocity when the foot is in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def _filtered_contact_force_mag(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Max force magnitude (over sensor history) for a SINGLE-BODY, ball-
    filtered contact sensor.

    IMPORTANT: assumes sensor_cfg.name points to a ContactSensor whose
    prim_path resolves to exactly one body per env (e.g. left_ankle_link
    only), with filter_prim_paths_expr targeting the ball. PhysX contact
    filter pairs are unreliable when prim_path is a multi-body regex - this
    was the root cause of force_matrix_w reading all zero despite visible
    contact. Requires kick_env_cfg.py to define one ContactSensorCfg per
    tracked body instead of one regex covering pelvis/torso/knees/ankles.

    Returns: (num_envs,) tensor.
    """
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
    """Sparse, outcome-gated kick reward.

    Fires once per NEW foot-ball contact (rising edge, not held-contact) that
    results in the ball moving upward past min_ball_vel_z. This replaces the
    old per-step "any contact" reward, which paid for dribbling/pinning the
    ball against the foot and for shin-flailing that touched the ball without
    launching it (attempt #2 reward-hacking behavior).

    Also maintains:
      - env.last_contact_foot: (num_envs, 2) one-hot of which foot last
        touched, used by the last_contact_foot observation term.
      - env.contact_count: (num_envs,) running count of SCORED contacts,
        used by curriculum success gates.
    """
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
    """Penalizes ball contact on any non-foot body.

    Pass one SceneEntityCfg per single-body filtered contact sensor (e.g.
    pelvis, torso_link, left_knee_link, right_knee_link) - same multi-body
    filter-pair caveat as _filtered_contact_force_mag applies here.
    """
    illegal_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in illegal_sensor_cfgs:
        force = _filtered_contact_force_mag(env, cfg)
        illegal_contact = illegal_contact | (force > force_threshold)
    return illegal_contact.float()


# COMMENTED OUT (per review 6.4.2) - direction-blind: rewards any large
# |vel_z| including the ball falling fast, and exp(-1/vel_z^2) is a fragile
# shape (1/0 -> inf at vel_z==0; works numerically in torch but by accident,
# not by design). The min_ball_vel_z gate inside ball_foot_contact_reward
# now covers "did the kick send the ball upward". Kept here for reference.
#
# def ball_vel_z_reward(
#     env: ManagerBasedRLEnv,
#     ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
# ) -> torch.Tensor:
#     """Rewards specific Z velocity traits: exp(-1 / (1 + vel_z^2))."""
#     ball: RigidObject = env.scene[ball_cfg.name]
#     ball_vel_z_sq = torch.square(ball.data.root_lin_vel_w[:, 2])
#     ball_vel_z_sq = torch.clamp(ball_vel_z_sq, max=50.0)
#     return torch.exp(-1.0 / (ball_vel_z_sq))


def apex_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Triggers exactly once per flight when Z-velocity crosses from positive to
    negative. Rewards if the apex is between 1.5m and 2.5m.

    NOTE: band intentionally left at 1.5-2.5m (per review 6.4.1) rather than
    the 0.8-1.6m locked physics constant - do not change without explicit
    sign-off, since current kick-force rewards are tuned around this band.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    ball_vel_xy = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["debug/ball_vel_xy"] = ball_vel_xy.mean().item()

    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)

    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_prev_vel_z = ball_vel_z.clone()

    within_bounds = (ball_pos_z >= 1.5) & (ball_pos_z <= 2.5)

    return (at_apex & within_bounds).float()


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalizes the robot for high linear velocity in the z direction."""
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def foot_front_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_link", "right_ankle_link"]
    ),
    leg_length: list[float] = [0.9, 1.2],  # Min and Max length bounds
    leg_angle: list[float] = [15.0, 45.0],  # Min and Max angle bounds in degrees
    dropoff_factor: float = 4.0,
) -> torch.Tensor:
    """
    Rewards the robot when a foot is strictly in front of its base and at a
    specific height. Uses dynamic Gaussian distributions that drop to near-
    zero at the kinematic limits.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    angles_rad = torch.tensor(leg_angle, device=env.device) * (math.pi / 180.0)
    lengths = torch.tensor(leg_length, device=env.device)

    target_height_min = lengths[0] * torch.sin(angles_rad[0])
    target_height_max = lengths[1] * torch.sin(angles_rad[1])

    target_len_min = lengths[0] * (1.0 - torch.cos(angles_rad[0]))
    target_len_max = lengths[1] * (1.0 - torch.cos(angles_rad[1]))

    ideal_height = (target_height_max + target_height_min) / 2.0
    ideal_length = (target_len_max + target_len_min) / 2.0

    std_height = torch.clamp(
        (target_height_max - target_height_min) / dropoff_factor, min=1e-4
    )
    std_length = torch.clamp(
        (target_len_max - target_len_min) / dropoff_factor, min=1e-4
    )

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = robot.data.root_pos_w

    foot_z = foot_pos_w[..., 2]
    height_reward = torch.exp(-torch.square(foot_z - ideal_height) / (std_height**2))

    root_to_foot_w = foot_pos_w - root_pos_w.unsqueeze(1)

    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)
    yaw_quat_expanded = yaw_quat.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
    foot_pos_b = math_utils.quat_apply_inverse(yaw_quat_expanded, root_to_foot_w)

    foot_x = foot_pos_b[..., 0]
    length_reward = torch.exp(-torch.square(foot_x - ideal_length) / (std_length**2))
    front_mask = (foot_x > 0.05).float()

    combined_reward = height_reward * length_reward * front_mask
    reward = torch.sum(combined_reward, dim=1)
    return reward
