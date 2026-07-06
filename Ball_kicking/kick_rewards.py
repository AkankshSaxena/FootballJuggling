from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_to_ball_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward tracking a target velocity that reaches the ball in 0.5s."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # Target velocity = (Ball_pos - Robot_pos) / 0.5s
    pos_diff = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    target_vel_xy = pos_diff / 0.5

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
) -> torch.Tensor:
    """Rewards proximity to the ball using exp(-0.1 * dist^2)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist_sq = torch.sum(
        torch.square(robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2]), dim=-1
    )
    return torch.exp(-0.1 * dist_sq)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward air time only if the foot has been in the air past the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


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


def ball_foot_contact_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=".*ankle_link"
    ),
) -> torch.Tensor:
    """
    Rewards any foot contact with the ball.
    Tracks alternating foot history natively in the env for later stage use.
    """
    foot_sensor: ContactSensor = env.scene[foot_sensor_cfg.name]
    foot_forces = foot_sensor.data.net_forces_w[:, :2, :]
    foot_force_mag = torch.norm(foot_forces, dim=-1)

    left_contact = foot_force_mag[:, 0] > 1.0
    right_contact = foot_force_mag[:, 1] > 1.0
    any_contact = left_contact | right_contact

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

    return any_contact.float()


def ball_vel_z_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Rewards specific Z velocity traits: exp(-1 / (1 + vel_z^2))."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_z_sq = torch.square(ball.data.root_lin_vel_w[:, 2])

    return torch.exp(-1.0 / (1.0 + ball_vel_z_sq))


def apex_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Triggers exactly once per flight when Z-velocity crosses from positive to negative.
    Rewards if the apex is between 1.5m and 2.5m (2m +- 0.5m).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)

    # Detect exactly when ball reaches apex (velocity flips from up to down)
    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_prev_vel_z = ball_vel_z.clone()

    # Band requirement: 1.5m to 2.5m
    within_bounds = (ball_pos_z >= 1.5) & (ball_pos_z <= 2.5)

    return (at_apex & within_bounds).float()


def track_ball_vel_xy_exp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Rewards minimizing the ball's horizontal velocity using exp(-norm(v_xy)^2)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_xy_sq = torch.sum(torch.square(ball.data.root_lin_vel_w[:, :2]), dim=-1)

    return torch.exp(-ball_vel_xy_sq)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])
