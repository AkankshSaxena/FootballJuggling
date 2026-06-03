import torch
import math
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg


def leg_raise_and_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_height: float = 0.1,
    max_height: float = 0.3,
    min_time: float = 0.2,
    max_time: float = 0.5,
) -> torch.Tensor:
    """
    Rewards maintaining the ankle above ground between 0.1m and 0.3m
    for an air time of 0.2s to 0.5s.
    """
    # Assuming sensor_cfg points to the foot/ankle bodies
    foot_height = env.scene[sensor_cfg.name].data.root_pos_w[:, 2]
    air_time = getattr(env, "feet_air_time", torch.zeros_like(foot_height))

    height_mask = (foot_height >= min_height) & (foot_height <= max_height)
    time_mask = (air_time >= min_time) & (air_time <= max_time)

    return (height_mask & time_mask).float()


def track_ball_velocity_xy_exp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.5,
) -> torch.Tensor:
    """Penalizes the ball's motion in the XY direction."""
    ball_vel_xy = env.scene[ball_cfg.name].data.root_lin_vel_w[:, :2]
    return torch.exp(-torch.sum(torch.square(ball_vel_xy), dim=1) / std**2)


def ball_xy_drift_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Grows linearly with ||ball_pos_xy - spawn_pos_xy||.
    Approximates spawn_pos_xy based on robot root + stage distance.
    """
    ball_pos_xy = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    robot_pos_xy = env.scene[robot_cfg.name].data.root_pos_w[:, :2]

    dist_val = getattr(env, "stage6_distance", 0.15)
    spawn_pos_xy = robot_pos_xy.clone()
    spawn_pos_xy[:, 0] += dist_val  # Assuming +X forward spawn

    drift = torch.norm(ball_pos_xy - spawn_pos_xy, dim=1)
    return -drift  # Linear penalty


def target_impulse_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_apex_height: float = 1.0,
    mass_ball: float = 0.43,  # standard football mass
    gravity: float = 9.81,
    tolerance: float = 0.40,  # 40% tolerance for stages 3-7
) -> torch.Tensor:
    """
    Rewards impacts where the impulse on the ball is within the range:
    2.5 * M_B * sqrt(2 * G * (H_B - H_F)) +- tolerance
    """
    ball_vel = env.scene[ball_cfg.name].data.root_lin_vel_w[:, 2]
    ball_pos_z = env.scene[ball_cfg.name].data.root_pos_w[:, 2]

    # Calculate target upward velocity (v = sqrt(2*g*h))
    # Using upward velocity directly simplifies impulse calculation over mass
    target_vel_z = torch.sqrt(
        2 * gravity * torch.clamp(target_apex_height - ball_pos_z, min=0.01)
    )
    target_impulse = 2.5 * mass_ball * target_vel_z

    # Actual calculated vertical impulse based on velocity change (approximation post-impact)
    actual_impulse = mass_ball * ball_vel

    # Reward 1.0 if within tolerance band, 0.0 otherwise
    lower_bound = target_impulse * (1.0 - tolerance)
    upper_bound = target_impulse * (1.0 + tolerance)

    in_band = (actual_impulse >= lower_bound) & (actual_impulse <= upper_bound)
    return in_band.float()


def ball_apex_height_reward(
    env: ManagerBasedRLEnv, target_height: float = 1.0, tolerance: float = 0.2
) -> torch.Tensor:
    """Rewards when the ball's apex height reaches 1m +- 0.2m."""
    apex_height = getattr(
        env, "ball_apex_height", torch.zeros(env.num_envs, device=env.device)
    )

    lower_bound = target_height - tolerance
    upper_bound = target_height + tolerance

    in_band = (apex_height >= lower_bound) & (apex_height <= upper_bound)
    return in_band.float()


def alternate_foot_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Rewards when an alternating foot touches the ball.
    Relies on env tracking `current_contact_foot` and `last_contact_foot`.
    """
    current_foot = getattr(
        env, "current_contact_foot", torch.zeros(env.num_envs, device=env.device)
    )
    last_foot = getattr(
        env, "last_contact_foot", torch.zeros(env.num_envs, device=env.device)
    )

    # Assuming 1 for Left, 2 for Right, 0 for None
    valid_contact = current_foot > 0
    alternated = current_foot != last_foot

    return (valid_contact & alternated).float()


def juggle_streak_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    juggle_streak_bonus = streak_count^2 * 1.5
    """
    streak_count = getattr(
        env, "juggle_streak_count", torch.zeros(env.num_envs, device=env.device)
    )
    return (streak_count**2) * 1.5
