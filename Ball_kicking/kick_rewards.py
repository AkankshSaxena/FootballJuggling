from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_GRAVITY: float = 9.81
_BALL_MASS: float = 0.43
_FOOT_HEIGHT_CONST: float = 0.1


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalise vertical linear velocity of the robot base (should be 0).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward zero XY base linear velocity (robot should stay still).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_xy = robot.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel_xy), dim=1) / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward zero yaw angular velocity (robot should not spin).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel_z = robot.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(ang_vel_z) / std**2)


def track_lin_vel_ball_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward low XY velocity of the ball (ball should stay above robot, not drift).

    Stage 1: ball is underground — vel is zero, reward always 1.0 (harmless).

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    return torch.exp(-torch.sum(torch.square(ball_vel_xy), dim=1) / std**2)


def track_lin_vel_ball_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward low Z velocity of the ball (used as penalty in Stages 2 & 3+).

    Note: weight is set to 0 in stages where Z motion is intentional.

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]
    return torch.exp(-torch.square(ball_vel_z) / std**2)


def dof_pos_limits(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise joints that exceed soft position limits (identical to locomotion).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    out_of_limits = -(
        robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 0]
    ).clamp(max=0.0)
    out_of_limits += (
        robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 1]
    ).clamp(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_hip(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"]
    ),
) -> torch.Tensor:
    """Penalise hip yaw/roll deviation from zero (same as H1 locomotion).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos), dim=1)


def joint_deviation_arms(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            ".*_shoulder_pitch",
            ".*_shoulder_roll",
            ".*_shoulder_yaw",
            ".*_elbow",
        ],
    ),
) -> torch.Tensor:
    """Penalise arm joint deviation from default pose (same as H1 locomotion).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos - default_pos), dim=1)


def joint_deviation_torso(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["torso_joint"]),
) -> torch.Tensor:
    """Penalise torso joint deviation from default pose (same as H1 locomotion).

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos - default_pos), dim=1)


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def leg_raise_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_link", "right_ankle_link"]
    ),
    min_height: float = 0.1,
    max_height: float = 0.3,
    min_time: float = 0.05,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Reward each ankle that is held in the height band [min_height, max_height]
    for a duration in [min_time, max_time] seconds.

    Uses ``env.leg_raise_timer`` (num_envs, 2) which is incremented each step
    while the ankle is in the height band and reset when it leaves.

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    feet_pos_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]  # (num_envs, 2)

    in_height_band = (feet_pos_z >= min_height) & (feet_pos_z <= max_height)

    # Initialize timer and counter if they don't exist
    if not hasattr(env, "leg_raise_timer"):
        env.leg_raise_timer = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "leg_raise_counts"):
        env.leg_raise_counts = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    # Store previous timer state to detect the exact moment it crosses min_time
    prev_timer = env.leg_raise_timer.clone()

    env.leg_raise_timer = torch.where(
        in_height_band,
        env.leg_raise_timer + env.step_dt,
        torch.zeros_like(env.leg_raise_timer),
    )

    in_time_band = (env.leg_raise_timer >= min_time) & (env.leg_raise_timer <= max_time)
    # Increment count for any foot that just crossed the min_time threshold this timestep
    just_reached_min_time = (prev_timer < min_time) & (env.leg_raise_timer >= min_time)
    env.leg_raise_counts += just_reached_min_time.long().sum(dim=1)
    # Reward = number of feet satisfying both conditions (0, 1, or 2)
    return (in_height_band & in_time_band).float().sum(dim=1)


def ball_foot_contact_reward(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg,
    ball_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward each detected ball–foot contact event.

    Also updates ``env.last_contact_foot``, ``env.juggle_streak_buf``,
    and ``env.contact_count`` for use by other reward terms and the
    curriculum manager.

    Contact is detected when the foot contact sensor reports a force above
    threshold on the timestep a foot is near the ball.

    Returns:
        Tensor of shape (num_envs,).
    """
    foot_sensor: ContactSensor = env.scene[foot_sensor_cfg.name]
    ball_sensor: ContactSensor = env.scene[ball_sensor_cfg.name]

    # Net force on each foot body this step: (num_envs, 2, 3)
    # Sensor bodies order: [left_ankle, right_ankle]
    foot_forces = foot_sensor.data.net_forces_w[:, :2, :]  # (num_envs, 2, 3)
    foot_force_mag = torch.norm(foot_forces, dim=-1)  # (num_envs, 2)

    # Ball sensor net force: (num_envs, 1, 3) — contact with anything
    ball_forces = ball_sensor.data.net_forces_w[:, 0, :]  # (num_envs, 3)
    ball_force_mag = torch.norm(ball_forces, dim=-1)  # (num_envs,)

    FORCE_THRESH = 1.0  # N

    # A contact with the ball occurs when both the foot AND ball sensors fire
    ball_contacted = ball_force_mag > FORCE_THRESH  # (num_envs,)
    left_contact = (foot_force_mag[:, 0] > FORCE_THRESH) & ball_contacted
    right_contact = (foot_force_mag[:, 1] > FORCE_THRESH) & ball_contacted

    any_contact = left_contact | right_contact  # (num_envs,)

    # --- Update tracking buffers ---
    if not hasattr(env, "last_contact_foot"):
        env.last_contact_foot = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "juggle_streak_buf"):
        env.juggle_streak_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "contact_count"):
        env.contact_count = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    # Current contact one-hot: left=[1,0], right=[0,1]
    current_contact_foot = torch.stack(
        [left_contact.float(), right_contact.float()], dim=1
    )

    # Streak: increments if the contacting foot differs from the last
    prev_was_left = env.last_contact_foot[:, 0] > 0.5
    prev_was_right = env.last_contact_foot[:, 1] > 0.5
    alternated = (left_contact & prev_was_right) | (right_contact & prev_was_left)

    env.juggle_streak_buf = torch.where(
        any_contact & alternated,
        env.juggle_streak_buf + 1,
        torch.where(
            any_contact, torch.zeros_like(env.juggle_streak_buf), env.juggle_streak_buf
        ),
    )
    env.contact_count += any_contact.long()

    # Update last contact foot only when a contact actually occurs
    env.last_contact_foot = torch.where(
        any_contact.unsqueeze(-1).expand_as(current_contact_foot),
        current_contact_foot,
        env.last_contact_foot,
    )

    return any_contact.float()


def target_impulse_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact_sensor"),
    target_apex_height: float = 1.0,
    mass_ball: float = _BALL_MASS,
    gravity: float = _GRAVITY,
    foot_height: float = _FOOT_HEIGHT_CONST,
    tolerance: float = 0.40,
) -> torch.Tensor:
    """Reward impacts where the delivered impulse is within the target band.

    Target impulse = 2.5 * M_B * sqrt(2 * G * (H_B - H_F))
    where H_F is the fixed foot-height constant (0.1 m).

    Tolerance is ± tolerance fraction (0.20 for Stage 2, 0.40 for Stages 3–7).

    Only fires on the step a foot–ball contact is detected.

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    foot_sensor: ContactSensor = env.scene[foot_sensor_cfg.name]

    # Detect contact this step
    foot_forces = foot_sensor.data.net_forces_w[:, :2, :]
    foot_force_mag = torch.norm(foot_forces, dim=-1)
    contact_this_step = foot_force_mag.sum(dim=1) > 1.0  # (num_envs,)

    ball_pos_z = ball.data.root_pos_w[:, 2]  # (num_envs,)
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]  # (num_envs,)

    # Target upward velocity to reach apex
    h_diff = torch.clamp(
        torch.tensor(target_apex_height, device=env.device)
        - torch.tensor(foot_height, device=env.device),
        min=0.01,
    )
    target_vel_z = math.sqrt(2.0 * gravity * float(h_diff))

    target_impulse = 2.5 * mass_ball * target_vel_z  # scalar
    actual_impulse = mass_ball * ball_vel_z  # (num_envs,)

    lower = target_impulse * (1.0 - tolerance)
    upper = target_impulse * (1.0 + tolerance)

    in_band = (actual_impulse >= lower) & (actual_impulse <= upper)
    return (contact_this_step & in_band).float()


def ball_apex_height_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_height: float = 1.0,
    tolerance: float = 0.2,
) -> torch.Tensor:
    """Reward when the ball's apex this episode reaches target_height ± tolerance.

    Apex is tracked via ``env.ball_apex_height`` which is updated here by
    comparing current ball Z velocity sign change (peak detection).

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_z = ball.data.root_pos_w[:, 2]
    ball_vel_z = ball.data.root_lin_vel_w[:, 2]

    # Initialise buffers if needed
    if not hasattr(env, "ball_apex_height"):
        env.ball_apex_height = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "ball_prev_vel_z"):
        env.ball_prev_vel_z = torch.zeros(env.num_envs, device=env.device)

    # Apex detected when velocity crosses from positive to negative
    at_apex = (env.ball_prev_vel_z > 0.0) & (ball_vel_z <= 0.0)
    env.ball_apex_height = torch.where(at_apex, ball_pos_z, env.ball_apex_height)
    env.ball_prev_vel_z = ball_vel_z.clone()

    in_band = (env.ball_apex_height >= target_height - tolerance) & (
        env.ball_apex_height <= target_height + tolerance
    )
    return (at_apex & in_band).float()


def ball_xy_velocity_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalise ball motion in the XY plane (ball should go straight up).

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    return torch.sum(torch.square(ball_vel_xy), dim=1)


def ball_xy_drift_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise distance of ball XY from its spawn position (grows linearly).

    Spawn position is approximated as robot_pos_xy + stage6_spawn_dist along X.
    Used in Stage 6 where spawn distance is incrementally updated.

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_pos_xy = ball.data.root_pos_w[:, :2]
    robot_pos_xy = robot.data.root_pos_w[:, :2]

    dist_val: float = float(getattr(env, "stage6_spawn_dist", 0.15))
    spawn_pos_xy = robot_pos_xy.clone()
    spawn_pos_xy[:, 0] += dist_val

    drift = torch.norm(ball_pos_xy - spawn_pos_xy, dim=1)
    return drift


def alternate_foot_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward when the contacting foot alternates from the previous contact.

    Reads ``env.last_contact_foot`` (one-hot [L, R]) which is updated by
    ``ball_foot_contact_reward`` each step.

    Returns:
        Tensor of shape (num_envs,).
    """
    if not hasattr(env, "last_contact_foot"):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "juggle_streak_buf"):
        return torch.zeros(env.num_envs, device=env.device)

    # A non-zero streak means the most recent contact was an alternation
    alternated = env.juggle_streak_buf > 0
    return alternated.float()


def juggle_streak_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus that scales quadratically with the consecutive alt-foot streak.

    Formula: streak_count² × 1.5

    Returns:
        Tensor of shape (num_envs,).
    """
    if not hasattr(env, "juggle_streak_buf"):
        return torch.zeros(env.num_envs, device=env.device)

    streak = env.juggle_streak_buf.float()
    return streak.pow(2) * 1.5


def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Large negative reward on the step the episode terminates early (fall).

    Returns:
        Tensor of shape (num_envs,).
    """
    return (~env.reset_terminated).float() * 0.0 + env.reset_terminated.float() * -1.0


def robot_xy_drift_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the robot for drifting away from its spawn origin in the XY plane."""
    asset = env.scene[asset_cfg.name]

    # Extract XY coordinates of the robot and its spawn origin
    robot_pos_xy = asset.data.root_pos_w[:, :2]
    env_origins_xy = env.scene.env_origins[:, :2]

    # Calculate Euclidean distance (L2 norm) in the XY plane
    drift_dist = torch.norm(robot_pos_xy - env_origins_xy, dim=-1)

    return drift_dist
