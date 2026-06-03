import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_rotate_inverse


def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Relative position in world frame
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w

    # Rotate into robot base frame
    rel_pos_b = quat_rotate_inverse(robot.data.root_quat_w, rel_pos_w)
    return rel_pos_b


def ball_linear_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Rotate ball's world linear velocity into robot base frame
    ball_lin_vel_b = quat_rotate_inverse(
        robot.data.root_quat_w, ball.data.root_lin_vel_w
    )
    return ball_lin_vel_b


def ball_linear_velocity_world(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    return ball.data.root_lin_vel_w


def last_contact_foot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Retrieves the last foot to make contact with the ball.
    Assumes `env.last_contact_foot` (shape: [num_envs, 2]) is initialized in the environment
    and updated during the step based on foot contact sensors (e.g., [1, 0] for Left, [0, 1] for Right).
    """
    if not hasattr(env, "last_contact_foot"):
        # Fallback to zero tensor if not yet initialized by events/setup
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return env.last_contact_foot.clone()
