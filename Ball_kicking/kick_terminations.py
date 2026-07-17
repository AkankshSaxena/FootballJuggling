import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Terminate episodes that reach the time limit.
    Relies on self.episode_length_s = 15.0 defined in env_cfg.
    """
    return env.episode_length_buf >= env.max_episode_length


def torso_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminates the episode if the robot's torso (root) height falls below 0.3m.
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < minimum_height


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 3.0,
) -> torch.Tensor:
    """
    Terminates the episode if the 2D (XY) Euclidean distance between the robot
    and the ball exceeds the allowed max_distance (e.g., 3.0m).
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Calculate 2D Euclidean distance (X, Y only) between robot root and ball root
    distance = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )

    return distance > max_distance
