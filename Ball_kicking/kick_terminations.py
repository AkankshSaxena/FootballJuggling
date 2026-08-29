import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def torso_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < minimum_height


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 3.0,
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    distance = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )

    return distance > max_distance


def ball_on_ground_timeout(
    env: ManagerBasedRLEnv,
    delay_s: float = 2.0,
    ground_height: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    on_ground = ball.data.root_pos_w[:, 2] < ground_height

    if not hasattr(env, "ball_ground_since"):
        env.ball_ground_since = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )

    t = env.episode_length_buf.float() * env.step_dt
    newly = on_ground & torch.isinf(env.ball_ground_since)  # latch first touchdown only
    env.ball_ground_since = torch.where(newly, t, env.ball_ground_since)

    elapsed = t - env.ball_ground_since
    fire = elapsed >= delay_s

    env.extras.setdefault("log", {})["debug/ball_grounded_frac"] = (
        torch.isfinite(env.ball_ground_since).float().mean().item()
    )
    return fire
