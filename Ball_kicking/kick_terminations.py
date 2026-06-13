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


def ball_ground_contact_with_delay(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ground_threshold: float = 0.12,  # Approximate radius of a size 5 football
    delay_s: float = 2.0,
) -> torch.Tensor:
    """
    Terminates the episode if the ball contacts the ground and remains there
    (or falls below threshold) for longer than the specified delay (2s).
    """
    
    stage = getattr(env, "curriculum_stage", 1)
    if stage == 1:
        # Return a tensor of all False (no terminations)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ball = env.scene[ball_cfg.name]
    ball_height = ball.data.root_pos_w[:, 2]

    # Identify environments where the ball is on the ground
    is_on_ground = ball_height <= ground_threshold

    # Initialize the contact timer tracking if it doesn't exist yet
    if not hasattr(env, "ball_ground_contact_time"):
        env.ball_ground_contact_time = torch.full(
            (env.num_envs,), -1.0, device=env.device, dtype=torch.float32
        )

    # Calculate current physical time per environment
    current_time = env.episode_length_buf * env.step_dt

    # Record the timestamp of new ground contacts
    new_contact_mask = is_on_ground & (env.ball_ground_contact_time < 0)
    env.ball_ground_contact_time = torch.where(
        new_contact_mask, current_time, env.ball_ground_contact_time
    )

    # Reset the timer if the ball goes back up (e.g., bounced but recovered, though unlikely)
    env.ball_ground_contact_time = torch.where(
        ~is_on_ground, -1.0, env.ball_ground_contact_time
    )

    # Calculate duration since contact and check against delay
    time_since_contact = current_time - env.ball_ground_contact_time
    terminate_mask = (env.ball_ground_contact_time >= 0) & (
        time_since_contact >= delay_s
    )

    return terminate_mask