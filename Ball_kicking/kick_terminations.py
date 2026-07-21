import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject


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


def ball_on_ground_timeout(
    env: ManagerBasedRLEnv,
    delay_s: float = 2.0,
    ground_height: float = 0.15,  # ball radius 0.12 + 0.03 margin -> center-Z at/near rest
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate `delay_s` after the ball first reaches the ground (Stage 3+).

    LATCHING: countdown starts at the first frame the ball center drops below
    ground_height and does NOT cancel if the ball bounces back up. Ground contact
    == failed juggle; the delay is a grace window, not a recoverable state. Set
    delay_s=0.0 to end the episode instantly on touchdown.

    Height proxy (flat plane at Z=0, 0.12 m sphere): center-Z < 0.15 stands in for
    "resting on the ground" without another PhysX filtered sensor. Inert whenever the
    ball is Z-pinned above ground_height (Stages 1-2, floor 0.27), so it is safe to
    register from the start.

    Register with time_out=False (default): FAILURE, not a horizon cutoff. Sets
    reset_terminated -> incurs termination_penalty (-100) and is treated as a true
    terminal (no value bootstrap). This is the pressure that forces the policy to keep
    the ball aloft in Stage 3 (do nothing -> ball falls -> -100). If you want a softer
    penalty than a fall, split it into its own reward term keyed on this condition.

    STATE env.ball_ground_since MUST be reset per episode in reset_ball_state,
    alongside the contact_count / prev_kick_foot resets from before.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    on_ground = ball.data.root_pos_w[:, 2] < ground_height

    if not hasattr(env, "ball_ground_since"):
        env.ball_ground_since = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )

    t = env.episode_length_buf.float() * env.step_dt
    newly = on_ground & torch.isinf(env.ball_ground_since)  # latch first touchdown only
    env.ball_ground_since = torch.where(newly, t, env.ball_ground_since)

    elapsed = t - env.ball_ground_since  # -inf where never grounded -> fire=False
    fire = elapsed >= delay_s

    env.extras.setdefault("log", {})["debug/ball_grounded_frac"] = (
        torch.isfinite(env.ball_ground_since).float().mean().item()
    )
    return fire
