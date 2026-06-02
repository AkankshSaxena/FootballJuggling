"""Custom event functions for the H1 juggling task."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def init_contact_buffer(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Initializes and resets the foot contact tracking buffer.

    State mapping: -1 = no contact, 0 = left foot, 1 = right foot.
    """
    if not hasattr(env, "last_contact_foot"):
        # First time setup (mode="startup")
        env.last_contact_foot = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    else:
        # Reset buffer for specific environments that just terminated
        env.last_contact_foot[env_ids] = -1


def reset_ball_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("football"),
):
    """Resets the ball using the dynamic drop radius from the curriculum."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # Clone the initial state (which contains your Z=1.0 drop height and X=0.5 default)
    root_states = asset.data.default_root_state[env_ids].clone()

    # Read the current curriculum radius for these specific envs (default to 0 if not set)
    if hasattr(env, "ball_radius"):
        radii = env.ball_radius[env_ids]
    else:
        radii = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)

    # Generate random (X, Y) offsets within the current radius
    # (torch.rand gives [0, 1) -> scaled to [-radius, radius])
    dx = (torch.rand_like(radii) * 2.0 - 1.0) * radii
    dy = (torch.rand_like(radii) * 2.0 - 1.0) * radii

    # Apply offsets to position
    root_states[:, 0] += dx
    root_states[:, 1] += dy

    # Zero out velocities (columns 7 to 13 are lin_vel and ang_vel)
    root_states[:, 7:] = 0.0

    # Write the new states directly to the physics engine
    asset.write_root_state_to_sim(root_states, env_ids)
