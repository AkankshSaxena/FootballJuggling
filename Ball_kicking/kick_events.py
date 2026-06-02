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

    # Get the base default state (which is currently local)
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # --- FIX: Get the world-frame origins of the specific environments ---
    env_origins = env.scene.env_origins[env_ids]
    
    # --- FIX: Shift the local coordinates to global world coordinates ---
    root_states[:, 0] += env_origins[:, 0]
    root_states[:, 1] += env_origins[:, 1]
    root_states[:, 2] += env_origins[:, 2]

    # Read the current curriculum radius
    if hasattr(env, "ball_radius"):
        radii = env.ball_radius[env_ids]
    else:
        radii = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)

    # Generate random (X, Y) offsets within the current radius
    dx = (torch.rand_like(radii) * 2.0 - 1.0) * radii
    dy = (torch.rand_like(radii) * 2.0 - 1.0) * radii

    # Apply curriculum offsets
    root_states[:, 0] += dx
    root_states[:, 1] += dy

    # Zero out velocities (columns 7 to 13)
    root_states[:, 7:] = 0.0

    # Write the globally-shifted states directly to the physics engine
    asset.write_root_state_to_sim(root_states, env_ids)