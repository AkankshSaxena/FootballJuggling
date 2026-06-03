import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg

def reset_ball_position_and_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Resets the ball's position and velocity based on the current curriculum stage."""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    
    # Retrieve current curriculum stage (defaults to 1 if not initialized)
    stage = getattr(env, "curriculum_stage", 1)
    num_resets = len(env_ids)
    
    # Initialize base tensors
    ball_pos = torch.zeros((num_resets, 3), device=env.device)
    ball_vel = torch.zeros((num_resets, 6), device=env.device)
    ball_quat = torch.zeros((num_resets, 4), device=env.device)
    ball_quat[:, 0] = 1.0  # Identity quaternion (w, x, y, z)
    
    if stage == 1:
        # Stage 1: No ball spawned. Place far underground.
        ball_pos[:, 2] = -10.0 
    else:
        robot_pos = robot.data.root_pos_w[env_ids]
        
        # Default heights and distances for Stage 2 & 3
        heights = torch.full((num_resets,), 0.3, device=env.device)
        distances = torch.full((num_resets,), 0.15, device=env.device)
        
        # Override based on stage
        if stage == 4:
            heights.fill_(0.6)
        elif stage >= 5:
            heights.fill_(1.0)
            
        if stage == 6:
            # Stage 6: Incrementing distance stored in environment
            dist_val = getattr(env, "stage6_distance", 0.15)
            distances.fill_(dist_val)
        elif stage == 7:
            # Stage 7: Random distance between 0.15m and 0.6m
            distances = torch.rand((num_resets,), device=env.device) * (0.6 - 0.15) + 0.15
            
        # Spawn ball in front of the robot (Assuming +X is forward in world coordinates)
        ball_pos[:, 0] = robot_pos[:, 0] + distances
        ball_pos[:, 1] = robot_pos[:, 1]
        ball_pos[:, 2] = heights

    # Apply state to simulation
    ball.write_root_pose_to_sim(torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)

def reset_custom_tracking_variables(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor
):
    """Resets tracking variables for contacts, streaks, and apex logic during an episode reset."""
    if hasattr(env, "last_contact_foot"):
        env.last_contact_foot[env_ids] = 0.0
    if hasattr(env, "juggle_streak_count"):
        env.juggle_streak_count[env_ids] = 0.0
    if hasattr(env, "ball_apex_height"):
        env.ball_apex_height[env_ids] = 0.0
    if hasattr(env, "ball_ground_contact_time"):
        env.ball_ground_contact_time[env_ids] = -1.0    root_states[:, 1] += dy

    # Zero out velocities (columns 7 to 13)
    root_states[:, 7:] = 0.0

    # Write the globally-shifted states directly to the physics engine
    asset.write_root_state_to_sim(root_states, env_ids)