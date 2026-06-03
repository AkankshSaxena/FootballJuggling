import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv


def advance_curriculum_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    Evaluates transition gates and updates curriculum stages and distances.
    This tracks success rates across a rolling buffer of episodes to trigger promotions.
    """
    if not hasattr(env, "curriculum_stage"):
        env.curriculum_stage = 1
        env.stage6_distance = 0.15
        # 100-episode rolling success buffer per environment
        env.episode_success_buf = torch.zeros((env.num_envs, 100), device=env.device)
        env.episode_counts = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    stage = env.curriculum_stage

    # Check success criteria for terminating environments
    success = torch.zeros_like(env_ids, dtype=torch.bool)

    contacts = getattr(env, "ball_contact_counts", torch.zeros_like(env_ids))[env_ids]
    leg_raises = getattr(env, "leg_raise_counts", torch.zeros_like(env_ids))[env_ids]
    apex_valid = getattr(
        env, "ball_apex_valid", torch.zeros_like(env_ids, dtype=torch.bool)
    )[env_ids]

    # Assume maximum episode length (15s) is tracked by episode_length_buf matching max length
    timeout = (env.episode_length_buf[env_ids] * env.step_dt) >= 14.9

    if stage == 1:
        success = (leg_raises >= 15) & timeout
    elif stage == 2:
        success = (contacts >= 15) & timeout
    elif stage == 3:
        success = (contacts >= 10) & apex_valid & timeout
    elif stage in [4, 5]:
        success = (contacts >= 2) & apex_valid & timeout
    elif stage == 6:
        success = (contacts >= 2) & apex_valid & timeout

    # Update rolling success buffer
    for i, env_idx in enumerate(env_ids):
        count = env.episode_counts[env_idx]
        env.episode_success_buf[env_idx, count % 100] = success[i].float()
        env.episode_counts[env_idx] += 1

    # Calculate global success rate across all environments
    total_episodes_avg = env.episode_counts.float().mean().item()
    global_success_rate = env.episode_success_buf.mean().item()

    # Transition Logic
    if stage == 1 and total_episodes_avg >= 100:
        if global_success_rate >= 0.90:
            env.curriculum_stage = 2
            env.episode_counts.zero_()

    elif stage == 2 and total_episodes_avg >= 50:
        recent_50_success = env.episode_success_buf[:, :50].mean().item()
        if recent_50_success >= 0.90:
            env.curriculum_stage = 3
            env.episode_counts.zero_()

    elif stage == 3 and total_episodes_avg >= 100:
        if global_success_rate >= 0.90:
            env.curriculum_stage = 4
            env.episode_counts.zero_()

    elif stage == 4 and total_episodes_avg >= 100:
        if global_success_rate >= 0.85:
            env.curriculum_stage = 5
            env.episode_counts.zero_()

    elif stage == 5 and total_episodes_avg >= 100:
        if global_success_rate >= 0.80:
            env.curriculum_stage = 6
            env.episode_counts.zero_()
            env.stage6_distance = 0.15

    elif stage == 6:
        # Distance Increment Logic (every 10 episodes)
        if total_episodes_avg > 0 and int(total_episodes_avg) % 10 == 0:
            idx_start = (int(total_episodes_avg) - 10) % 100
            idx_end = int(total_episodes_avg) % 100

            # Handle wrapping buffer indices safely
            if idx_end > idx_start:
                recent_10_success = (
                    env.episode_success_buf[:, idx_start:idx_end].mean().item()
                )
            else:
                recent_10_success = (
                    torch.cat(
                        [
                            env.episode_success_buf[:, idx_start:],
                            env.episode_success_buf[:, :idx_end],
                        ],
                        dim=1,
                    )
                    .mean()
                    .item()
                )

            if recent_10_success >= 0.80 and env.stage6_distance < 0.6:
                env.stage6_distance = min(0.6, env.stage6_distance + 0.03)

        # Stage Transition Logic
        if total_episodes_avg >= 100 and env.stage6_distance >= 0.6:
            if global_success_rate >= 0.80:
                env.curriculum_stage = 7
                env.episode_counts.zero_()
