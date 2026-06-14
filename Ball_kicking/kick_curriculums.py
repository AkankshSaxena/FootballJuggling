import os
import torch
from isaaclab.envs import ManagerBasedRLEnv

SIGNAL_FILE = "/tmp/advance_stage.signal"


def advance_curriculum_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    if not hasattr(env, "curriculum_stage"):
        env.curriculum_stage = 1
        env.stage6_distance = 0.15
        env.episode_success_buf = torch.zeros((env.num_envs, 100), device=env.device)
        env.episode_counts = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)
        prev = env.curriculum_stage
        env.curriculum_stage = min(env.curriculum_stage + 1, 7)
        env.episode_counts.zero_()
        if prev != env.curriculum_stage:
            print(f"\n[CURRICULUM] Stage {prev} → {env.curriculum_stage}\n")

    stage = env.curriculum_stage

    default_zero = torch.zeros(env.num_envs, device=env.device)
    contacts = getattr(env, "contact_count", default_zero)[env_ids]
    leg_raises = getattr(env, "leg_raise_counts", default_zero)[env_ids]
    apex_heights = getattr(env, "ball_apex_height", default_zero)[env_ids]
    apex_valid = (apex_heights >= 0.8) & (apex_heights <= 1.6)
    timeout = (env.episode_length_buf[env_ids] * env.step_dt) >= 14.9

    if stage == 1:
        success = (leg_raises >= 15) & timeout
    elif stage == 2:
        success = (contacts >= 15) & timeout
    elif stage == 3:
        success = (contacts >= 10) & apex_valid & timeout
    elif stage in [4, 5]:
        success = (contacts >= 2) & apex_valid & timeout
    else:  # 6, 7
        success = (contacts >= 2) & apex_valid & timeout

    for i, env_idx in enumerate(env_ids):
        count = env.episode_counts[env_idx]
        env.episode_success_buf[env_idx, count % 100] = success[i].float()
        env.episode_counts[env_idx] += 1

    if stage == 6:
        total_avg = env.episode_counts.float().mean().item()
        if total_avg > 0 and int(total_avg) % 10 == 0:
            idx_s = (int(total_avg) - 10) % 100
            idx_e = int(total_avg) % 100
            if idx_e > idx_s:
                recent = env.episode_success_buf[:, idx_s:idx_e].mean().item()
            else:
                recent = (
                    torch.cat(
                        [
                            env.episode_success_buf[:, idx_s:],
                            env.episode_success_buf[:, :idx_e],
                        ],
                        dim=1,
                    )
                    .mean()
                    .item()
                )
            if recent >= 0.80 and env.stage6_distance < 0.6:
                env.stage6_distance = min(0.6, env.stage6_distance + 0.03)
                print(f"[CURRICULUM] Stage 6 distance → {env.stage6_distance:.2f}")

    if "log" not in env.extras:
        env.extras["log"] = {}

    env.extras["log"]["curriculum/stage"] = float(env.curriculum_stage)
    env.extras["log"]["curriculum/stage6_distance"] = float(
        getattr(env, "stage6_distance", 0.0)
    )
    env.extras["log"]["curriculum/success_rate"] = env.episode_success_buf.mean().item()
    env.extras["log"]["curriculum/episodes_avg"] = (
        env.episode_counts.float().mean().item()
    )
