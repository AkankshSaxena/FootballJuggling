# Ball_kicking: Setup

Files in this folder are an Isaac Lab task extension. Copy them into an Isaac Lab **v2.3.2** checkout (see `../setup_isaaclab.sh`).

`TASKS` = `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity`

## 1. Copy files

| File | Destination |
|---|---|
| `kick_env_cfg.py` | `TASKS/config/h1/` |
| `rsl_rl_kick_ppo_cfg.py` | `TASKS/config/h1/agents/` |
| `kick_rewards.py`, `kick_events.py`, `kick_observations.py`, `kick_terminations.py`, `kick_swing.py` | `TASKS/mdp/` |

## 2. Register the task

`kick___init__.py` contains only the two `gym.register(...)` calls. **Append** its contents to the end of `TASKS/config/h1/__init__.py`. Don't overwrite that file, or the stock H1 tasks break.

This registers `Isaac-Velocity-Kick-H1-v0` and `Isaac-Velocity-Kick-H1-Play-v0`.

## 3. Train (per stage)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Kick-H1-v0 --num_envs 4096 --headless --logger wandb
```

Each stage is a fresh run. Stage settings are in [`../docs/curriculum.md`](../docs/curriculum.md). To carry weights into the next stage, add `--resume --load_run <run_folder> --checkpoint <model.pt>`.

## 4. Play with Video Generation

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Kick-H1-v0 \
  --num_envs 32 \
  --checkpoint <path-to-FootballJuggling>/checkpoints/h1_stage3.3.pt \
  --video \
  --video_length 1250 \
  --headless
```

## Helper scripts (not needed for training)

`Body_names.py` prints the H1 body names. `leg_lengths.py` prints joint limits and leg geometry for the swing reward. Run them with `./isaaclab.sh -p <script>`.