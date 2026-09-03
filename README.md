# FootballJuggling — H1 Humanoid Ball Juggling (RL)

## 1. Overview
Training a Unitree H1 humanoid to perform sustained football juggling using pure reinforcement learning, no motion capture, no reference trajectories, no kinematic retargeting. Policy trained with PPO (RSL-RL) in Isaac Lab on PhysX GPU, 4096 parallel environments. Behavior is shaped entirely through a staged reward curriculum: the ball starts constrained to a vertical rail at foot height, and physical constraints (Z → Y → X) are released one at a time as the policy masters each stage, ending in a fully free 0.45 kg ball. Reward terms are gated on the ball object and force thresholds (not just contact) to distinguish a genuine strike from a carry, since a frictionless ball is trivially exploitable by pure contact rewards.

## 2. Current Stage
Stage 3.3 complete: sustained single-leg (right foot) free-ball juggling, bounded XY drift, working 25 s demo. Training frozen; packaging phase in progress.

## 3. Videos

1. Learn to swing (no ball contact) 
2.1 – 2.3. Learn to kick (ball on constrained rail) 
3.1 – 3.3. Learn to maintain juggling (free ball) 

## 4. Failure Cases

1. **Reward hacking** -
   Video: https://github.com/user-attachments/assets/8a442c6b-f6b7-4c00-838a-fa5a21ae84d7

2. **No strike** — 
   Video: https://github.com/user-attachments/assets/7ba3672e-8df0-436c-a6fc-5a417cc20f4e

3. **Trapping exploit** 

4. **Carry/dribble exploit** 

## 5. System Specs

**Cloud VM (primary)**

| | |
|---|---|
| GPU | A6000 / L40S 48GB |
| Driver | 550 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 (hard-pinned) |
| Python | 3.11 |
| Torch | 2.7.0+cu128 |
| RL library | RSL-RL 3.1.2 (PPO) |
| Physics | PhysX GPU |
| Parallel envs | 4096 |
| Logging | WandB |

## 6. Errors Found & Fixed

| Issue | Fix |
|---|---|
| Isaac Lab `main` drifted to v3.0 (Py3.12, Isaac Sim 6.0) | Hard-pin to `v2.3.2` |
| `setuptools>=81` drops `pkg_resources`, breaks `flatdict==4.0.1` build | Pin `setuptools<81` |
| `H1_MINIMAL_CFG` has no collision geometry — no contact registered | Use `H1_CFG` |
| `net_forces_w_history` aggregates all contacts (ground/self) — false-positive rewards | Use `force_matrix_w_history` (ball-filtered) |
| PhysX GPU puts free-flying ball to sleep, ignores scripted velocity writes | `sleep_threshold=0.0` on ball rigid body |
| `preserve_order=False` in `SceneEntityCfg` silently misassigns body indices | Always set `preserve_order=True` |
| Curriculum released X-axis (fore-aft) before Y-axis (lateral) — opened a drift-based reward-hacking path | Reordered axis release to Z → Y → X |
| `ball_gravity_scale` (reduced effective gravity) let the policy loft the ball off under-powered, non-genuine strikes | Disabled reduced-gravity event; train on true gravity only |
| Compounded entropy/noise_std runaway (stagnated at 1.4) for stage 2.3  | Lowered `entropy_coef` 0.01 → 0.005, raised `apex_height` band to 1.2–2.8 m, raised `ball_foot_contact` min-force threshold |
