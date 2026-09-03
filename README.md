# FootballJuggling — H1 Humanoid Ball Juggling (RL)

## 1. Overview
Training a Unitree H1 humanoid to perform sustained football juggling using pure reinforcement learning, no motion capture, no reference trajectories, no kinematic retargeting. Policy trained with PPO (RSL-RL) in Isaac Lab on PhysX GPU, 4096 parallel environments. Behavior is shaped entirely through a staged reward curriculum: the ball starts constrained to a vertical rail at foot height, and physical constraints (Z → Y → X) are released one at a time as the policy masters each stage, ending in a fully free 0.45 kg ball. Reward terms are gated on the ball object and force thresholds (not just contact) to distinguish a genuine strike from a carry, since a frictionless ball is trivially exploitable by pure contact rewards.

## 2. Current Stage
Stage 3.3 complete: sustained single-leg (right foot) free-ball juggling, bounded XY drift, working 25 s demo. Training frozen; packaging phase in progress.

https://github.com/user-attachments/assets/64f6f2e4-2f3d-4168-b444-81cdae748338

## 3. Videos
Stage 1 - Learn to swing (no ball contact) 

https://github.com/user-attachments/assets/b21341cd-47f3-4cae-8b68-bb0c16bdde1d


Stage 2 (2.1 – 2.3) - Learn to kick (ball on constrained rail)

https://github.com/user-attachments/assets/8f16c0de-7955-4d83-8cba-59339ee5a430

https://github.com/user-attachments/assets/98f49dea-1b90-4ad4-84bc-b31ff60af1e8

https://github.com/user-attachments/assets/94ffdb7b-694c-4269-a010-6372c13ee944


Stage 3 (3.1 – 3.3) - Learn to maintain juggling (free ball) 

https://github.com/user-attachments/assets/f1fa5a1f-569d-4d8c-9f4a-eaad2f26b821

https://github.com/user-attachments/assets/636f253b-14db-4416-b7f2-d80ce7dd9780

https://github.com/user-attachments/assets/e6ea45d9-8f77-415c-b192-7584d6aa6bd7




## 4. Failure Cases

1. **Reward hacking** -
   
https://github.com/user-attachments/assets/8a442c6b-f6b7-4c00-838a-fa5a21ae84d7

2. **No strike** -
   
https://github.com/user-attachments/assets/7ba3672e-8df0-436c-a6fc-5a417cc20f4e

3. **Trapping exploit** -
   
https://github.com/user-attachments/assets/e0ac9624-a191-42e6-8370-7d2b75638b56

4. **Ball Dropping** -
   
https://github.com/user-attachments/assets/787a5b2f-e46e-40ac-8e55-98b084e74fd2


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
