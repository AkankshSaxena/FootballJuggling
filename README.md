# FootballJuggling — H1 Humanoid Ball Juggling (RL)

## 1. Overview
Training a Unitree H1 humanoid to perform sustained football juggling using pure reinforcement learning, with no motion capture or kinematic retargeting. Policy trained with PPO (RSL-RL) in Isaac Lab on PhysX GPU, 4096 parallel environments. Behavior is shaped entirely through a 4-stage reward curriculum (balance & swing → ball contact on a rail → free-ball juggling → domain randomization, planned), with stages 1-3 trained. The ball starts constrained to a vertical rail at foot height, and physical constraints (Z → Y → X) are released one at a time as the policy masters each stage, ending in a fully free 0.45 kg ball. Reward terms are gated on the ball object and force thresholds (not just contact) to distinguish a genuine strike from a carry, since a frictionless ball is trivially exploitable by pure contact rewards.

## 2. Current Stage
Stages 1-3 complete (final: Stage 3.3): sustained single-leg (right foot) free-ball juggling, bounded XY drift, 25 s demo. Stage 4 (domain randomization) is planned.

**Training logs:** [W&B project](https://forge.coreweave.com/wandb/akanksh-iitd-self/isaaclab) (final runs: `s1`-`s3.3`; the rest are earlier iterations).

**Best checkpoint:** [checkpoints/h1_stage3.3.pt](checkpoints/h1_stage3.3.pt)

https://github.com/user-attachments/assets/64f6f2e4-2f3d-4168-b444-81cdae748338

**Ablation (frozen checkpoint, altered observations):** confirms the policy reacts to the ball and isn't replaying a timed motion.

| Observation change | Scored kicks / episode |
|---|---|
| None (baseline) | 30 |
| Swing-phase clock shifted by half a period | ~ 30 (no change) |
| Swing-phase output offset +0.4 rad (out-of-distribution) | ~ 30 (no change) |
| Ball position & velocity zeroed | ~ 0 (collapsed) |

## 3. Videos
**Stage 1** - Learn to swing (no ball contact) 

https://github.com/user-attachments/assets/b21341cd-47f3-4cae-8b68-bb0c16bdde1d


**Stage 2 (2.1 – 2.3)** - Learn to kick (ball on constrained rail)

https://github.com/user-attachments/assets/8f16c0de-7955-4d83-8cba-59339ee5a430

https://github.com/user-attachments/assets/98f49dea-1b90-4ad4-84bc-b31ff60af1e8

https://github.com/user-attachments/assets/94ffdb7b-694c-4269-a010-6372c13ee944


**Stage 3 (3.1 – 3.3)** - Learn to maintain juggling (free ball) 

https://github.com/user-attachments/assets/f1fa5a1f-569d-4d8c-9f4a-eaad2f26b821

https://github.com/user-attachments/assets/636f253b-14db-4416-b7f2-d80ce7dd9780

https://github.com/user-attachments/assets/e6ea45d9-8f77-415c-b192-7584d6aa6bd7




## 4. Failure Cases

1. **Ball dropping (all constraints released at once)**
   Releasing every ball constraint at once left the policy unable to keep the ball up. Replaced with a staged release.

https://github.com/user-attachments/assets/1c68e2da-6aea-4830-bdeb-fbe9ffe54456

2. **Scooping instead of kicking**
   The robot scooped the ball and skipped the apex reward, since the contact reward depended on touch, not force. Added peak-force and launch-velocity thresholds to the kick gate.

https://github.com/user-attachments/assets/d5c7c824-6e95-453c-8a3c-bed6d427b537

3. **Limit-cycle farming (repeated tapping)**
   The policy tapped the ball over and over with quick swings to hold it near apex height, farming the apex and contact rewards without real strikes. Tightened the kick gate: rising-edge force trigger, a higher peak-force threshold and a minimum interval between scored kicks.

https://github.com/user-attachments/assets/9038352e-d40c-4bd2-b2a2-d5f8dfc3b18b

4. **Ball passing through the shin (missing collision geometry)**
   The ball penetrated the robot's shin with no contact registered, because `H1_MINIMAL_CFG` has no collision geometry. Switched to the full `H1_CFG`.

https://github.com/user-attachments/assets/499f1f50-f589-4313-af91-40f342ec54b8

5. **Rolling the ball along the legs (axis-order loophole)**
   The robot minimized the ball-drop penalty by rolling the ball along its legs. Releasing X before the still-constrained Y axis made this a sub-optimal path. Reordered the release to Z → Y → X.

https://github.com/user-attachments/assets/78b1274c-9902-4fbf-a688-ab2c5676774b

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
