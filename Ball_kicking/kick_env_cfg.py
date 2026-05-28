# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# ← import your custom reward functions separately
import isaaclab_tasks.manager_based.locomotion.velocity.config.h1.kick_rewards as kick_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import (
    H1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip


# ─────────────────────────────────────────────
# REWARDS
# ─────────────────────────────────────────────
@configclass
class H1KickRewards(RewardsCfg):
    """Reward terms for the kicking MDP."""

    # ── Keep from locomotion baseline ──
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None  # disable
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")},
    )

    # ── Kick-specific rewards (from kick_rewards.py) ──
    move_towards_ball = RewTerm(
        func=kick_mdp.move_towards_ball,
        weight=0.2,  # Tried with 0.5 previously
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
            "ball_cfg": SceneEntityCfg("football"),
        },
    )
    ball_feet_contact = RewTerm(
        func=kick_mdp.ball_feet_contact,
        weight=0.3,  # Tried with 2 previously
        params={
            "sensor_cfg": SceneEntityCfg(
                "foot_ball_contact_sensor", body_names=".*ankle_link"
            ),
        },
    )
    ball_upward_velocity = RewTerm(
        func=kick_mdp.ball_upward_velocity,
        weight=15.0,  # tried with 3 previously
        params={
            "ball_cfg": SceneEntityCfg("football"),
        },
    )


# ─────────────────────────────────────────────
# MAIN ENV CONFIG
# ─────────────────────────────────────────────
@configclass
class H1KickEnvCfg(H1FlatEnvCfg):  # ← flat, not rough base
    rewards: H1KickRewards = H1KickRewards()

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ──
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Ball ──
        self.scene.football = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Football",
            spawn=sim_utils.SphereCfg(
                radius=0.11,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=0.7,  # ball bounce
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0)  # orange
                ),
                activate_contact_sensors=True,  # CRITICAL — enables contact reporting
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.11)),
        )

        # ── Contact sensor: foot vs ball ──
        # Separate from existing contact_forces (which is foot vs ground)
        self.scene.foot_ball_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*ankle_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Football"],
            track_air_time=False,
        )

        # ── Ball observations ──
        # Ball world position → agent knows where ball is
        self.observations.policy.ball_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("football")},
        )

        # ── Ball reset on episode reset ──
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.3, 0.7),
                    "y": (-0.3, 0.3),
                    "z": (0.11, 0.11),  # keep on ground
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("football"),
            },
        )

        # ── Disable unused events ──
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            ".*torso_link"
        ]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ── Reward weights ──
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # ── Commands ──
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ── Terminations ──
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


# ─────────────────────────────────────────────
# PLAY CONFIG
# ─────────────────────────────────────────────
@configclass
class H1KickEnvCfg_PLAY(H1KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
