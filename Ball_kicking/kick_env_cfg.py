import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp.h1.kick_rewards as kick_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp.h1.kick_curriculums as kick_curriculums  # Make sure to import the file containing init_contact_buffer

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import (
    H1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp.kick_events as kick_events


@configclass
class CurriculumCfg:
    advance_stage = CurrTerm(
        func=kick_curriculums.juggling_stage_curriculum,
        # Evaluated on reset automatically
    )


@configclass
class H1KickRewards(RewardsCfg):
    """Reward terms for the juggling MDP."""

    # ── Keep from locomotion baseline ──
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
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

    # ── Juggling-specific rewards ──
    juggle_impulse = RewTerm(
        func=kick_mdp.juggle_impulse_gaussian,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("foot_ball_contact_sensor"), "std": 2.0},
    )
    apex_height = RewTerm(
        func=kick_mdp.apex_height_band,
        weight=2.0,
        params={"ball_cfg": SceneEntityCfg("football")},
    )
    root_vel_penalty = RewTerm(
        func=kick_mdp.root_velocity_penalty,
        weight=-0.5,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    ball_drop_penalty = RewTerm(
        func=kick_mdp.floor_is_lava,
        weight=-100.0,
        params={"ball_cfg": SceneEntityCfg("football")},
    )
    hand_contact = RewTerm(
        func=kick_mdp.hand_contact_penalty,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("hand_ball_contact_sensor")},
    )
    alternate_foot = RewTerm(
        func=kick_mdp.alternate_foot_penalty,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("foot_ball_contact_sensor")},
    )


@configclass
class H1KickEnvCfg(H1FlatEnvCfg):
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
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.45
                ),  # Updated mass to 0.45kg
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=0.7,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0)
                ),
                activate_contact_sensors=True,
            ),
            # Initial state dropped from 1.0m height
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)),
        )

        curriculum: CurriculumCfg = CurriculumCfg()

        # ── Contact sensors ──
        self.scene.foot_ball_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*ankle_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Football"],
            track_air_time=False,
        )

        self.scene.hand_ball_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(.*_shoulder_.*|.*_elbow.*|.*_wrist.*|.*_hand.*)",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Football"],
            track_air_time=False,
        )

        # ── Observations ──
        self.observations.policy.ball_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("football")},
        )
        self.observations.policy.ball_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("football")},
        )

        # ── Events ──
        # Register the custom contact buffer
        self.events.setup_contact_buffer = EventTerm(
            func=kick_events.init_contact_buffer,
            mode="startup",
        )

        self.events.reset_ball = EventTerm(
            func=kick_events.reset_ball_curriculum,
            mode="reset",
            params={
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

        # Terminate when ball hits the ground
        self.terminations.ball_dropped = DoneTerm(
            func=kick_mdp.ball_ground_contact,
            params={"ball_cfg": SceneEntityCfg("football")},
        )


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
