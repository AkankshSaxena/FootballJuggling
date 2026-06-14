from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets import H1_MINIMAL_CFG
from isaaclab.sensors import ContactSensorCfg

# Importing your custom kick_* modules
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    kick_curriculums,
    kick_events,
    kick_observations,
    kick_rewards,
    kick_terminations,
)


@configclass
class H1JuggleSceneCfg(InteractiveSceneCfg):
    """Configuration for the juggling scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Robot asset
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Ball asset
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class H1JuggleActionsCfg:
    """Action specifications for the environment."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class H1JuggleObservationsCfg:
    """Observation specifications for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state (using standard mdp imports instead of lambdas)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Custom Ball State (routed to kick_observations)
        ball_pos_robot_frame = ObsTerm(
            func=kick_observations.ball_position_in_robot_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        ball_lin_vel_robot_frame = ObsTerm(
            func=kick_observations.ball_linear_velocity_in_robot_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        last_contact_foot = ObsTerm(func=kick_observations.last_contact_foot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class H1JuggleEventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # 1.0 scales default positions exactly
            "velocity_range": (0.0, 0.0),
        },
    )

    # Your existing ball reset
    reset_ball = EventTerm(
        func=kick_events.reset_ball_position_and_velocity,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # Your existing tracking vars reset
    reset_tracking_vars = EventTerm(
        func=kick_events.reset_juggling_state,
        mode="reset",
    )


@configclass
class H1JuggleRewardsCfg:
    """Reward configuration matching curriculum stages."""

    # Penalties (routed to kick_rewards)
    lin_vel_z_l2 = RewTerm(func=kick_rewards.lin_vel_z_l2, weight=-1.5)

    feet_slide = RewTerm(
        func=kick_rewards.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )

    dof_pos_limits = RewTerm(func=kick_rewards.dof_pos_limits, weight=-0.5)

    leg_raise = RewTerm(
        func=kick_rewards.leg_raise_reward,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_link", "right_ankle_link"]
            )
        },
        weight=0.2,
    )

    target_impulse = RewTerm(
        func=kick_rewards.target_impulse_reward,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "target_apex_height": 1.0,
            # ADD THIS: Point the function to the correct sensor we just created
            "foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_link"
            ),
        },
        weight=3.0,
    )

    ball_apex = RewTerm(
        func=kick_rewards.ball_apex_height_reward,
        params={"target_height": 1.0, "tolerance": 0.2},
        weight=3.0,
    )

    alternate_foot = RewTerm(func=kick_rewards.alternate_foot_reward, weight=2.0)

    juggle_streak = RewTerm(func=kick_rewards.juggle_streak_bonus, weight=1.0)

    # Custom Penalties (routed to kick_rewards)
    ball_xy_velocity = RewTerm(
        func=kick_rewards.track_lin_vel_ball_xy_exp,
        params={"ball_cfg": SceneEntityCfg("ball")},
        weight=1.0,
    )

    ball_xy_drift = RewTerm(
        func=kick_rewards.ball_xy_drift_penalty,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.98},  # H1 nominal standing height
    )

    orientation_penalty = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    robot_xy_drift = RewTerm(
        func=kick_rewards.robot_xy_drift_penalty,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-2.0,  # Start with a strong penalty to anchor the robot
    )


@configclass
class H1JuggleTerminationsCfg:
    """Termination criteria."""

    time_out = DoneTerm(func=kick_terminations.time_out, time_out=True)

    robot_falls = DoneTerm(
        func=kick_terminations.torso_height_below,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )

    ball_ground_contact = DoneTerm(
        func=kick_terminations.ball_ground_contact_with_delay,
        params={"ball_cfg": SceneEntityCfg("ball"), "delay_s": 2.0},
    )


@configclass
class H1JuggleCurriculumCfg:
    """Curriculum configurations."""

    stage_progression = CurrTerm(func=kick_curriculums.advance_curriculum_stage)


@configclass
class H1JuggleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree H1 Juggle Environment."""

    scene: H1JuggleSceneCfg = H1JuggleSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: H1JuggleActionsCfg = H1JuggleActionsCfg()
    # CRITICAL FIX: These must retain their default attribute names to be recognized by Isaac Lab.
    observations: H1JuggleObservationsCfg = H1JuggleObservationsCfg()
    events: H1JuggleEventCfg = H1JuggleEventCfg()
    rewards: H1JuggleRewardsCfg = H1JuggleRewardsCfg()
    terminations: H1JuggleTerminationsCfg = H1JuggleTerminationsCfg()
    curriculum: H1JuggleCurriculumCfg = H1JuggleCurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 15.0
        self.decimation = 4
        self.sim.dt = 0.005

        # Fix ground penetration
        self.sim.physx.solver_type = 1
        self.sim.physx.num_position_iterations = 8
        self.sim.physx.num_velocity_iterations = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_collision_stack_size = 2**26
