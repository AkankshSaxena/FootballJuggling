from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

# from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
    """Configuration for the stage 1 juggling scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Robot asset
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Ball asset
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                linear_damping=0.1,
                angular_damping=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle_link",
        history_length=3,
        track_air_time=True,
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
            func=kick_observations.ball_position_in_robot_frame
        )
        ball_lin_vel_robot_frame = ObsTerm(
            func=kick_observations.ball_linear_velocity_in_robot_frame
        )
        feet_position_in_robot_frame = ObsTerm(
            func=kick_observations.feet_position_in_robot_frame
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
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    reset_ball = EventTerm(
        func=kick_events.reset_ball_state,
        mode="reset",
        params={"distance_offset": 0.5, "height_offset": 0.3},
    )

    constrain_ball = EventTerm(
        func=kick_events.constrain_ball_to_z_axis,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # fires every env step
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "distance_offset": 0.5,
            "min_height": 0.3,
        },
    )


@configclass
class H1JuggleRewardsCfg:
    """Reward configuration matching curriculum stages."""

    # Juggling Penalties
    termination_penalty = RewTerm(func=kick_rewards.termination_penalty, weight=-10.0)
    ball_robot_dist_penalty = RewTerm(
        func=kick_rewards.ball_robot_dist_penalty, weight=-1.0
    )

    # Juggling Rewards
    gated_ball_contact = RewTerm(
        func=kick_rewards.gated_ball_contact_reward, weight=2.0
    )
    apex_height = RewTerm(func=kick_rewards.apex_height_reward, weight=2.0)

    # Tracking
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.98}
    )
    orientation_penalty = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=kick_rewards.lin_vel_z_l2, weight=-1.0)
    track_lin_vel_xy = RewTerm(func=kick_rewards.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel_z = RewTerm(func=kick_rewards.track_ang_vel_z_exp, weight=1.0)
    dof_pos_limits = RewTerm(func=kick_rewards.dof_pos_limits, weight=-1.0)
    joint_dev_hip = RewTerm(func=kick_rewards.joint_deviation_hip, weight=-0.3)
    joint_dev_arms = RewTerm(func=kick_rewards.joint_deviation_arms, weight=-0.3)
    feet_slide = RewTerm(
        func=kick_rewards.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # Tracking (0 weight, logged to wandb)
    track_max_ball_vel_z = RewTerm(func=kick_rewards.track_max_ball_vel_z, weight=0.0)


@configclass
class H1JuggleTerminationsCfg:
    """Termination criteria."""

    time_out = DoneTerm(func=kick_terminations.time_out, time_out=True)
    robot_falls = DoneTerm(func=kick_terminations.torso_height_below)
    robot_out_of_bounds = DoneTerm(func=kick_terminations.robot_out_of_bounds)


@configclass
class H1JuggleActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class H1JuggleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree H1 Juggle Environment."""

    scene: H1JuggleSceneCfg = H1JuggleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: H1JuggleObservationsCfg = H1JuggleObservationsCfg()
    events: H1JuggleEventCfg = H1JuggleEventCfg()
    rewards: H1JuggleRewardsCfg = H1JuggleRewardsCfg()
    terminations: H1JuggleTerminationsCfg = H1JuggleTerminationsCfg()
    actions: H1JuggleActionsCfg = H1JuggleActionsCfg()

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
