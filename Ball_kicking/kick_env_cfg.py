import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# Import custom modules
import observations
import events
import rewards
import terminations
import curriculum


@configclass
class H1JuggleSceneCfg(InteractiveSceneCfg):
    """Configuration for the juggling scene."""

    # Ground plane
    terrain = sim_utils.GroundPlaneCfg()

    # Robot asset
    robot: ArticulationCfg = (
        MISSING  # Must be populated with H1 specific articulation config
    )

    # Ball asset
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            mass=0.43,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                restitution=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3)),
    )

    # Lights
    light = sim_utils.DomeLightCfg(intensity=2000.0)


@configclass
class H1JuggleObservationsCfg:
    """Observation specifications for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state (assuming base locomotion states are included here)
        base_lin_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_lin_vel_b)
        base_ang_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_ang_vel_b)
        joint_pos = ObsTerm(func=lambda env: env.scene["robot"].data.joint_pos)
        joint_vel = ObsTerm(func=lambda env: env.scene["robot"].data.joint_vel)

        # Custom Ball State
        ball_pos_robot_frame = ObsTerm(
            func=observations.ball_position_in_robot_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        ball_lin_vel_robot_frame = ObsTerm(
            func=observations.ball_linear_velocity_in_robot_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        last_contact_foot = ObsTerm(func=observations.last_contact_foot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class H1JuggleEventCfg:
    """Configuration for events."""

    reset_ball = EventTerm(
        func=events.reset_ball_position_and_velocity,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_tracking_vars = EventTerm(
        func=events.reset_custom_tracking_variables,
        mode="reset",
    )


@configclass
class H1JuggleRewardsCfg:
    """Reward configuration matching curriculum stages."""

    # Penalties
    lin_vel_z_l2 = RewTerm(func=rewards.lin_vel_z_l2, weight=-1.0)
    feet_slide = RewTerm(func=rewards.feet_slide, weight=-0.2)
    dof_pos_limits = RewTerm(func=rewards.dof_pos_limits, weight=-0.5)

    # Custom Ball & Juggling Rewards
    leg_raise = RewTerm(
        func=rewards.leg_raise_and_air_time,
        params={
            "sensor_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle", "right_ankle"]
            )
        },
        weight=2.0,
    )

    target_impulse = RewTerm(
        func=rewards.target_impulse_reward,
        params={"ball_cfg": SceneEntityCfg("ball"), "target_apex_height": 1.0},
        weight=5.0,
    )

    ball_apex = RewTerm(
        func=rewards.ball_apex_height_reward,
        params={"target_height": 1.0, "tolerance": 0.2},
        weight=5.0,
    )

    alternate_foot = RewTerm(func=rewards.alternate_foot_reward, weight=2.0)

    juggle_streak = RewTerm(func=rewards.juggle_streak_bonus, weight=1.0)

    # Custom Penalties
    ball_xy_velocity = RewTerm(
        func=rewards.track_ball_velocity_xy_exp,
        params={"ball_cfg": SceneEntityCfg("ball")},
        weight=1.0,  # Uses track_exp, positive weight encourages value to stay near 1 (vel=0)
    )

    ball_xy_drift = RewTerm(
        func=rewards.ball_xy_drift_penalty,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,  # Penalty function returns negative natively
    )


@configclass
class H1JuggleTerminationsCfg:
    """Termination criteria."""

    time_out = DoneTerm(func=terminations.time_out, time_out=True)

    robot_falls = DoneTerm(
        func=terminations.torso_height_below,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )

    ball_ground_contact = DoneTerm(
        func=terminations.ball_ground_contact_with_delay,
        params={"ball_cfg": SceneEntityCfg("ball"), "delay_s": 2.0},
    )


@configclass
class H1JuggleCurriculumCfg:
    """Curriculum configurations."""

    stage_progression = CurrTerm(func=curriculum.advance_curriculum_stage)


@configclass
class H1JuggleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree H1 Juggle Environment."""

    # Core
    scene: H1JuggleSceneCfg = H1JuggleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: H1JuggleObservationsCfg = H1JuggleObservationsCfg()
    events: H1JuggleEventCfg = H1JuggleEventCfg()
    rewards: H1JuggleRewardsCfg = H1JuggleRewardsCfg()
    terminations: H1JuggleTerminationsCfg = H1JuggleTerminationsCfg()
    curriculum: H1JuggleCurriculumCfg = H1JuggleCurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 15.0
        self.decimation = 4
        self.sim.dt = 0.005  # 200Hz

        # Override to headless constraints mapped via CLI in Isaac Lab
