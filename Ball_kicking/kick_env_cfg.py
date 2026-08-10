import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets import H1_CFG
from isaaclab.sensors import ContactSensorCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    kick_events,
    kick_observations,
    kick_rewards,
    kick_terminations,
)

SWING_THETA_MAX_DEG = 60.0
SWING_TIME = 0.8
SWING_PERIOD = 0.8


@configclass
class H1JuggleSceneCfg(InteractiveSceneCfg):
    """Scene: robot, ball, ground, contact sensors."""

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

    robot: ArticulationCfg = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                linear_damping=0.1,
                angular_damping=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.8,
            ),
        ),
        # Pre-reset spawn (reset_ball re-places it every episode). y=-0.08 = robot's
        # right (right-foot side), matching reset_ball's lateral_offset=0.08.
        # NOTE: with randomize_side=True on reset_ball below, actual per-episode side
        # is randomized -- this init_state is only the very first pre-reset pose.
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.47, -0.075, 0.24)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.075, 3.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    # Ground/self contact (feet_slide, one_foot_ground_contact). No filter -> the
    # multi-body regex prim_path is fine here.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(pelvis|torso_link|.*_knee_link|.*_ankle_link|.*_elbow_link)",
        history_length=3,
        track_air_time=True,
    )

    # Ball-filtered sensors: ONE per body (PhysX filter pairs unreliable on a
    # multi-body regex prim_path -> force_matrix_w read all zero). Do not merge.
    # ---- Ball-filtered illegal-contact sensors: ONE per body ----
    left_shoulder_pitch_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_shoulder_pitch_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    left_shoulder_roll_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_roll_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_shoulder_roll_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    left_shoulder_yaw_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_shoulder_yaw_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    left_elbow_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elbow_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_elbow_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_elbow_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    left_ankle_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_ankle_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    pelvis_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    torso_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    left_knee_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )
    right_knee_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_knee_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball.*"],
    )


@configclass
class H1JuggleObservationsCfg:
    """Policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        ball_pos_robot_frame = ObsTerm(
            func=kick_observations.ball_position_in_robot_frame
        )
        ball_lin_vel_robot_frame = ObsTerm(
            func=kick_observations.ball_linear_velocity_in_robot_frame
        )
        feet_position_in_robot_frame = ObsTerm(
            func=kick_observations.feet_position_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_link", "right_ankle_link"],
                    preserve_order=True,
                )
            },
        )
        knees_position_in_robot_frame = ObsTerm(  # optional (redundant w/ joint_pos)
            func=kick_observations.knees_position_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_knee_link", "right_knee_link"],
                    preserve_order=True,
                )
            },
        )
        # REQUIRED for foot_swing_knee_extend -- without it the triangular swing
        # target is unobservable and unlearnable. theta_max_deg/swing_time/period
        # MUST match the foot_swing_knee_extend RewTerm params below exactly.
        swing_phase = ObsTerm(
            func=kick_observations.swing_phase,
            params={
                "theta_max_deg": SWING_THETA_MAX_DEG,
                "swing_time": SWING_TIME,
                "period": SWING_PERIOD,
            },
        )
        last_contact_foot = ObsTerm(func=kick_observations.last_contact_foot)

        target_leg = ObsTerm(func=kick_observations.target_leg_command)

        def __post_init__(self):
            self.enable_corruption = (
                False  # if enabled for Stage 3, exclude swing_phase
            )
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class H1JuggleEventCfg:
    """Events. reset_* run in declaration order -> robot reset BEFORE ball,
    since reset_ball reads the robot pose to place the ball."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # STAGE 0: park overhead (0 forward, 3 up). STAGE 1: distance_offset=0.5-1.0,
    # height_offset=0.3. STAGE 2+: reduce further / randomize.
    # lateral_offset > 0 shifts the ball toward the robot's RIGHT (right-foot side)
    # so it sits under the kicking foot's swing arc, which is off-centre in y.
    # randomize_side=True: the SIGN of lateral_offset is randomized per env per
    # episode (magnitude unchanged) -- this is what actually drives bilateral
    # training. Also sets env.active_leg directly per env from the spawn side (see
    # reset_ball_state docstring in kick_events.py). Without this flag, every env
    # spawns on the fixed right side and active_leg never becomes left.
    reset_ball = EventTerm(
        func=kick_events.reset_ball_state,
        mode="reset",
        params={
            "distance_offset": 0.0,
            "lateral_offset": 0.075,
            "height_offset": 3.0,
            "randomize_side": True,
        },
    )

    # STAGE 0/1: min_height parks/pins the ball on Z. STAGE 2: DISABLE this whole
    # term so the ball flies on contact.
    # follow_robot=True: ball tracks the robot (stays distance_offset in front, on
    # whichever side reset_ball randomized to, as it walks/turns), reusing
    # reset_ball's per-env distance/lateral offsets. Set follow_robot=False to pin
    # the ball at its fixed spawn anchor instead.
    constrain_ball = EventTerm(
        func=kick_events.constrain_ball_to_z_axis,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every env step
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "min_height": 3.0,
            "follow_robot": True,
        },
    )

    # ball_gravity = EventTerm(
    #     func=kick_events.ball_gravity_scale,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),  # every env step
    #     params={"ball_cfg": SceneEntityCfg("ball"), "k": 1.0},
    # )


@configclass
class H1JuggleRewardsCfg:
    """All terms present for every stage; weights set for STAGE 0.
    Per-stage weight changes noted inline."""

    # Regularization
    termination_penalty = RewTerm(func=kick_rewards.termination_penalty, weight=-100.0)
    orientation_penalty = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    lin_vel_z_l2 = RewTerm(func=kick_rewards.lin_vel_z_l2, weight=-1.5)

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
        weight=-0.1,
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
    feet_slide = RewTerm(
        func=kick_rewards.feet_slide,
        weight=-0.35,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    track_ang_vel_z = RewTerm(
        func=kick_rewards.track_ang_vel_z_exp, weight=1.0, params={"std": 0.5}
    )
    track_lin_vel_xy = RewTerm(
        func=kick_rewards.track_lin_vel_xy_exp, weight=1.0, params={"std": 0.5}
    )

    # Stability
    ball_robot_dist = RewTerm(
        func=kick_rewards.ball_robot_dist_reward,
        weight=0.3,
        params={"kick_range": 0.0, "std": 1.0},
    )
    one_foot_ground_contact = RewTerm(
        func=kick_rewards.one_foot_ground_contact,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_link"),
            "force_threshold": 1.0,
        },
    )
    # Bilateral swing target. active_leg (and its ball-Y-sign hysteresis update) is
    # computed INSIDE this reward function -- see kick_rewards.foot_swing_knee_extend.
    # h_left/h_prime_left are PLACEHOLDERS (copied from right) -- measure the actual
    # left hip->ankle geometry before trusting Stage 1 results (see spec §3.5).
    foot_swing_knee_extend = RewTerm(
        func=kick_rewards.foot_swing_knee_extend,
        weight=4.0,
        params={
            "right_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_hip_pitch_link", "right_ankle_link"],
                preserve_order=True,
            ),
            "left_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_hip_pitch_link", "left_ankle_link"],
                preserve_order=True,
            ),
            "h_right": 0.7384,
            "h_prime_right": 0.80,
            "h_left": 0.7384,  # PLACEHOLDER -- not measured
            "h_prime_left": 0.80,  # PLACEHOLDER -- not measured
            "theta_max_deg": SWING_THETA_MAX_DEG,
            "swing_time": SWING_TIME,
            "period": SWING_PERIOD,
            "std": 0.15,
        },
    )

    # Juggling
    ball_foot_contact = RewTerm(
        func=kick_rewards.ball_foot_contact_reward,
        weight=0.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_ankle_ball_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_ankle_ball_contact"),
            "min_peak_force": 25.0,
            "min_ball_vel_z": 1.2,
            "min_kick_interval_s": 0.4,
        },
    )
    ball_illegal_contact_penalty = RewTerm(
        func=kick_rewards.ball_illegal_contact_penalty,
        weight=-0.0,
        params={
            "illegal_sensor_cfgs": [
                SceneEntityCfg("pelvis_ball_contact"),
                SceneEntityCfg("torso_ball_contact"),
                SceneEntityCfg("left_knee_ball_contact"),
                SceneEntityCfg("right_knee_ball_contact"),
                SceneEntityCfg("left_shoulder_pitch_ball_contact"),
                SceneEntityCfg("right_shoulder_pitch_ball_contact"),
                SceneEntityCfg("left_shoulder_roll_ball_contact"),
                SceneEntityCfg("right_shoulder_roll_ball_contact"),
                SceneEntityCfg("left_shoulder_yaw_ball_contact"),
                SceneEntityCfg("right_shoulder_yaw_ball_contact"),
                SceneEntityCfg("left_elbow_ball_contact"),
                SceneEntityCfg("right_elbow_ball_contact"),
            ],
            "force_threshold": 0.1,
        },
    )
    apex_height = RewTerm(
        func=kick_rewards.apex_height_reward,
        weight=0.0,
        params={"apex_min": 0.8, "apex_max": 1.6},  # locked-constant band
    )

    # Debug
    log_kinematics = RewTerm(func=kick_rewards.log_kinematics, weight=1.0)

    # Later stage rewards
    ball_xy_force_penalty = RewTerm(
        func=kick_rewards.ball_xy_force_penalty,
        weight=0.0,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("left_ankle_ball_contact"),
                SceneEntityCfg("right_ankle_ball_contact"),
            ],
            "force_threshold": 0.1,
        },
    )
    track_ball_vel_xy = RewTerm(
        func=kick_rewards.track_ball_vel_xy_exp,
        weight=0.0,
        params={"std": 0.5},
    )
    track_ball_pos_xy = RewTerm(
        func=kick_rewards.track_ball_pos_xy_exp,
        weight=0.0,
        params={"std": 0.5, "reach": 0.5},
    )
    alternate_foot_bonus = RewTerm(
        func=kick_rewards.alternate_foot_bonus,
        weight=0.0,
    )


@configclass
class H1JuggleTerminationsCfg:
    time_out = DoneTerm(func=kick_terminations.time_out, time_out=True)
    robot_falls = DoneTerm(func=kick_terminations.torso_height_below)
    # max_distance vs env_spacing mismatch -- see note; lower to ~3.0 before ball flies.
    robot_out_of_bounds = DoneTerm(func=kick_terminations.robot_out_of_bounds)
    # ball_on_ground = DoneTerm(
    #     func=kick_terminations.ball_on_ground_timeout,
    #     time_out=False,
    #     params={"delay_s": 0.0, "ground_height": 0.15},
    # )


@configclass
class H1JuggleActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class H1JuggleEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H1 Juggle Environment (configured for STAGE 0)."""

    scene: H1JuggleSceneCfg = H1JuggleSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: H1JuggleObservationsCfg = H1JuggleObservationsCfg()
    events: H1JuggleEventCfg = H1JuggleEventCfg()
    rewards: H1JuggleRewardsCfg = H1JuggleRewardsCfg()
    terminations: H1JuggleTerminationsCfg = H1JuggleTerminationsCfg()
    actions: H1JuggleActionsCfg = H1JuggleActionsCfg()

    def __post_init__(self):
        self.episode_length_s = 25.0
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.physx.solver_type = 1
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        self.scene.ball.spawn.rigid_props.solver_position_iteration_count = 8
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_collision_stack_size = 2**26
