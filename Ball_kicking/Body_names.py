from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import torch
from isaaclab.assets import Articulation
from isaaclab_assets import H1_CFG
import isaaclab.sim as sim_utils

sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

robot_cfg = H1_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

sim.reset()

LEG = "left"
hip_idx = robot.data.body_names.index(f"{LEG}_hip_pitch_link")
ankle_idx = robot.data.body_names.index(f"{LEG}_ankle_link")
knee_ids, knee_names = robot.find_joints([f"{LEG}_knee.*"])
assert len(knee_ids) == 1, f"Expected exactly 1 knee joint match, got {knee_names}"
knee_id = knee_ids[0]
limits = robot.data.default_joint_pos_limits[0, knee_id]
print(
    f"{LEG}_knee joint limits: [{limits[0]:.3f}, {limits[1]:.3f}] rad "
    f"([{math.degrees(limits[0]):.1f}, {math.degrees(limits[1]):.1f}] deg)"
)
print(
    "  -> if lower bound is ~0.0, the 0.0='straight' assumption is consistent "
    "with a knee that can only flex one direction.\n"
)


def settle_and_measure(knee_angle: float, settle_steps: int = 60) -> float:
    """Drive knee to knee_angle via the actual position-TARGET path (not just a
    raw state override, which the implicit PD actuator undoes within one step),
    let it settle, then measure hip-ankle distance."""
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[0, knee_id] = knee_angle
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()

    for _ in range(settle_steps):
        sim.step()
        robot.update(sim.get_physics_dt())
        robot.write_data_to_sim()

    pos = robot.data.body_pos_w[0]
    return torch.norm(pos[hip_idx] - pos[ankle_idx]).item()


default_knee_angle = robot.data.default_joint_pos[0, knee_id].item()
print(
    f"Default knee joint angle: {default_knee_angle:.4f} rad "
    f"({math.degrees(default_knee_angle):.2f} deg)"
)

h = settle_and_measure(default_knee_angle)
print(f"h  (hip-ankle dist, spawn/knee-bent pose) = {h:.4f} m")

h_prime = settle_and_measure(0.0)
print(f"h' (hip-ankle dist, knee forced to 0.0)    = {h_prime:.4f} m")

if h >= h_prime:
    print(
        "\n*** WARNING: h >= h_prime. Either 0.0 is NOT the straight-knee value "
        f"for this joint convention, or the default pose is already straight. "
        f"Check the URDF/joint limits for {LEG}_knee above before trusting these "
        "numbers. ***"
    )
else:
    theta_c = math.degrees(math.acos(h / h_prime))
    print(f"\nOK: h < h_prime as expected. theta_c = acos(h/h') = {theta_c:.2f} deg")

simulation_app.close()
