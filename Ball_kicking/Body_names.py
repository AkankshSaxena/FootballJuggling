from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation
from isaaclab_assets import H1_MINIMAL_CFG  # or H1_CFG, whichever your env actually uses
import isaaclab.sim as sim_utils

sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

robot_cfg = H1_MINIMAL_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

sim.reset()
print("BODY NAMES:", robot.data.body_names)

simulation_app.close()