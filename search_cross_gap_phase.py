import numpy as np
from typing import Any, List

from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, HPIPM_MODE
from mpc_controller.utils.solver import QuadrupedAcadosSolver
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from scene.cross_gap import setup_scene


SIM_DT = 1e-3
ROBOT_NAME = "go2"
RECOMPILE = False
N_OPT_NODES = 50
DURATION = 2.
MAX_IT = 40

GAP_LENGTH = 0.7  # Adjustable gap between start and goal
WALL_ANGLE = np.radians(65)  # Adjustable wall angle

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)
sim = Simulator(robot_description.xml_scene_path)

################# Setup scene task
surfaces = setup_scene(sim, gap_length=GAP_LENGTH, wall_angle=WALL_ANGLE)

##################  Solver
# Opt
config_opt = MPCOptConfig(
    time_horizon=DURATION,
    n_nodes=N_OPT_NODES,
    replanning_freq=0, Kp=0, Kd=0,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=12,
    opt_peak=True,
    warm_start_sol=False,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.robust
)

# Cost
def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

W = [
        0e0, 0e0, 1e0,      # Base position weights
        1e0, 1e0, 1e0,      # Base orientation (ypr) weights
        1e1, 1e1, 1e2,      # Base linear velocity weights
        2e1, 4e1, 4e1,      # Base angular velocity weights
    ]

HSE_SCALE = [15., 5., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W, 1.),
    W_base=__init_np(W, 2.),
    W_joint=__init_np(HSE_SCALE + [1.] * len(HSE_SCALE), 3.),
    W_e_joint=__init_np(HSE_SCALE + [.01] * len(HSE_SCALE), 5.),
    W_acc=__init_np(HSE_SCALE, 1.e-3),
    W_swing=__init_np([5e3] * n_feet),
    W_eeff_ori=__init_np([1e-1] * n_feet),
    W_cnt_f_reg = __init_np([[0.03, 0.03, 0.05]] * n_feet),
    W_foot_pos_constr_stab = __init_np([1e1] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

solver = QuadrupedAcadosSolver(
    robot_description.urdf_path,
    pin_feet_frames,
    config_opt,
    config_cost,
    height_offset = 0.,
    print_info = False,
    compute_timings = False,
    )


if __name__ == "__main__":
    
    ITERATIONS = 5000
    C = 5.
    ALPHA = 1.
    N_PHASES = 6
    GOAL = (1, 1, 1, 1)
    START_NODE = (0, (1, 1, 1, 1), (0, 0, 0, 0))
        
    # MCTS search 
    mcts = MCTSPhaseLocomotionTask(
        C=C,
        alpha_exploration=ALPHA,
        sim=sim,
        solver=solver,
        n_phases=N_PHASES,
        surfaces=surfaces,
        goal_surf_id=GOAL
        )
    
    mcts.run(START_NODE, ITERATIONS)
