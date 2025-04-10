import numpy as np
from typing import Any, List

from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.mpc_acyclic import AcyclicMPC
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from scene.climb_box import setup_scene

# SIM
ROBOT_NAME = "go2"
SIM_DT = 1e-3
# CLOSE LOOP MPC
DT = 0.035
N_NODES_MPC = 35
# TRAJ OPT
RECOMPILE = False
N_NODES_SOLVER = 50
DURATION = N_NODES_SOLVER * DT * 2. # Traj opt with a coarser discretization
MAX_IT = 50
MAX_QP = 7
# SCENE PARAM
HEIGHT = 0.2
EDGE = 0.4

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)
sim = Simulator(robot_description.xml_scene_path)

################# Setup scene task
surfaces = setup_scene(sim, height=HEIGHT, edge=EDGE, vis_normal=False)
surfaces = surfaces[:2]

##################  Solver
# Opt
config_solver = MPCOptConfig(
    time_horizon=DURATION,
    n_nodes=N_NODES_SOLVER,
    replanning_freq=1, Kp=1, Kd=1,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=MAX_QP,
    opt_peak=True,
    warm_start_sol=False,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed,
    solver_name="traj_opt_solver"
)

config_close_loop = MPCOptConfig(
    time_horizon=N_NODES_MPC * DT,
    n_nodes=N_NODES_MPC,
    replanning_freq=25,
    Kp=30,
    Kd=7.,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=MAX_QP,
    opt_peak=True,
    warm_start_sol=True,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed,
    solver_name="mpc_solver"
)


# Cost
def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

W = [
        1e0, 1e0, 1e0,      # Base position weights
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

config_gait = GaitConfig(
    "acyclic",
    1.,
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    0.35,
    0.055,
)

mpc_solver = AcyclicMPC(
    robot_description.urdf_path,
    pin_feet_frames,
    config_solver,
    config_cost,
    config_gait,
    joint_ref=robot_description.q0,
    sim_dt=SIM_DT,
    height_offset=0.,
    print_info=False,
    compute_timings=False,
    solve_async=False,
)
mpc_solver.config_opt.recompile = False

mpc_close_loop = AcyclicMPC(
    robot_description.urdf_path,
    pin_feet_frames,
    config_close_loop,
    config_cost,
    config_gait,
    joint_ref=robot_description.q0,
    sim_dt=SIM_DT,
    height_offset=0.,
    print_info=False,
    compute_timings=False,
    solve_async=False,
)
mpc_close_loop.config_opt.recompile = False

if __name__ == "__main__":
    
    ITERATIONS = 5000
    C = 1.
    ALPHA = 0.75
    N_PHASES = 10
    GOAL = (1, 1, 1, 1)
    START_NODE = (0, (1, 1, 1, 1), (0, 0, 0, 0))
    MIN_RES = 0.
    MIN_AVG_COLLISION = 1.5
    
    # MCTS search 
    mcts = MCTSPhaseLocomotionTask(
        C=C,
        alpha_exploration=ALPHA,
        sim=sim,
        mpc_solver=mpc_solver,
        mpc_close_loop=mpc_close_loop,
        n_phases=N_PHASES,
        surfaces=surfaces,
        goal_surf_id=GOAL,
        min_mpc_log10_prod_res=MIN_RES,
        min_mpc_avg_collision=MIN_AVG_COLLISION,
        )
    
    mcts.run(START_NODE, ITERATIONS)
