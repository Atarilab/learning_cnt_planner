import numpy as np
import os
from typing import Any, List

from mj_pin.utils import get_robot_description
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.mpc_acyclic import AcyclicMPC

FILE_NAME = os.path.split(__file__)[1]

BASE_SAVE_DIR = 'experiments/binary_reward/climb_box/height/0.5'
# SIM
ROBOT_NAME = "go2"
SIM_DT = 1/500
# CLOSE LOOP MPC
DT = 0.035
N_NODES_MPC = 50
# TRAJ OPT
RECOMPILE = False
N_NODES_SOLVER = 50
DURATION = 3 # Traj opt with a coarser discretization
MAX_IT = 75
MAX_QP = 7

##################  Solver
# Opt
config_solver = MPCOptConfig(
    time_horizon=DURATION,
    n_nodes=N_NODES_SOLVER,
    replanning_freq=1, Kp=1, Kd=1,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=7,
    opt_peak=True,
    warm_start_sol=False,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.balance,
    torque_limit=True,
    mu=0.7,
    solver_name="traj_opt_solver"
)

config_close_loop = MPCOptConfig(
    time_horizon=N_NODES_MPC * DT,
    n_nodes=N_NODES_MPC,
    replanning_freq=30,
    Kp=20.,
    Kd=2.5,
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
        0e0, 0e0, 0e0,      # Bas5e position weights
        5e0, 1e1, 3e1,      # Base orientation (ypr) weights
        1e0, 1e0, 1e-1,      # Base linear velocity weights
        5e0, 5e0, 5e0,      # Base angular velocity weights
    ]

W_e = [
        0e0, 0e0, 0e0,      # Base position weights
        1e0, 1e0, 1e0,      # Base orientation (ypr) weights
        1e1, 1e1, 1e1,      # Base linear velocity weights
        1e1, 1e1, 1e1,      # Base angular velocity weights
    ]

n_feet = 4
HSE_SCALE = [3., 2., 1.] *  n_feet

config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W_e, 50.),
    W_base=__init_np(W, 25.),
    W_joint=__init_np(HSE_SCALE + [0.3] * len(HSE_SCALE), 50.),
    W_e_joint=__init_np(HSE_SCALE + [1.] * len(HSE_SCALE), 50.),
    W_acc=__init_np(HSE_SCALE, 1.e-2),
    W_swing=__init_np([0.] * n_feet),
    W_eeff_ori=__init_np([1e2] * n_feet),
    W_cnt_f_reg = __init_np([[0.1, 0.1, 0.1]] * n_feet),
    W_foot_pos_constr_stab = __init_np([1e2] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([0.0e4]),
    W_swing_v = 300.,
    W_below_plane = 2000.,
    W_plane_border = 500.,
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

config_cost_close_loop = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W, 100.),
    W_base=__init_np(W, 150.),
    W_joint=__init_np(HSE_SCALE + [0.05] * len(HSE_SCALE), 10.),
    W_e_joint=__init_np(HSE_SCALE + [0.001] * len(HSE_SCALE), 0.1),
    W_acc=__init_np(HSE_SCALE, 7.e-3),
    W_swing=__init_np([0.] * n_feet),
    W_eeff_ori=__init_np([1e2] * n_feet),
    W_cnt_f_reg = __init_np([[0.1, 0.1, 0.1]] * n_feet),
    W_foot_pos_constr_stab = __init_np([1e3] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([0.0e4]),
    W_swing_v = 500.,
    W_below_plane = 5000.,
    W_plane_border = 200.,
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

config_gait = GaitConfig(
    "acyclic",
    DURATION,
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    0.3,
    0.06,
)


### INIT
robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]

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
    config_cost_close_loop,
    config_gait,
    joint_ref=robot_description.q0,
    sim_dt=SIM_DT,
    height_offset=0.,
    print_info=False,
    compute_timings=False,
    solve_async=True,
)
mpc_close_loop.config_opt.recompile = False
