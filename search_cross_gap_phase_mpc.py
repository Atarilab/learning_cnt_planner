import numpy as np
from typing import Any, List
import copy

from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.utils.solver import QuadrupedAcadosSolver
from mpc_controller.mpc_acyclic import AcyclicMPC, LocomotionMPC
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from scene.cross_gap import setup_scene
import yaml


SIM_DT = 1e-3
ROBOT_NAME = "go2"
RECOMPILE = False
N_OPT_NODES = 50
DURATION = 2.5
MAX_IT = 40

GAP_LENGTH = 0.3  # Adjustable gap between start and goal
WALL_ANGLE = np.radians(65)  # Adjustable wall angle
SIM_DT = 0.001
robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)
sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT)

################# Setup scene task
surfaces = setup_scene(sim, gap_length=GAP_LENGTH, wall_angle=WALL_ANGLE)

##################  Solver
# Opt
DT = 0.035
NODES = 35
config_opt = MPCOptConfig(
    time_horizon=NODES * DT,
    n_nodes=NODES,
    replanning_freq=25,
    Kp=30,
    Kd=7.,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=7,
    opt_peak=True,
    warm_start_sol=True,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed,
)

# Cost
def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

W = [
        0e0, 0e0, 0e0,      # Base position weights
        1e1, 4e1, 4e1,      # Base orientation (ypr) weights
        1e0, 1e0, 5e0,      # Base linear velocity weights
        5e0, 3e1, 3e1,      # Base angular velocity weights
    ]

HSE_SCALE = [15., 5., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W, 0.5),
    W_base=__init_np(W, 5.),
    W_joint=__init_np(HSE_SCALE + [0.02] * len(HSE_SCALE), 5.),
    W_e_joint=__init_np(HSE_SCALE + [0.01] * len(HSE_SCALE), 0.1),
    W_acc=__init_np(HSE_SCALE, 5.e-4),
    W_swing=__init_np([2e4] * n_feet),
    W_eeff_ori=__init_np([10.] * n_feet),
    W_cnt_f_reg = __init_np([[0.01, 0.01, 0.05]] * n_feet),
    W_foot_pos_constr_stab = __init_np([1e1] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

config_gait = GaitConfig(
    "acyclic",
    DURATION,
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    0.3,
    0.05,
)

mpc = AcyclicMPC(
    robot_description.urdf_path,
    pin_feet_frames,
    config_opt,
    config_cost,
    config_gait,
    joint_ref=robot_description.q0,
    sim_dt=SIM_DT,
    height_offset=0.,
    print_info=False,
    compute_timings=True,
    solve_async=False,
)

if __name__ == "__main__":
    
    ITERATIONS = 5000
    C = 5.
    ALPHA = 1.
    N_PHASES = 6
    GOAL = (1, 1, 1, 1)
    START_NODE = (0, (1, 1, 1, 1), (0, 0, 0, 0))
    
    # Trot 
    # phase_sequence = [
    #     (0, (1, 1, 1, 1), (0, 0, 0, 0)),
    #     (0, (1, 1, 1, 1), (0, 0, 0, 0)),
    #     (0, (0, 1, 1, 0), (0, 0)),
    #     (1, (1, 0, 0, 1), (1, 0)),
    #     (2, (0, 1, 1, 0), (1, 0)),
    #     (3, (1, 0, 0, 1), (1, 1)),
    #     (2, (0, 1, 1, 0), (1, 1)),
    #     (6, (1, 1, 1, 1), (1, 1, 1, 1))
    #     ]
    
    # Cross using walls
    phase_sequence = [
        (0, (1, 1, 1, 1), (0, 0, 0, 0)),
        (0, (1, 1, 1, 1), (0, 0, 0, 0)),
        (1, (0, 0, 1, 1), (0, 0)),
        (2, (1, 1, 1, 1), (2, 3, 0, 0)),
        (1, (1, 1, 0, 0), (2, 3)),
        (2, (1, 1, 1, 1), (2, 3, 2, 3)),
        (3, (0, 1, 1, 1), (3, 2, 3)),
        (1, (1, 0, 1, 1), (1, 2, 3)),
        (1, (1, 1, 0, 1), (1, 1, 3)),
        (1, (1, 1, 1, 0), (1, 1, 1)),
        (6, (1, 1, 1, 1), (1, 1, 1, 1))
        ]
    
    from search.utils.save import save_phase_sequence_to_yaml, load_phase_sequence_from_yaml
    dir_path = "./data/cross_gap_mpc"
    save_phase_sequence_to_yaml(dir_path, phase_sequence)
    seq, n = load_phase_sequence_from_yaml(dir_path)
    print(seq == phase_sequence)
    # MCTS search 
    mcts = MCTSPhaseLocomotionTask(
        C=C,
        alpha_exploration=ALPHA,
        sim=sim,
        mpc_solver=mpc,
        mpc_close_loop=mpc,
        n_phases=N_PHASES,
        surfaces=surfaces,
        goal_surf_id=GOAL
        )
    print("Node per phase", mcts.node_per_phase)
    mcts.node_per_phase = 6
    duration = len(phase_sequence) * mcts.node_per_phase * (config_opt.time_horizon / config_opt.n_nodes)

    # Low horizon MPC
    start_phase = 0 if phase_sequence[0] else 1
    
    cnt_sequence, patches = mcts.get_sequence_patches_from_path(phase_sequence[start_phase:], mcts.node_per_phase)
    # cnt_sequence = cnt_sequence[:, :mcts.opt_nodes]
    patch_center, patch_rot, patch_size = mcts.get_contact_patch(cnt_sequence, patches, mcts.surfaces)
    mpc.set_cnt_plan(
        cnt_sequence,
        patch_center,
        patch_rot,
        patch_size
    )
    
    
    # q_open_loop_traj = mpc.open_loop(*sim.get_initial_state(), duration)
    # sim.visualize_trajectory(q_open_loop_traj)
    
    sim.run(duration+1., use_viewer=True, controller=mpc, record_video=True)
    mpc.print_timings()
    
    # # Solver
    # q_sol, v_sol, dt_sol = m30cts.run_solver(phase_sequence)
    # v = np.zeros_like(q_sol[0])
    # q_mj_traj = np.stack([mcts.mpc.solver.dyn.convert_to_mujoco(q, v)[0] for q in q_sol])
    # time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
    # # sim.visualize_trajectory(q_mj_traj, time_traj, record_video=False)
    # mpc.keep_solution_as_reference()
    # mpc.update_configs(config_cost_track)
    
    # # Open loop
    # q_open_loop_traj = mpc.open_loop(q_mj_traj[0], np.zeros(len(q_sol[0])), DURATION)
    # sim.visualize_trajectory(q_open_loop_traj)
    
    # # Close loop
    # mpc.reset()
    # sim.run(DURATION+1.5, controller=mpc)
    
    # mpc.plot_traj("tau")
    # mpc.plot_traj("f")
    # mpc.plot_traj("q")
    # mpc.plot_traj("v")
    # mpc.show_plots()
    
    # mcts.run(START_NODE, ITERATIONS)
