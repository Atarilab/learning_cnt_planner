import numpy as np
from typing import Any, List
import time
import os

from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from configs_mpc_solver import *

# from scene.climb_box import setup_scene, SCENE_NAME
# HEIGHT = 0.4
# EDGE = 0.4
# OFFSET = 0.5

# sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT)
# surfaces = setup_scene(sim, height=HEIGHT, edge=EDGE, offset=OFFSET, save_dir="", vis_normal=False)
# if HEIGHT <= 0.3:
#     surfaces = surfaces[:2]

# phase_sequence = [
#     (0, (1, 1, 1, 1), (0, 0, 0, 0)),
#     (0, (0, 1, 1, 1), (0, 0, 0)),
#     (0, (1, 0, 1, 1), (1, 0, 0)),
#     (0, (1, 1, 1, 0), (1, 1, 0)),
#     (0, (1, 1, 0, 1), (1, 1, 0)),
#     (0, (1, 0, 1, 1), (1, 0, 0)),
#     (0, (0, 1, 1, 1), (1, 0, 0)),
#     (0, (1, 1, 1, 1), (1, 1, 0, 0)),
#     # (0, (1, 1, 1, 1), (1, 1, 0, 0)),
#     (0, (1, 1, 1, 0), (1, 1, 0)),
#     (0, (1, 1, 0, 1), (1, 1, 1)),
#     (0, (1, 1, 1, 1), (1, 1, 1, 1)),
# ]

    
from scene.cross_gap import setup_scene, SCENE_NAME
GAP_LENGTH = 0.9  # Adjustable gap between start and goal
WALL_ANGLE = np.radians(65)  # Adjustable wall angle
h_offset = 0.1

sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT)
surfaces = setup_scene(sim, gap_length=GAP_LENGTH, wall_angle=WALL_ANGLE, height=h_offset/2, save_dir="",)
q0 = robot_description.q0
q0[2] += h_offset
sim.set_initial_state(q0)


phase_sequence = [
    (0, (1, 1, 1, 1), (0, 0, 0, 0)),
    (0, (1, 1, 1, 1), (0, 0, 0, 0)),
    (1, (0, 0, 1, 1), (0, 0)),
    (2, (1, 1, 1, 1), (2, 3, 0, 0)),
    (1, (1, 1, 1, 0), (2, 3, 0)),
    (1, (1, 1, 0, 1), (2, 3, 0)),
    (1, (1, 1, 0, 0), (2, 3,)),
    (2, (1, 1, 1, 1), (2, 3, 2, 3)),
    (3, (0, 1, 1, 1), (3, 2, 3)),
    (1, (1, 0, 1, 1), (1, 2, 3)),
    (1, (1, 1, 0, 1), (1, 1, 3)),
    (1, (1, 1, 1, 0), (1, 1, 1)),
    (6, (1, 1, 1, 1), (1, 1, 1, 1))
    ]
    
if __name__ == "__main__":
    
    ITERATIONS = 2000
    C = 1.5
    ALPHA = 0.5
    N_PHASES = 12
    GOAL = (1, 1, 1, 1)
    START_NODE = (0, (1, 1, 1, 1), (0, 0, 0, 0))
    MIN_IN_CNT = 2
    MIN_RES = 0.
    MIN_AVG_COLLISION = 0.
    
    # MCTS search 
    mcts = MCTSPhaseLocomotionTask(
        C=C,
        alpha_exploration=ALPHA,
        sim=sim,
        mpc_solver=mpc_solver,
        mpc_close_loop=mpc_close_loop,
        n_phases=N_PHASES,
        surfaces=surfaces,
        min_in_cnt=MIN_IN_CNT,
        goal_surf_id=GOAL,
        min_mpc_log10_prod_res=MIN_RES,
        min_mpc_avg_collision=MIN_AVG_COLLISION,
        save_dir=""
        )

    q_sol, v_sol, dt_sol = mcts.run_traj_opt(phase_sequence)
    q_mj_traj = np.stack([mcts.mpc_solver.solver.dyn.convert_to_mujoco(q, v)[0] for q, v in zip(q_sol, v_sol)])
    time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
    mcts.sim.visualize_trajectory(q_mj_traj, time_traj, record_video=False)