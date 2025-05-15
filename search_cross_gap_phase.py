import numpy as np
from typing import Any, List
import os
import time

from mj_pin.simulator import Simulator
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from scene.cross_gap import setup_scene, SCENE_NAME
from configs_mpc_solver import *

# SCENE PARAM
GAP_LENGTH = 1.5
WALL_ANGLE = np.radians(65)  # Adjustable wall angle
h_offset = 0.1

sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT)
surfaces = setup_scene(sim, gap_length=GAP_LENGTH, wall_angle=WALL_ANGLE, height=h_offset/2, save_dir="",)
q0 = robot_description.q0
q0[2] += h_offset
sim.set_initial_state(q0)

if __name__ == "__main__":
    
    ITERATIONS = 1000
    C = 1.
    ALPHA = 0.33
    N_PHASES = 10
    GOAL = (1, 1, 1, 1)
    START_NODE = (0, (1, 1, 1, 1), (0, 0, 0, 0))
    MIN_RES = 0.
    MIN_AVG_COLLISION = 0.
    KEEP_SEQ_ID = True
    BINARY_REWARD = True
    MIN_IN_CNT = 0
    
    save_dir = os.path.join(BASE_SAVE_DIR, f"{SCENE_NAME}_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}")
    surfaces = setup_scene(sim, gap_length=GAP_LENGTH, wall_angle=WALL_ANGLE, height=h_offset/2, save_dir=save_dir,)

    os.makedirs(save_dir, exist_ok=True)
    # Copy the current script to the save directory
    current_script_path = os.path.abspath(__file__)
    destination_path = os.path.join(save_dir, os.path.basename(current_script_path))
    with open(current_script_path, 'r') as src, open(destination_path, 'w') as dst:
        dst.write(src.read())
        
    configs_mpc_path = os.path.join(os.path.dirname(current_script_path), "configs_mpc_solver.py")
    destination_path = os.path.join(save_dir, os.path.basename(configs_mpc_path))
    with open(configs_mpc_path, 'r') as src, open(destination_path, 'w') as dst:
        dst.write(src.read())
    
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
        save_dir=save_dir,
        keep_seq_id=KEEP_SEQ_ID,
        binary_reward=BINARY_REWARD,
        )
    
    mcts.run(START_NODE, ITERATIONS)
