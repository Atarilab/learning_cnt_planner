import numpy as np
from typing import Any, List
import time
import os

from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from configs_mpc_solver import *
from search.utils.save import load_phase_sequence_from_yaml

from scene.climb_box import setup_scene, SCENE_NAME
HEIGHT = 0.21
EDGE = 0.4
OFFSET = 0.4

sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT)
surfaces = setup_scene(sim, height=HEIGHT, edge=EDGE, offset=OFFSET, save_dir="", vis_normal=False)
if HEIGHT <= 0.2:
    surfaces = surfaces[:2]
    
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

    # # BATCH TEST MPC
    # EXP_DIR = "experiments/climb_box/height/0.2"
    # WITH_VIEWER = False
    # TRAJ_DIRS = [
    #     os.path.join(EXP_DIR, run_dir, traj_dir)
    #     for run_dir in os.listdir(EXP_DIR)
    #     if os.path.isdir(os.path.join(EXP_DIR, run_dir))
    #     for traj_dir in os.listdir(os.path.join(EXP_DIR, run_dir))
    #     if os.path.isdir(os.path.join(EXP_DIR, run_dir, traj_dir))
    #     and "__pycache__" not in traj_dir
    # ]
    # print("N_TRAJ", len(TRAJ_DIRS))
    # # MAX 260
    # for i, traj_dir in enumerate(TRAJ_DIRS):
    #     print("traj dir", i, traj_dir)
    #     phase_sequence, nodes_per_seq = load_phase_sequence_from_yaml(traj_dir)
    #     phase_sequence = [phase_sequence[0]] + phase_sequence
    #     nodes_per_seq = 7
        
    #     if WITH_VIEWER:
    #         mcts.run_mpc(phase_sequence, nodes_per_seq, record_video=False, use_viewer=True)
        
    #         user_input = input("Press Enter to skip trajectory")
    #         if user_input != "":
    #             print("Saving trajectory directory...")
    #             with open("saved_trajectories.txt", "a") as file:
    #                 file.write(traj_dir + "\n")
    #     else:
    #         success = mcts.run_mpc(phase_sequence, nodes_per_seq, record_video=False, use_viewer=False)
    #         print("Trajectory", i, "success", success)
    #         if success:
    #             with open("saved_trajectories_success.txt", "a") as file:
    #                 file.write(traj_dir + "\n")
                
    trajectory_dir = "./experiments/climb_box/height/0.2/climb_box_2025-05-07_16-24-18/collision_free_iteration_526"
    trajectory_dir = "./experiments/climb_box/height/0.2/climb_box_2025-05-07_16-00-48/collision_free_iteration_791"
    # trajectory_dir = "./experiments/climb_box/height/0.2/climb_box_2025-05-07_16-00-48/collision_free_iteration_795"
    # trajectory_dir = "./experiments/climb_box/height/0.2/climb_box_2025-05-07_16-47-46/collision_free_iteration_75"
    trajectory_dir = ""
    if trajectory_dir:
        phase_sequence, nodes_per_seq = load_phase_sequence_from_yaml(trajectory_dir)
    
    if not trajectory_dir:
        phase_sequence = [
            (0, (1, 1, 1, 1), (0, 0, 0, 0)),
            (0, (1, 1, 1, 1), (0, 0, 0, 0)),
            (1, (1, 1, 1, 1), (0, 0, 0, 0)),
            (2, (1, 1, 0, 1), (0, 0, 0)),
            (3, (1, 1, 1, 0), (0, 0, 0)),
            (4, (0, 1, 1, 1), (0, 0, 0)),
            (5, (1, 0, 1, 1), (1, 0, 0)),
            (6, (1, 1, 1, 0), (1, 1, 0)),
            (7, (1, 1, 1, 1), (1, 1, 0, 0)),
            (8, (1, 1, 0, 1), (1, 1, 0)),
            (9, (1, 1, 1, 1), (1, 1, 1, 0)),
            # (10, (1, 0, 1, 1), (1, 1, 0)),
            # (10, (0, 1, 1, 1), (1, 1, 0)),
            (11, (0, 1, 1, 1), (1, 1, 1)),
            (11, (1, 0, 1, 1), (1, 1, 1)),
            # (10, (1, 1, 1, 0), (1, 1, 1)),
            (10, (1, 1, 1, 0), (1, 1, 1)),
            (9, (1, 1, 1, 1), (1, 1, 1, 1)),
            (12, (1, 1, 1, 1), (1, 1, 1, 1))
            ]
    for phase in phase_sequence:
        print(phase)
    nodes_per_seq = 10
    # Repeat first
    phase_sequence = [phase_sequence[0]] + phase_sequence
    success = mcts.run_mpc(phase_sequence, nodes_per_seq, record_video=True, use_viewer=True)
    print("Success", success)