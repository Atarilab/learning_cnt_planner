import numpy as np
import argparse
import time
from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mj_pin.abstract import VisualCallback

from mpc_controller.mpc import LocomotionMPC
from scene.stepping_stones import MjSteppingStones
from search.utils.mcts import MCTSBase
from search.graph_gait_sequence import GaitParallelGraph
from main import ReferenceVisualCallback
import numpy as np
from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from typing import Any, List

from scene.primitives import Box, Surface
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.config.quadruped.mpc_gait import QuadrupedSlowTrot, QuadrupedTrot
from mpc_controller.utils.solver import QuadrupedAcadosSolver
from mpc_controller.utils.contact_planner import ContactPlanner, CustomContactPlanner
from main import ReferenceVisualCallback

SIM_DT = 1e-3
ROBOT_NAME = "go2"
RECOMPILE = False
N_OPT_NODES = 25
DURATION = 1.
MAX_IT = 30
V_DES = np.array([.8, 0., 0.])

def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)

################## Simulator
sim = Simulator(robot_description.xml_scene_path)
q0_mj, v0_mj = sim.get_initial_state()

##################  Solver
# Opt
config_opt = MPCOptConfig(
    time_horizon=DURATION,
    n_nodes=N_OPT_NODES,
    replanning_freq=0, Kp=0, Kd=0,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=10,
    opt_peak=True,
    warm_start_sol=False,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed
)
dt_nodes = config_opt.get_dt_nodes()

config_gait = GaitConfig(
    "custom",
    nominal_period=0.5,
    stance_ratio=[0.5, 0.5, 0.5, 0.5],
    phase_offset=[0.5, 0.5, 0.5, 0.5],
    nom_height=0.3,
    step_height=0.05
)
config_gait = QuadrupedTrot(nominal_period=0.5, nom_height=0.3, step_height=0.05)
cnt_planner = CustomContactPlanner(pin_feet_frames, dt_nodes, config_gait)

# Cost
W = [
        0e0, 0e0, 1e1,      # Base position weights
        1e0, 1e0, 1e0,      # Base orientation (ypr) weights
        1e2, 1e2, 2e2,      # Base linear velocity weights
        1e0, 1e1, 1e1,      # Base angular velocity weights
    ]
W_e = [
        1e2, 1e2, 1e0,      # Base position weights
        1e0, 1e1, 1e1,      # Base orientation (ypr) weights
        1e2, 1e2, 1e2,      # Base linear velocity weights
        1e0, 1e1, 1e1,      # Base angular velocity weights
    ]
HSE_SCALE = [25., 5., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W_e, 1.),
    W_base=__init_np(W, 1.),
    W_joint=__init_np(HSE_SCALE + [0.5] * len(HSE_SCALE), 1.),
    W_e_joint=__init_np(HSE_SCALE + [0.1] * len(HSE_SCALE), 5.),
    W_acc=__init_np(HSE_SCALE, 1.e-3),
    W_swing=__init_np([5e3] * n_feet),
    W_cnt_f_reg = __init_np([[0.01, 0.01, 0.01]] * n_feet, 1),
    W_foot_pos_constr_stab = __init_np([1e2] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

# Init solver
base_ref = np.zeros(12)
base_ref[2] = config_gait.nom_height
base_ref[:3] += V_DES * DURATION
base_ref[6:9] = V_DES
base_ref_e = base_ref.copy()

solver = QuadrupedAcadosSolver(
    robot_description.urdf_path,
    pin_feet_frames,
    config_opt,
    config_cost,
    height_offset = 0.,
    print_info = False,
    compute_timings = False,
    )
solver.set_contact_restriction(True)
q0, v0 = solver.dyn.convert_from_mujoco(q0_mj, v0_mj)

class MCTSGaitSequence(MCTSBase):
    def __init__(self, graph, C = 0.01, increase_res_it = []):
        super().__init__(graph, C)
        self.set_resolution(1)
        self.increase_res_it = increase_res_it
        self.max_sim_step = 1
        self.max_reward = 0.
        
    def set_resolution(self, res : int) -> None:
        self.resolution = res
        self.graph.set_resolution(res)
        print("resolution set to", res)
        
    def UCB(self, node):
        ucb = super().UCB(node)
        if node not in self.value_visit:
            return ucb  # Encourage exploration of unexplored nodes
        # Explore nodes with minimal number of swich first
        _, n = self.value_visit[node]
        if n == 0:
            return ucb
        
        return ucb + sum(sum(a != b for a, b in zip(seq, seq[1:])) for seq in node) / n

    def resize(self, node, L : int) -> np.ndarray:
        """
        resize node of size n to size <L>.
        if L > len(node): interpolate
        else: take regularly spaced
        """

        resized = np.array(
            [np.fromiter(
                (string[i] for i in np.round(np.linspace(0,1,L)*(len(string)-1)).astype(np.int32)), dtype=np.int8)
             for string in node]
        )
        arr_resized = np.array(resized)
        return arr_resized
    
    def select(self, node):
        """
        Traverse the tree using UCB until an unexplored node is found.
        """
        node = super().select(node)

        if len(self.current_search_path) == 1:
            self.set_resolution(self.resolution + 1)
            node = super().select(node)
            
        return node
            
    def evaluate(self, simulation_path : list) -> float:

        cnt_sequence = self.resize(simulation_path[-1], cnt_planner.nodes_per_cycle)
        if np.any(np.all(cnt_sequence == cnt_sequence[:, [0]], axis=1)):
            return 0.

        solver.reset()
        solver.set_contact_restriction(False)
        solver.dyn.update_pin(q0, v0)
        cnt_planner.set_periodic_sequence(cnt_sequence)
        cnt_sequence = cnt_planner.get_contacts(0, N_OPT_NODES+1)
        peak_sequence = np.ones_like(cnt_sequence) - cnt_sequence

        solver.init(
            i_node=0,
            q=q0,
            v=v0,
            base_ref=base_ref,
            base_ref_e=base_ref_e,
            joint_ref=q0[-12:],
            step_height=config_gait.step_height,
            cnt_sequence=cnt_sequence,
            cnt_locations=None,
            swing_peak=peak_sequence,
        )
        try:
            q_sol, v_sol, _, _, dt_sol = solver.solve()
            solver
        except:
            return 0.

        # Compute reward
        reward = 1
        avg_vel_error = np.mean(np.abs(V_DES[:2] - v_sol[:, :2]))
        # avg_vel_error = np.mean(np.abs(base_ref_e[:2] - q_sol[-1, :2]))
        W_VEL = 3.
        reward *= np.exp(- W_VEL * avg_vel_error)
        n_qp = solver.solver.get_stats("nlp_iter")
        W_NQP = .1
        reward *= np.exp(- W_NQP * n_qp/MAX_IT)

        if reward > self.max_reward:
            print(simulation_path[-1], reward, avg_vel_error, n_qp)
            self.max_reward = reward
            time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
            q_mj_sol = np.stack([solver.dyn.convert_to_mujoco(q, v)[0] for q, v in zip(q_sol, v_sol)])
            sim.visualize_trajectory(q_mj_sol, time_traj, record_video=False)
        
        return reward
    
    def best_sequence(self, root, max_resolution : int):
        """
        Returns the best gait sequence with a given resolution.
        """
        path = [root]
        node = root

        # Take maximum average reward child
        while (node in self.value_visit 
               and self.value_visit[node][1] > 0
               and not self.is_leaf(node)):
            
            children = self.graph.get_neighbors(node)
            
            node = max(children, key=lambda child: self.value_visit[child][0] / (self.value_visit[child][1]+1))
            res = max((np.log(len(n)) / np.log(2) for n in node))
            if path[-1] == node or res > max_resolution:
                break
            path.append(node)

        return path

N_NODES = 8
ITERATIONS = 600
graph = GaitParallelGraph(len(pin_feet_frames), N_NODES)

N_STEP = graph.base - 1
increase_res_it = np.logspace(0, np.log10(ITERATIONS), N_STEP+2, dtype=np.int32)[1:-1]
mcts = MCTSGaitSequence(graph, C=1.5, increase_res_it=increase_res_it)
mcts.run(graph.start_node, ITERATIONS)

best_node = mcts.best_sequence(('1', '1', '1', '1'), max_resolution=2)
print(best_node)
# vis_callback = ReferenceVisualCallback(mpc)

# sim.vs.set_high_quality()
# sim.vs.track_obj = "base"
# sim.run(
#     use_viewer=True,
#     controller=mpc,
#     visual_callback=vis_callback,
#     record_video=False
#     )

# mpc.print_timings()