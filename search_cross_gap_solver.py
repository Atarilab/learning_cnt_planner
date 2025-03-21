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
N_OPT_NODES = 50
DURATION = 2.
MAX_IT = 50

GAP_LENGTH = 0.4  # Adjustable gap between start and goal
WALL_ANGLE = np.radians(65)  # Adjustable wall angle

AVAILABLE_SURFACES = [0, 6, 12, 18]

def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)


def get_contact_patch(
    contact_seq: np.ndarray,
    contact_patches: List[List[int]],
    surfaces: List[Surface]
):
    """
    Determines the contact surface properties (center, normal, size) for each foot at each time node.

    :param contact_seq: (n_feet, n_nodes) binary contact sequence (1 if in contact, 0 otherwise)
    :param contact_patches: List of valid surface indices for each foot
    :param surfaces: List of available surfaces

    :return:
        - surface_centers: (n_feet, n_nodes, 3) array of surface centers per foot per time step
        - surface_normals: (n_feet, n_nodes, 3) array of surface normals per foot per time step
        - surface_sizes: (n_feet, n_nodes, 2) array of surface sizes per foot per time step (size_x, size_y)
    """
    n_feet, n_nodes = contact_seq.shape

    # Initialize output arrays with zeros
    surface_centers = np.zeros((n_feet, n_nodes, 3))
    surface_normals = np.zeros((n_feet, n_nodes, 3))
    surface_sizes = np.zeros((n_feet, n_nodes, 2))

    for i_foot in range(n_feet):
        cnt_phase = -1
        last_in_cnt = False
        for i_node in range(n_nodes):
            if contact_seq[i_foot, i_node] == 1:  # Foot is in contact
                # If make cnt
                if not last_in_cnt:
                    cnt_phase += 1
                last_in_cnt = True
            else:
                last_in_cnt = False
                
            # Select a surface from the valid patches
            i = min(cnt_phase, len(contact_patches[i_foot])-1)
            id_surf = AVAILABLE_SURFACES[contact_patches[i_foot][i]]
            surface = surfaces[id_surf]

            # Assign surface properties
            surface_centers[i_foot, i_node] = surface.center
            surface_normals[i_foot, i_node] = surface.normal
            surface_sizes[i_foot, i_node] = [surface.size_x, surface.size_y]             

    return surface_centers, surface_normals, surface_sizes


################## Start Box
# Parameters
thick = 0.01
large = 0.22

pos_start = [0.0, 0.0, -thick]
size_start = [large, large, thick]
euler_start = [0.0, 0.0, 0.0]
start = Box(pos_start, size_start, euler_start)

################## Walls
wall_offset = large + GAP_LENGTH / 2.0
wall_size = [GAP_LENGTH / 2., large, thick / 2.0]
wall_euler = [WALL_ANGLE, 0.0, 0.0]
wall_gap = large * (1 + np.sin(WALL_ANGLE) * 2) # Adjustable gap between walls

wall_1_pos = [wall_offset, wall_gap / 2., large + 0.0]
wall_2_pos = [wall_offset, -wall_gap / 2., large + 0.0]

wall_1 = Box(wall_1_pos, wall_size, wall_euler)
wall_2 = Box(wall_2_pos, wall_size, [-angle for angle in wall_euler])

################## End Box
pos_end = np.array(pos_start) + np.array([GAP_LENGTH + 2 * large, 0.0, 0.0])
size_end = [large, large, thick]
euler_end = [0.0, 0.0, 0.0]
end = Box(pos_end, size_end, euler_end)

################## Surfaces
plane = Surface(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), size_x=1e10, size_y=1e10)
surfaces = start.get_surfaces() + end.get_surfaces() + wall_1.get_surfaces() + wall_2.get_surfaces()

################## Simulator
sim = Simulator(robot_description.xml_scene_path)
sim.edit.add_box(pos_start, size_start, euler_start)
sim.edit.add_box(wall_1_pos, wall_size, wall_euler)
sim.edit.add_box(wall_2_pos, wall_size, [-angle for angle in wall_euler])
sim.edit.add_box(pos_end, size_end, euler_end)

q0_mj, v0_mj = sim.get_initial_state()

# Viz surfaces
scale_normal = 0.08
radius = 0.008
N_SHPERE = 8
for i in  AVAILABLE_SURFACES:
    s = surfaces[i]
    # normal and center
    for i in range(N_SHPERE):
        normal_v = s.center + s.normal * scale_normal * (i/N_SHPERE)
        # center
        if i == 0:
            sim.edit.add_sphere(normal_v, radius*2, color="black")
        # normal
        sim.edit.add_sphere(normal_v, radius, color="red")

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
        0e0, 1e0, 1e0,      # Base orientation (ypr) weights
        1e0, 1e0, 3e2,      # Base linear velocity weights
        1e0, 1e1, 1e1,      # Base angular velocity weights
    ]

HSE_SCALE = [15., 5., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W, 3.),
    W_base=__init_np(W, 1.),
    W_joint=__init_np(HSE_SCALE + [1.] * len(HSE_SCALE), 1.),
    W_e_joint=__init_np(HSE_SCALE + [.1] * len(HSE_SCALE), 1.),
    W_acc=__init_np(HSE_SCALE, 1.e-3),
    W_swing=__init_np([1e3] * n_feet),
    W_cnt_f_reg = __init_np([[0.01, 0.01, 0.05]] * n_feet),
    W_foot_pos_constr_stab = __init_np([5e1] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

# Init solver
base_ref = np.zeros(12)
base_ref[2] = config_gait.nom_height
base_ref_e = base_ref.copy()
# base_ref_e[2] += HEIGHT

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

patches = [
    [0, 2, 1],
    [0, 3, 1],
    [0, 2, 1],
    [0, 3, 1],
]

class MCTSTaskSequence(MCTSBase):
    def __init__(self, graph, C = 0.01, increase_res_it = []):
        super().__init__(graph, C)
        self.set_resolution(0)
        self.increase_res_it = increase_res_it
        self.max_sim_step = 1
        self.max_reward = 0.
        self.increase_step = 0
        
    def set_resolution(self, res : int) -> None:
        self.resolution = res
        self.graph.set_resolution(res)
        
    def cnt_start_end_score(self, node) -> float:
        return sum((int(n[0]) + int(n[-1]) for n in node)) / (len(node) * 2)

    def UCB(self, node) -> float:
        """
        Upper Confidence Bound (UCB) formula for MCTS.
        UCB = Q/N + C * sqrt(log(N_parent) / N)
        """
        return super().UCB(node) + self.cnt_start_end_score(node) / (self.value_visit[node][1] + 1)
    
    def rollout_policy(self, node):
        """
        Selects a random child node during rollout.
        """
        if self.is_leaf(node):
            return node
        children = self.graph.get_neighbors(node)
        np.random.shuffle(children)
        return max(children, key=lambda child : self.cnt_start_end_score(child))

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
    
    def simulate(self, node):
        for _ in range(self.max_sim_step): self.graph.increase_res()
        sim_path = super().simulate(node)
        for _ in range(self.max_sim_step): self.graph.decrease_res()
        return sim_path
    
    def select(self, node):
        """
        Traverse the tree using UCB until an unexplored node is found.
        """
        if (self.increase_step < len(self.increase_res_it) and
            self.it >= self.increase_res_it[self.increase_step]):
            print("Resolution set to", self.increase_step + 1)
            self.set_resolution(self.increase_step + 1)
            self.increase_step += 1
            # self.C /= 3.
            
        node = super().select(node)

        # if len(self.current_search_path) == 1:
        #     self.set_resolution(self.resolution + 1)
        #     node = super().select(node)
            
        return node
            
    def evaluate(self, simulation_path : list) -> float:
        seq = simulation_path[-1]
        cnt_sequence = self.resize(seq, N_OPT_NODES+1)
        # No switch
        if np.any(np.all(cnt_sequence == cnt_sequence[:, [0]], axis=1)):
            return 0.
        # Not in contact in the end
        if np.all(cnt_sequence[:, -1] == 0., axis=0):
            return 0.
        
        solver.reset()
        solver.set_contact_restriction(True)
        solver.dyn.update_pin(q0, v0)
        
        patch_center, patch_normal, patch_size = get_contact_patch(cnt_sequence, patches, surfaces)
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
            cnt_locations=patch_center,
            swing_peak=peak_sequence,
        )
        try:
            solver.setup_contact_patch(patch_center, patch_normal, patch_size)
            solver.update_solver()
            q_sol, v_sol, _, _, dt_sol = solver.solve()
            solver
        except:
            return 0.

        # Compute reward
        reward = 1.
        
        W_DIST = 1.
        reward *= np.exp(- W_DIST * np.abs(pos_end[0] - q_sol[-1, 0]))
        
        n_qp = solver.solver.get_stats("nlp_iter")
        W_NQP = 1.
        reward *= np.exp(- W_NQP * n_qp/MAX_IT)

        if reward > self.max_reward:
            print(simulation_path[-1], reward, q_sol[-1, 0], n_qp)
            time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
            q_mj_sol = np.stack([solver.dyn.convert_to_mujoco(q, v)[0] for q, v in zip(q_sol, v_sol)])
            sim.visualize_trajectory(q_mj_sol, time_traj, record_video=False)
            self.max_reward = reward
            self.max_reward_it = self.it
        
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

if __name__ == "__main__":
    
    USE_PHASE_GARPH = True
    N_NODES = 16
    ITERATIONS = 600
    
    if USE_PHASE_GARPH:
        from search.graph_phase_patch import GraphPhasePatch
        
        # Create graph
        N_PHASES = 5
        node_per_phase = N_OPT_NODES / N_PHASES
        patches = [i for i in range(len(AVAILABLE_SURFACES))]
        graph = GraphPhasePatch(N_OPT_NODES, node_per_phase, len(pin_feet_frames), patches)
        
        # MCTS search 
        mcts = MCTSTaskSequence(graph, C=0.5)
        start_node = ()
        mcts.run(start_node, ITERATIONS)
        
    else:
        graph = GaitParallelGraph(len(pin_feet_frames), N_NODES)

        N_STEP = graph.base
        increase_res_it = np.logspace(0, np.log10(2*ITERATIONS), N_STEP + 1, dtype=np.int32)[:4]
        n_base = int(np.log2(N_NODES))
        increase_res_it = [0] + [int(2**(i*2))  for i in range(2, 2 + n_base - 1)]
        print(increase_res_it)
        mcts = MCTSTaskSequence(graph, C=0.5, increase_res_it=increase_res_it)
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