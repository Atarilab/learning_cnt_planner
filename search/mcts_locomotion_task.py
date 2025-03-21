import numpy as np
import mujoco
from collections import defaultdict
from typing import List, Tuple

from mpc_controller.utils.solver import QuadrupedAcadosSolver
from mj_pin.simulator import Simulator
from search.utils.mcts import MCTSBase
from search.graph_phase_patch import GraphPhasePatchWithPos
from scene.primitives import Surface
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {(end_time - start_time) * 1e3:.4f} ms")
        return result
    return timed

class MCTSPhaseLocomotionTask(MCTSBase):
    def __init__(self,
                 C : float,
                 alpha_exploration : float,
                 sim : Simulator,
                 solver : QuadrupedAcadosSolver,
                 n_phases : int,
                 surfaces : List[Surface],
                 goal_surf_id : List[int],
                 ):
        self.sim = sim
        self.solver = solver
        self.surfaces = surfaces
        
        # Solver variables
        q0_mj, v0_mj = sim.get_initial_state()
        self.q0, self.v0 = solver.dyn.convert_from_mujoco(q0_mj, v0_mj)
        self.base_ref = np.zeros(12)
        self.base_ref[2] = q0_mj[2]
        self.base_ref_e = self.base_ref.copy()
        self.step_height = 0.05
        self.opt_nodes = self.solver.config_opt.n_nodes
        
        # Init graph
        self.node_per_phase = int(self.opt_nodes / n_phases)
        self.n_phases = n_phases
        self.n_cnt = len(self.solver.feet_frame_names)
        self.patch_pos = np.array([surf.center for surf in surfaces])
        self.goal_pos = np.array([self.patch_pos[i] for i in goal_surf_id])

        graph = GraphPhasePatchWithPos(
            self.opt_nodes,
            self.node_per_phase,
            self.n_cnt,
            goal_surf_id,
            self.patch_pos,
        )

        # For heuristics
        dist_all_patches = np.linalg.norm(self.patch_pos[None, :, :] - self.patch_pos[:, None, :], axis=-1)
        self.max_dist_patches = np.max(dist_all_patches)
        self.mean_pos_patches = np.mean(self.patch_pos.reshape(-1, 3), axis=0, keepdims=True)
        
        # To compute reward
        self.max_reward = 0.
        self.mj_feet_frames = ["FL", "FR", "RL", "RR"]
        self.allowed_collision = [
                mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, obj) if isinstance(obj, str)
                else int(obj)
                for obj
                in ["floor"] + sim.edit.name_allowed_collisions + self.mj_feet_frames
            ]
        
        self.alpha_exploration = alpha_exploration
        super().__init__(graph, C)

    def heuristic_bias(self, node):
        return self.distance_to_goal(node)
    
    def count_kin_collision(self,
                            q_mj_traj,
                            allowed_collision = [],
                            safe_collision = []
                            ):
        collision_count = defaultdict(int)
        
        for q_mj in q_mj_traj:
            self.sim.mj_data.qpos[:] = q_mj
            mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
            
            for geom1, geom2 in zip(self.sim.mj_data.contact.geom1, self.sim.mj_data.contact.geom2):
                if geom1 in safe_collision or geom2 in safe_collision:
                    continue
                
                # If collision
                if not(geom1 in allowed_collision and geom2 in allowed_collision):
                    name1 = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or "robot"
                    name2 = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or "robot"
                    collision_count[name1] += 1
                    collision_count[name2] += 1
                
        return collision_count
    
    def distance_to_goal(self, node):
        _, cnt, patch = node
        n_in_cnt = sum(cnt)
        
        if n_in_cnt == 0:
            patch_pos = self.mean_pos_patches.reshape(-1, 3)
            goal_pos = self.goal_pos

        else:
            patch_pos = np.take_along_axis(self.patch_pos, np.array(patch).reshape(-1, 1), axis=0).reshape(-1, 3)
            goal_pos = self.goal_pos[np.array(cnt) == 1].reshape(-1, 3)
        
        avg_dist_to_goal = np.linalg.norm(np.mean(patch_pos - goal_pos, axis=0))
        return 1 - avg_dist_to_goal / self.max_dist_patches
    
    def rollout_policy(self, node):
        """
        Selects a random child node during rollout.
        """
        children = self.graph.get_neighbors(node)
        
        if np.random.rand() < self.alpha_exploration:
            child_id = np.random.choice(len(children))
        else:
            biases = np.array([self.distance_to_goal(child) for child in children]) + 1e-6
            s = biases.sum()
            n = len(biases)
            # Add mean to increase lower probabilities
            biases += s / n
            probabilities = biases / (2 * s)
            child_id = np.random.choice(len(children), p=probabilities)
            
        return children[child_id]
    
    def get_sequence_patches_from_path(self, path):
        seq = np.array([list(phase[1]) for phase in path]).T.repeat(self.node_per_phase, axis=-1)
        seq = np.concatenate([seq, seq[:, None, -1]], axis=-1)
        
        n_eeff = len(path[-1][1])
        patches_dict = {i_eeff : [] for i_eeff in range(n_eeff)}
        last_cnt = {i_eeff : 0 for i_eeff in range(n_eeff)}
        for phase in path:
            _, cnt, patch = phase
            i_patch = 0
            for i_eeff, c in enumerate(cnt):
                if c:
                    if last_cnt[i_eeff] == 0:
                        patches_dict[i_eeff].append(patch[i_patch])
                last_cnt[i_eeff] = c
                i_patch += c
        patches = list(patches_dict.values())
        return seq, patches
    

    def get_contact_patch(
        self,
        contact_seq: np.ndarray,
        contact_patches: List[List[int]],
        surfaces: List[Surface],
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
        surface_rot = np.zeros((n_feet, n_nodes, 3, 3))
        surface_sizes = np.zeros((n_feet, n_nodes, 2))
        
        for i_foot in range(n_feet):
            cnt_phase = 0
            last_in_cnt = False
            for i_node in range(n_nodes):
                if contact_seq[i_foot, i_node] == 1:  # Foot is in contact
                    last_in_cnt = True
                else:
                    # If break cnt, go to next phase
                    if last_in_cnt:
                        cnt_phase += 1
                    last_in_cnt = False
                    
                # Select a surface from the valid patches
                i = min(cnt_phase, len(contact_patches[i_foot])-1)
                id_surf = contact_patches[i_foot][i]
                surface = surfaces[id_surf]

                # Assign surface properties
                surface_centers[i_foot, i_node] = surface.center
                surface_normals[i_foot, i_node] = surface.normal
                surface_rot[i_foot, i_node] = surface.rot
                surface_sizes[i_foot, i_node] = [surface.size_x, surface.size_y]             

        return surface_centers, surface_normals, surface_rot, surface_sizes
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def run_solver(self, simulation_path : list, reset : bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        start_phase = 0 if simulation_path[0] else 1
        
        cnt_sequence, patches = self.get_sequence_patches_from_path(simulation_path[start_phase:])
        cnt_sequence = cnt_sequence[:, :self.opt_nodes+1]

        patch_center, patch_normal, patch_rot, patch_size = self.get_contact_patch(cnt_sequence, patches, self.surfaces)
        peak_sequence = np.ones_like(cnt_sequence) - cnt_sequence
        
        if reset:
            self.solver.reset()
            
        self.solver.set_contact_restriction(True)
        self.solver.dyn.update_pin(self.q0, self.v0)

        self.solver.init(
            i_node=0,
            q=self.q0,
            v=self.v0,
            base_ref=self.base_ref,
            base_ref_e=self.base_ref_e,
            joint_ref=self.q0[-12:],
            step_height=self.step_height,
            cnt_sequence=cnt_sequence,
            cnt_locations=patch_center,
            swing_peak=peak_sequence,
        )
        try:
            self.solver.setup_contact_patch(patch_center, patch_normal, patch_rot, patch_size)
            self.solver.update_solver()
            q_sol, v_sol, _, _, dt_sol = self.solver.solve()
            return q_sol, v_sol, dt_sol
        
        except Exception as e:
            print(e)
            return [], [], []

    def evaluate(self, simulation_path : list) -> float:

        q_sol, v_sol, dt_sol = self.run_solver(simulation_path)
        if len(q_sol) == 0:
            return 0
        
        # Compute reward
        reward = 1.

        # Reward on the residuals
        log10_prod_res = np.log10(np.prod(self.solver.solver.get_stats("residuals")))
        W_RES_POS = 1/3
        W_RES_NEG = 1/6
        reward *= self.sigmoid(-(
            log10_prod_res * W_RES_POS * max(log10_prod_res, 0) +
            log10_prod_res * W_RES_NEG * min(log10_prod_res, 0)
            ))
        
        # Reward on the number of collisions
        v = np.zeros_like(v_sol[0])
        q_mj_traj = np.stack([self.solver.dyn.convert_to_mujoco(q, v)[0] for q in q_sol])
        all_collisions = self.count_kin_collision(
                                         q_mj_traj,
                                         self.allowed_collision,
                                         )
        n_robot_collision = sum([all_collisions[k] for k in ["robot"] + self.mj_feet_frames])
        avg_robot_collision = n_robot_collision / len(q_sol)
        W_COLLISION = 0.2
        reward *= np.exp(-W_COLLISION * avg_robot_collision)
        
        if reward > self.max_reward:
            print(simulation_path, reward, avg_robot_collision, log10_prod_res)
            time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
            self.sim.visualize_trajectory(q_mj_traj, time_traj, record_video=False)
            self.max_reward = reward
            self.max_reward_it = self.it
        
        return reward