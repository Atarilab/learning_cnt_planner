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

SIM_DT = 1e-3
ROBOT_NAME = "go2"

# MPC Controller
robot_desc = get_robot_description(ROBOT_NAME)
feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

mpc = LocomotionMPC(
    path_urdf=robot_desc.urdf_path,
    feet_frame_names = feet_frame_names,
    robot_name=ROBOT_NAME,
    joint_ref = robot_desc.q0,
    gait_name="trot",
    contact_planner="custom",
    interactive_goal=False,
    sim_dt=SIM_DT,
    print_info=False,
    )
mpc.restrict_cnt = False
mpc.solver.set_contact_restriction(False)

sim = Simulator(robot_desc.xml_scene_path)
# gait_sequence = mpc.contact_planner.gait_sequence
# print(gait_sequence)
# # mpc.contact_planner.set_periodic_sequence(gait_sequence)
# mpc.set_command([1.5, 0,0])
# sim.run(controller=mpc)

class MCTSGaitSequence(MCTSBase):
    def __init__(self, graph, C = 0.01, sim_time : float = 2, increase_res_it = []):
        super().__init__(graph, C)
        self.sim_time = sim_time
        self.v_des = [0.3, 0., 0.]
        self.set_resolution(1)
        self.increase_res_it = increase_res_it
        self.max_sim_step =  0
        
    def set_resolution(self, res : int) -> None:
        self.resolution = res
        self.graph.set_resolution(res)

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
        self.current_search_path = [node]

        while (node in self.value_visit and not self.is_leaf(node)):
            node = self.best_child(node)
            if node == self.current_search_path[-1]:
                break
            self.current_search_path.append(node)

        return node
            
    def evaluate(self, simulation_path : list) -> float:

        cnt_sequence = self.resize(simulation_path[-1], mpc.contact_planner.nodes_per_cycle)
        if np.any(np.all(cnt_sequence == cnt_sequence[:, [0]], axis=1)):
            return 0.
        print(cnt_sequence)
        print(self.current_search_path[-1])
        mpc.reset()
        mpc.contact_planner.set_periodic_sequence(cnt_sequence)
        mpc.set_command(self.v_des)
        sim.run(
            self.sim_time,
            use_viewer=False,
            controller=mpc,
            allowed_collision=["floor"] + [f[:2] for f in feet_frame_names],
            )
        time.sleep(0.05)
        
        # Compute reward
        reward = 0.
        
        if not sim.collided:
            avg_vel = sim.mj_data.qpos[:3] / self.sim_time
            W_VEL = 1.
            reward = np.exp(- W_VEL * np.sum(np.abs(self.v_des - avg_vel)))
        return reward
    
    def backpropagate(self, reward):
        print(self.it, self.resolution, reward)

        super().backpropagate(reward)
        
        if self.it in self.increase_res_it:
            self.set_resolution(self.resolution + 1)
        

N_NODES = 8
ITERATIONS = 300
graph = GaitParallelGraph(len(feet_frame_names), N_NODES)

N_STEP = graph.base - 1
increase_res_it = np.logspace(0, np.log10(ITERATIONS), N_STEP+2, dtype=np.int32)[1:-1]
mcts = MCTSGaitSequence(graph, C=1., increase_res_it=increase_res_it)
mcts.run(graph.start_node, ITERATIONS)

best_sequence = mcts.best_path(graph.start_node)
print(best_sequence)

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