import numpy as np
from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from typing import Any, List

from scene.primitives import Box, Surface
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.config.quadruped.mpc_gait import QuadrupedSlowTrot, QuadrupedTrot
from mpc_controller.utils.solver import QuadrupedAcadosSolver
from mpc_controller.utils.contact_planner import ContactPlanner
from main import ReferenceVisualCallback

SIM_DT = 1e-3
ROBOT_NAME = "go2"
RECOMPILE = False
HEIGHT = 0.3
EDGE = 0.4
N_NODES = 50*2
DURATION = 1.5*2
AVAILABLE_SURFACES = [0, 1, 4]

def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)

################## Box
offset = 0.4

pos =   [offset + EDGE/2., 0., HEIGHT/2.]
size =  [EDGE/2., EDGE/1.5, HEIGHT/2.]
euler = [0., 0., 0.]
box = Box(pos, size, euler)
plane = Surface(np.array([0., 0., 0.]), np.array([0., 0., 1.]), size_x=1e10, size_y=1e10)

surfaces = [plane] + box.get_surfaces()

################## Simulator
sim = Simulator(robot_description.xml_scene_path)
sim.edit.add_box(pos, size, euler)
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

################## Search contact patch
contact_patches = [
    [0, 1, 1],
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
]

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

##################  Solver
# Opt
config_opt = MPCOptConfig(
    time_horizon=DURATION,
    n_nodes=N_NODES,
    replanning_freq=0, Kp=0, Kd=0,
    recompile=RECOMPILE,
    max_iter=50,
    max_qp_iter=15,
    opt_peak=True,
    warm_start_sol=False,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed
)
dt_nodes = config_opt.get_dt_nodes()

# Gait
config_gait = GaitConfig(
    gait_name="climb",
    nominal_period=DURATION  / 3.,
    stance_ratio=np.array([0.75, 0.75, 0.75, 0.75]),
    phase_offset=np.array([0.75, 0.5, 0.0, 0.0]),
    nom_height=0.29,
    step_height=0.05
)
config_gait = GaitConfig(
    gait_name="climb_pace",
    nominal_period=DURATION  / 3.,
    stance_ratio=np.array([0.75, 0.75, 0.75, 0.75]),
    phase_offset=np.array([0.75, 0.5, 0.25, 0.0]),
    nom_height=0.29,
    step_height=0.05
)
# config_gait = GaitConfig(
#     gait_name="climb_jump",
#     nominal_period=DURATION  / 3.,
#     stance_ratio=np.array([0.75, 0.75, 0.75, 0.75]),
#     phase_offset=np.array([0., 0., 0., 0.]),
#     nom_height=0.29,
#     step_height=0.05
# )
cnt_planner = ContactPlanner(pin_feet_frames, dt_nodes, config_gait)

# Cost
W = [
        0e0, 0e0, 0e0,      # Base position weights
        0e0, 1e0, 1e0,      # Base orientation (ypr) weights
        1e0, 1e0, 2e2,      # Base linear velocity weights
        1e0, 1e1, 1e1,      # Base angular velocity weights
    ]

HSE_SCALE = [15., 5., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W, 3.),
    W_base=__init_np(W, 1.),
    W_joint=__init_np(HSE_SCALE + [.1] * len(HSE_SCALE), 1.),
    W_e_joint=__init_np(HSE_SCALE + [.1] * len(HSE_SCALE), 1.),
    W_acc=__init_np(HSE_SCALE, 1.e-4),
    W_swing=__init_np([2e3] * n_feet),
    W_cnt_f_reg = __init_np([[0.01, 0.01, 0.01]] * n_feet),
    W_foot_pos_constr_stab = __init_np([5e1] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

# Init solver
cnt_sequence = cnt_planner.get_contacts(0, N_NODES + 1)
swing_peak = cnt_planner.get_peaks(0, N_NODES + 1)
patch_center, patch_normal, patch_size = get_contact_patch(cnt_sequence, contact_patches, surfaces)

base_ref = np.zeros(12)
base_ref[2] = config_gait.nom_height
base_ref_e = base_ref.copy()

solver = QuadrupedAcadosSolver(
    robot_description.urdf_path,
    pin_feet_frames,
    config_opt,
    config_cost,
    height_offset = 0.,
    print_info = True,
    compute_timings = True,
    )
solver.set_contact_restriction(True)
q0, v0 = solver.dyn.convert_from_mujoco(q0_mj, v0_mj)
solver.dyn.update_pin(q0, v0)

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
    swing_peak=swing_peak,
)
solver.setup_contact_patch(patch_center, patch_normal, patch_size)
solver.update_solver()

# Solve and visualize
q_sol, v_sol, _, _, dt_sol = solver.solve()
time_traj = np.concatenate(([0.], np.cumsum(dt_sol)))
q_mj_sol = np.stack([solver.dyn.convert_to_mujoco(q, v)[0] for q, v in zip(q_sol, v_sol)])
sim.visualize_trajectory(q_mj_sol, time_traj, record_video=False)