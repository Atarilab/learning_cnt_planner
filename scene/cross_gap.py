import numpy as np
from typing import List
from mj_pin.simulator import Simulator
from scene.primitives import Box, Surface
from scene.utils import vis_surfaces_normal

def setup_scene(sim : Simulator,
                gap_length : float,
                wall_angle : float,
                height : float = 0.,
                vis_normal : bool = False) -> List[Surface]:

    ################## Start Box
    # Parameters
    thick = 0.01
    height += 0.001
    large = 0.22

    pos_start = [0.0, 0.0, height]
    size_start = [large, large, height]
    euler_start = [0.0, 0.0, 0.0]
    start = Box(pos_start, size_start, euler_start)

    ################## Walls
    wall_offset = large + gap_length / 2.0
    wall_size = [gap_length / 2., large, thick / 2.0]
    wall_euler = [wall_angle, 0.0, 0.0]
    wall_gap = large * (1 + np.sin(wall_angle) * 2) # Adjustable gap between walls

    wall_height = 2 * large * np.cos(wall_angle)
    wall_1_pos = [wall_offset, wall_gap / 2., 2 * height + wall_height]
    wall_2_pos = [wall_offset, -wall_gap / 2., 2 * height + wall_height]

    wall_1 = Box(wall_1_pos, wall_size, wall_euler)
    wall_2 = Box(wall_2_pos, wall_size, [-angle for angle in wall_euler])

    ################## End Box
    pos_end = np.array(pos_start) + np.array([gap_length + 2 * large, 0.0, 0.0])
    size_end = [large, large, height]
    euler_end = [0.0, 0.0, 0.0]
    end = Box(pos_end, size_end, euler_end)

    ################## Surfaces
    all_surfaces = start.get_surfaces() + end.get_surfaces() + wall_1.get_surfaces() + wall_2.get_surfaces()
    # top start, top end, side L, side R
    AVAILABLE_SURFACES = [0, 6, 12, 18]
    surfaces = [all_surfaces[i] for i in AVAILABLE_SURFACES]
    
    if vis_normal:
        vis_surfaces_normal(sim, surfaces)

    ################## Simulator
    sim.edit.add_box(pos_start, size_start, euler_start)
    sim.edit.add_box(wall_1_pos, wall_size, wall_euler)
    sim.edit.add_box(wall_2_pos, wall_size, [-angle for angle in wall_euler])
    sim.edit.add_box(pos_end, size_end, euler_end)

    return surfaces