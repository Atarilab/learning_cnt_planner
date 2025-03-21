import numpy as np
from typing import List
from mj_pin.simulator import Simulator
from scene.primitives import Box, Surface
from scene.utils import vis_surfaces_normal

def setup_scene(sim : Simulator, height : float, edge : float, vis_normal : bool = False) -> List[Surface]:

    ################## Box
    offset = 0.4

    pos =   [offset + edge/2., 0., height/2.]
    size =  [edge/2., edge/1.5, height/2.]
    euler = [0., 0., 0.]
    box = Box(pos, size, euler)
    plane = Surface(np.array([0., 0., 0.]), np.array([0., 0., 1.]), rot=np.eye(3), size_x=1e10, size_y=1e10)

    all_surfaces = [plane] + box.get_surfaces()
    # floor, top, side
    AVAILABLE_SURFACES = [0, 1, 4]
    surfaces = [all_surfaces[i] for i in AVAILABLE_SURFACES]
    
    if vis_normal:
        vis_surfaces_normal(sim, all_surfaces)

    ################## Simulator
    sim.edit.add_box(pos, size, euler, allow_collision=False)

    return surfaces