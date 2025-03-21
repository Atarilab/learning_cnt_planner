from typing import List
from mj_pin.simulator import Simulator
from scene.primitives import Surface

SCALE_NORMAL = 0.08
RADIUS = 0.008
N_SHPERE = 8

# Viz surfaces
def vis_surfaces_normal(sim : Simulator, surfaces : List[Surface]):
    for s in  surfaces:
        # normal and center
        for i in range(N_SHPERE):
            normal_v = s.center + s.normal * SCALE_NORMAL * (i/N_SHPERE)
            # center
            if i == 0:
                sim.edit.add_sphere(normal_v, RADIUS*2, color="black")
            # normal
            sim.edit.add_sphere(normal_v, RADIUS, color="red")