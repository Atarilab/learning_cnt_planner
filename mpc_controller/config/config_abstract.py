import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from typeguard import typechecked

@dataclass
@typechecked
class GaitConfig():
    # Name of the gait
    gait_name : str
    # Nominal gait period (s)
    nominal_period : float
    # Gait stance ratio in [0,1]
    stance_ratio : np.ndarray
    # Gait offset between legs in [0,1]
    phase_offset : np.ndarray
    # Gait nominal height (m)
    nom_height : float
    # Gait step height (m)
    step_height : float

    def __post_init__(self):
        assert  (all(0 <= r <= 1 for r in self.stance_ratio)), "stance_ratio should be in [0,1]"
        assert  (all(0 <= r <= 1 for r in self.phase_offset)), "phase_offset should be in [0,1]"

@dataclass
@typechecked
class MPCOptConfig():
    # Time horizon (s)
    time_horizon : float
    # Number of optimization nodes
    n_nodes : int
    # Bounds time between two nodes
    opt_dt_scale : np.ndarray
    # Replanning frequency
    replanning_freq : int
    # Real time iterations
    real_time : bool
    # Fixed contact timings
    opt_switch_time : bool
    # Constrained eeff locations
    opt_cnt_pos : bool
    # Use peak constrained
    opt_peak : bool
    # Reguralization running cost
    reg_eps: float
    # Reguralization terminal cost
    reg_eps_e: float

    def __post_init__(self):
        assert len(self.opt_dt_scale) == 2, "opt_dt_scale must be of shape 2"

    def get_dt_bounds(self) -> Tuple[float, float]:
        """
        Return optimization bounds for the time between two optimization nodes.
        """
        dt_nodes = self.get_dt_nodes()
        return (
            round(dt_nodes * self.opt_dt_scale[0], 4),
            round(dt_nodes * self.opt_dt_scale[1], 4),
        )
    
    def get_dt_nodes(self) -> float:
        """
        Return nominal time between two optimization nodes.
        """
        return round(self.time_horizon / self.n_nodes, 4)

@dataclass
@typechecked
class MPCCostConfig:
    # Weights for the terminal cost of base position, orientation, and velocity
    # [x, w, z, ox, oy, oz, vx, vy, vz, wx, wy, wz]
    W_e_base: np.ndarray
    # Weights for the running cost of base position, orientation, and velocity
    # [x, w, z, ox, oy, oz, vx, vy, vz, wx, wy, wz]
    W_base: np.ndarray
    # Weights for acceleration cost
    # [x, w, z, ox, oy, oz, vx, vy, vz, wx, wy, wz]
    W_acc: np.ndarray
    # Weights for swing cost (eeff motion)
    # length: number of eeff
    W_swing: np.ndarray
    # Weights for force regularization (per foot) [[x, y, z], ...]
    # length: number of eeff
    W_cnt_f_reg: np.ndarray
    # Weight constraint contact horizontal velocity
    W_feet_z_vel: np.ndarray

    def __post_init__(self):
        assert len(self.W_e_base) == 12, "W_e_base must be of shape 12"
        assert len(self.W_base) == 12, "W_base must be of shape 12"
        assert len(self.W_acc) == 12, "W_acc must be of shape 12"
        assert (len(self.W_swing) == len(self.W_cnt_f_reg) and
                len(self.W_swing) == len(self.W_feet_z_vel)),\
                "W_swing and W_foot should have the same length."
        for i, foot_weight in enumerate(self.W_cnt_f_reg):
            assert len(foot_weight) == 3, f"W_foot[{i}] must be of shape 3"
