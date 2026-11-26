"""Pure-Python world: two straight perpendicular lanes crossing at (0, 0).

Use it to develop and debug without CARLA. Same interface as CarlaWorld.
"""
from __future__ import annotations

from typing import Tuple

from .scenario import Scenario


class KinematicWorld:
    name = "kinematic"

    def reset(self, scn: Scenario, ghost: bool = False):
        self.scn = scn
        # ego drives +x, crossing car drives +y; s = 0 at their start points
        self.ego_cp_s = scn.ego_dist
        self.cross_cp_s = scn.cross_dist

    def step(self, ego_s: float, cross_s: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ego_xy = (ego_s - self.scn.ego_dist, 0.0)
        cross_xy = (0.0, cross_s - self.scn.cross_dist)
        return ego_xy, cross_xy

    def close(self):
        pass
