"""Scenario definition: geometry, crossing-car behavior, and one V2X fault.

Everything is plain data so the same scenario can run in the kinematic world
(no CARLA) or in CARLA, and be saved to / loaded from JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BehaviorEvent:
    """Constant acceleration `accel` (m/s^2) applied from time `t` for `dur` seconds."""
    t: float
    accel: float
    dur: float


@dataclass
class Behavior:
    """Open-loop speed profile of the crossing car (it does not react to the ego)."""
    v0: float
    events: List[BehaviorEvent] = field(default_factory=list)

    def accel_at(self, t: float) -> float:
        return sum(e.accel for e in self.events if e.t <= t < e.t + e.dur)

    def label(self) -> str:
        """Coarse behavior class, used to count distinct failure mechanisms."""
        if not self.events:
            return "constant"
        signs = [("+" if e.accel > 0 else "-") for e in sorted(self.events, key=lambda e: e.t) if e.accel != 0]
        if not signs:
            return "constant"
        if all(s == "+" for s in signs):
            return "accelerate"
        if all(s == "-" for s in signs):
            return "decelerate"
        return "slow-then-go" if signs[0] == "-" else "go-then-slow"


FAULT_TYPES = ["none", "drop", "freeze", "delay", "pos_perturb"]


@dataclass
class Fault:
    """One message-level fault on the crossing car's BSM stream.

    type:      none | drop | freeze | delay | pos_perturb
    start:     window start, seconds from simulation start
    duration:  window length in seconds (None = until the end)
    bias:      pos_perturb: along-track position bias in m (negative = reported behind true position)
    sigma:     pos_perturb: Gaussian position noise in m
    delay:     delay: extra delivery delay in s
    """
    type: str = "none"
    start: float = 0.0
    duration: Optional[float] = None
    bias: float = 0.0
    sigma: float = 0.0
    delay: float = 0.0

    def active(self, t: float) -> bool:
        if self.type == "none":
            return False
        end = float("inf") if self.duration is None else self.start + self.duration
        return self.start <= t < end


@dataclass
class Scenario:
    name: str
    base: str                      # which base scene it was derived from
    ego_dist: float                # ego distance to conflict point at t=0 (m)
    ego_speed: float               # ego desired speed (m/s)
    cross_dist: float              # crossing car distance to conflict point at t=0 (m)
    cross: Behavior
    fault: Fault
    seed: int = 0
    horizon: float = 20.0
    source: str = "manual"         # fixed | random | llm | manual
    hypothesis: str = ""           # LLM causal story (if any)
    predicts_failure: Optional[bool] = None

    def mechanism(self) -> str:
        return f"{self.fault.type}+{self.cross.label()}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        d = dict(d)
        cross = d.pop("cross")
        fault = d.pop("fault")
        beh = Behavior(v0=cross["v0"], events=[BehaviorEvent(**e) for e in cross.get("events", [])])
        return Scenario(cross=beh, fault=Fault(**fault), **d)


# Base scenes: initial conditions the methods start from.
# S1 is analytically conflicting; S2 and S3 are NOT (arrival gap > 3.5 s),
# so a constant-speed test would rarely make them dangerous.
BASE_SCENES = {
    "S1": dict(ego_dist=100.0, ego_speed=10.0, cross_dist=95.0, cross_v0=10.0,
               note="Both cars arrive at the conflict point at about the same time (~0.5 s apart)."),
    "S2": dict(ego_dist=100.0, ego_speed=10.0, cross_dist=150.0, cross_v0=10.0,
               note="Crossing car arrives ~5 s AFTER the ego at constant speed (analytically non-conflicting)."),
    "S3": dict(ego_dist=100.0, ego_speed=10.0, cross_dist=55.0, cross_v0=10.0,
               note="Crossing car arrives ~4.5 s BEFORE the ego at constant speed (analytically non-conflicting)."),
}


def base_scenario(base: str, seed: int = 0) -> Scenario:
    b = BASE_SCENES[base]
    return Scenario(name=f"{base}_base", base=base, ego_dist=b["ego_dist"], ego_speed=b["ego_speed"],
                    cross_dist=b["cross_dist"], cross=Behavior(v0=b["cross_v0"]), fault=Fault(), seed=seed)
