"""Physical-validity filter: semantic reasoning proposes, physics verifies.

Every candidate scenario (from any method) must pass these hard limits
before it is allowed to use simulation budget.
"""
from __future__ import annotations

from typing import List, Tuple

from .scenario import Scenario, FAULT_TYPES

LIMITS = dict(
    accel_min=-6.0,      # hardest braking allowed for the crossing car (m/s^2)
    accel_max=3.0,       # strongest acceleration (m/s^2)
    v_min=0.0,
    v_max=20.0,          # 72 km/h at an urban junction
    v0_min=2.0,
    dist_min=20.0,
    dist_max=200.0,
    event_dur_max=10.0,
    bias_max=40.0,
    sigma_max=10.0,
    delay_max=5.0,
    max_events=3,
)


def validate(s: Scenario, dt: float = 0.05) -> Tuple[bool, List[str]]:
    L = LIMITS
    why: List[str] = []
    if not (L["dist_min"] <= s.ego_dist <= L["dist_max"]):
        why.append(f"ego_dist {s.ego_dist} out of range")
    if not (L["dist_min"] <= s.cross_dist <= L["dist_max"]):
        why.append(f"cross_dist {s.cross_dist} out of range")
    if not (L["v0_min"] <= s.ego_speed <= L["v_max"]):
        why.append(f"ego_speed {s.ego_speed} out of range")
    if not (L["v0_min"] <= s.cross.v0 <= L["v_max"]):
        why.append(f"cross v0 {s.cross.v0} out of range")
    if len(s.cross.events) > L["max_events"]:
        why.append("too many behavior events")
    for e in s.cross.events:
        if not (L["accel_min"] <= e.accel <= L["accel_max"]):
            why.append(f"accel {e.accel} outside [{L['accel_min']}, {L['accel_max']}]")
        if not (0 <= e.t <= s.horizon) or not (0 < e.dur <= L["event_dur_max"]):
            why.append(f"event timing t={e.t}, dur={e.dur} invalid")
    # simulate the speed profile: summed overlapping events must keep speed physical
    v, t = s.cross.v0, 0.0
    while t < s.horizon:
        a = s.cross.accel_at(t)
        if not (L["accel_min"] - 1e-9 <= a <= L["accel_max"] + 1e-9):
            why.append(f"summed accel {a:.1f} at t={t:.1f}s infeasible")
            break
        v += a * dt
        # tolerance of a few time steps: a profile planned to reach exactly v_max is valid
        if v > L["v_max"] + 3 * L["accel_max"] * dt:
            why.append(f"speed reaches {v:.1f} m/s > {L['v_max']}")
            break
        v = max(v, 0.0)
        t += dt
    f = s.fault
    if f.type not in FAULT_TYPES:
        why.append(f"unknown fault {f.type}")
    if f.type != "none":
        if not (0 <= f.start <= s.horizon):
            why.append("fault start outside horizon")
        if f.duration is not None and not (0 < f.duration <= s.horizon):
            why.append("fault duration invalid")
        if abs(f.bias) > L["bias_max"] or not (0 <= f.sigma <= L["sigma_max"]):
            why.append("pos_perturb magnitude out of range")
        if not (0 <= f.delay <= L["delay_max"]):
            why.append("delay out of range")
        if f.type == "delay" and f.delay <= 0:
            why.append("delay fault with zero delay")
        if f.type == "pos_perturb" and f.bias == 0 and f.sigma == 0:
            why.append("pos_perturb with zero magnitude")
    return (len(why) == 0, why)
