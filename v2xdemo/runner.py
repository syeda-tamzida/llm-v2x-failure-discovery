"""Paired counterfactual execution and causal admission (LAST-V2X Eq. 7).

Each scenario runs twice with the same seed:
  * message-driven ego: decides from received (possibly faulty) BSMs
  * ground-truth twin:  same everything, but decides from the true crossing-car state
ADMITTED (communication-induced failure) only if the message-driven run violates
and the twin stays safe.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional

from .ima import IMAEgo, IMAParams
from .messages import BSM, Channel, FaultEngine, Receiver
from .scenario import Scenario

DT = 0.05
BSM_PERIOD_TICKS = 2          # 10 Hz
COLLISION_DIST = 1.8          # center-to-center (m), LAST-V2X Appendix D
MOVING = 0.5                  # a contact only counts if a vehicle moves faster than this (m/s)


def run_once(scn: Scenario, world, use_truth: bool, ima: IMAParams = IMAParams(),
             range_m: float = 45.0, loss_p: float = 0.0, keep_trace: bool = False) -> Dict:
    rng = random.Random(scn.seed)
    world.reset(scn, ghost=use_truth)
    ego = IMAEgo(scn.ego_speed, world.ego_cp_s, world.cross_cp_s, ima, DT)
    channel = Channel(range_m, loss_p, rng=random.Random(scn.seed + 1))
    faults = FaultEngine(scn.fault, rng=random.Random(scn.seed + 2))
    rx = Receiver()

    cs, cv = 0.0, scn.cross.v0
    min_gap = math.inf
    violation = False
    wrong_decisions = 0           # decisions taken while belief and truth disagreed
    first_rx_time: Optional[float] = None
    t_in_range: Optional[float] = None
    first_stop_t: Optional[float] = None
    trace = []
    steps = int(scn.horizon / DT)
    for k in range(steps):
        t = k * DT
        # --- crossing car: open-loop behavior profile ---
        cv = min(max(cv + scn.cross.accel_at(t) * DT, 0.0), 25.0)
        cs += cv * DT
        # --- V2X: sample, channel, fault, receive ---
        ego_xy, cross_xy = world.step(ego.s, cs)
        gap = math.hypot(ego_xy[0] - cross_xy[0], ego_xy[1] - cross_xy[1])
        if k % BSM_PERIOD_TICKS == 0:
            msg = BSM(t_gen=t, t_stamp=t, s=cs, v=cv)
            msg = msg if channel.deliver(gap) else None
            delivered = faults.process(msg, t)
        else:
            delivered = faults.process(None, t)
        rx.accept(delivered)
        if rx.latest is not None and first_rx_time is None:
            first_rx_time = t
        # --- ego decision ---
        if use_truth:
            belief = BSM(t, t, cs, cv) if gap <= range_m else None
        else:
            belief = rx.latest
        acting = ego.step(t, belief)
        if t_in_range is None and gap <= range_m:
            t_in_range = t
        if first_stop_t is None and ego.last_decision == "stop":
            first_stop_t = t
        if not use_truth and belief is not None:
            age = t - belief.t_stamp
            est = belief.s + belief.v * max(age, 0)
            if abs(est - cs) > 1.5 + cv * 0.5:      # position budget + age*speed (Appendix D)
                wrong_decisions += 1
        # --- safety monitor ---
        min_gap = min(min_gap, gap)
        if gap < COLLISION_DIST and (ego.v > MOVING or cv > MOVING):
            violation = True
        if keep_trace:
            trace.append(dict(t=round(t, 2), ego_s=round(ego.s, 2), ego_v=round(ego.v, 2), cross_s=round(cs, 2),
                              cross_v=round(cv, 2), gap=round(gap, 2), decision=acting,
                              belief_s=None if belief is None else round(belief.s, 2)))
        if violation:
            break
        if ego.s > world.ego_cp_s + 30 and cs > world.cross_cp_s + 30:
            break
    return dict(violation=violation, min_gap=round(min_gap, 3), wrong_belief_ticks=wrong_decisions,
                stale_fallback=ego.decided_on_stale, first_rx_time=first_rx_time,
                t_in_range=t_in_range, first_stop_t=first_stop_t,
                trace=trace if keep_trace else None)


def run_pair(scn: Scenario, world, **kw) -> Dict:
    msg = run_once(scn, world, use_truth=False, **kw)
    gt = run_once(scn, world, use_truth=True, **kw)
    if msg["violation"] and not gt["violation"]:
        label = "ADMITTED"
    elif msg["violation"] and gt["violation"]:
        label = "NON_ATTRIBUTABLE"
    elif gt["violation"]:
        label = "TWIN_ONLY"           # fault happened to make the ego safer
    else:
        label = "CLEAN"
    return dict(label=label, admitted=label == "ADMITTED",
                v_msg=msg["violation"], v_gt=gt["violation"],
                gap_msg=msg["min_gap"], gap_gt=gt["min_gap"],
                delta_g=round(gt["min_gap"] - msg["min_gap"], 3),
                wrong_belief_ticks=msg["wrong_belief_ticks"], stale_fallback=msg["stale_fallback"],
                t_in_range=msg["t_in_range"], first_rx_time=msg["first_rx_time"],
                stop_t_msg=msg["first_stop_t"], stop_t_gt=gt["first_stop_t"],
                trace_msg=msg["trace"], trace_gt=gt["trace"])
