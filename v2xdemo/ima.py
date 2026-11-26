"""Intersection Movement Assist (IMA) on the ego, after LAST-V2X Appendix D.

Rule: stop if a crossing track is predicted to reach the conflict point within
`window` (3.5 s) of the ego's own arrival, considering arrivals within `horizon` (12 s).
The ego only sees its BELIEF (received BSMs), never the true crossing car state,
except in the ground-truth twin run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .messages import BSM


@dataclass
class IMAParams:
    window: float = 3.5        # conflict window (s)
    horizon: float = 12.0      # only consider ego arrivals within this (s)
    age_budget: float = 0.5    # BSM freshness budget (s), SAE J2945/1 value used in the paper
    reaction: float = 1.0      # decision-to-action delay (s)
    a_comf: float = 3.5        # comfortable braking (m/s^2)
    a_max: float = 7.0         # emergency braking cap (m/s^2)
    a_acc: float = 2.0         # re-acceleration (m/s^2)
    stop_margin: float = 6.0   # stop this far before the conflict point (m)


class IMAEgo:
    def __init__(self, desired_speed: float, cp_s: float, cross_cp_s: float, p: IMAParams = IMAParams(), dt: float = 0.05):
        self.vd = desired_speed
        self.cp = cp_s               # ego conflict point along its path
        self.cross_cp = cross_cp_s   # crossing car conflict point along ITS path (known from MAP)
        self.p = p
        self.dt = dt
        self.s = 0.0
        self.v = desired_speed
        self.committed = False
        n = max(1, int(round(p.reaction / dt)))
        self._pending = deque(["go"] * n, maxlen=n)   # decisions waiting for reaction delay
        self.last_decision = "go"
        self.decided_on_stale = False

    def decide(self, t: float, belief: Optional[BSM]) -> str:
        p = self.p
        d_ego = self.cp - self.s
        if d_ego < -2.0 or self.committed:
            return "go"
        # cannot stop even with emergency braking -> committed
        if d_ego > 0 and d_ego < self.v ** 2 / (2 * p.a_max):
            self.committed = True
            return "go"
        if belief is None:
            return "go"                       # silent: ego knows nothing about the crossing car
        age = t - belief.t_stamp
        if age > p.age_budget:
            self.decided_on_stale = True
            return "stop"                     # stale-but-detectable -> conservative fallback
        cross_s = belief.s + belief.v * max(age, 0.0)   # extrapolate to now
        d_cross = self.cross_cp - cross_s
        if d_cross < -3.0:
            return "go"                       # believed already through the junction
        t_cross = d_cross / max(belief.v, 0.1)
        t_ego = max(d_ego, 0.0) / max(self.v, 3.0)
        if t_ego < p.horizon and abs(t_ego - t_cross) < p.window:
            return "stop"
        return "go"

    def step(self, t: float, belief: Optional[BSM]) -> str:
        decision = self.decide(t, belief)
        self.last_decision = decision
        acting = self._pending[0]
        self._pending.append(decision)
        p = self.p
        if acting == "stop" and not self.committed:
            dist = self.cp - p.stop_margin - self.s
            if dist > 0.5:
                a = -min(max(p.a_comf, self.v ** 2 / (2 * dist)), p.a_max)
            else:
                a = -p.a_max
        else:
            a = p.a_acc if self.v < self.vd else 0.0
        self.v = min(max(self.v + a * self.dt, 0.0), self.vd)
        self.s += self.v * self.dt
        return acting
