"""V2X information chain:  world -> BSM sampling -> channel -> fault injection -> receiver belief.

Mirrors the chain in LAST-V2X (Eq. 3), simplified to one stream (the crossing
car's BSMs, 10 Hz) and a range-limited channel.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional

from .scenario import Fault


@dataclass
class BSM:
    t_gen: float      # true generation time
    t_stamp: float    # timestamp written in the message (can be wrong)
    s: float          # sender's progress along its own lane (m from its start)
    v: float          # sender speed (m/s)


class Channel:
    """Delivers a message if sender and receiver are within `range_m`; optional random loss."""

    def __init__(self, range_m: float = 45.0, loss_p: float = 0.0, latency: float = 0.01, rng: Optional[random.Random] = None):
        self.range_m = range_m
        self.loss_p = loss_p
        self.latency = latency
        self.rng = rng or random.Random(0)

    def deliver(self, distance_m: float) -> bool:
        if distance_m > self.range_m:
            return False
        return self.rng.random() >= self.loss_p


class FaultEngine:
    """Applies one fault to messages that survived the channel."""

    def __init__(self, fault: Fault, rng: Optional[random.Random] = None):
        self.f = fault
        self.rng = rng or random.Random(0)
        self._frozen: Optional[BSM] = None
        self._queue: List[tuple] = []   # (deliver_time, msg)

    def process(self, msg: Optional[BSM], t: float) -> List[BSM]:
        """Feed this tick's channel output (or None); return messages delivered now."""
        f = self.f
        out: List[BSM] = []
        if msg is not None:
            if f.active(t):
                if f.type == "drop":
                    msg = None
                elif f.type == "freeze":
                    # hold the payload seen when the window opened, but keep fresh timestamps:
                    # a fresh-but-wrong message
                    if self._frozen is None:
                        self._frozen = msg
                    msg = replace(self._frozen, t_gen=msg.t_gen, t_stamp=msg.t_stamp)
                elif f.type == "pos_perturb":
                    noise = self.rng.gauss(0.0, f.sigma) if f.sigma > 0 else 0.0
                    msg = replace(msg, s=msg.s + f.bias + noise)
                elif f.type == "delay":
                    self._queue.append((t + f.delay, msg))
                    msg = None
            if msg is not None:
                out.append(msg)
        # release delayed messages that are due
        due = [m for (td, m) in self._queue if td <= t + 1e-9]
        self._queue = [(td, m) for (td, m) in self._queue if td > t + 1e-9]
        out.extend(due)
        return out


class Receiver:
    """Keeps the most recent accepted BSM (by timestamp) = the ego's belief about the crossing car."""

    def __init__(self):
        self.latest: Optional[BSM] = None

    def accept(self, msgs: List[BSM]):
        for m in msgs:
            if self.latest is None or m.t_stamp >= self.latest.t_stamp:
                self.latest = m
