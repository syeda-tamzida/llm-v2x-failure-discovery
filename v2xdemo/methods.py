"""Three ways to propose test scenarios, compared at the same simulation budget.

  fixed  : LAST-V2X-style. Varies the fault and the initial conditions, but the
           crossing car always drives at constant speed (fixed behavior).
  random : also samples random acceleration / braking events (behavior uncertainty
           without reasoning). This is the key control: if the LLM cannot beat it,
           the LLM is just "generating more scenarios".
  llm    : the LLM proposes causal hypotheses (behavior + fault + story), which are
           physics-filtered and simulated; results are fed back for the next round.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, Iterator, List

from .scenario import BASE_SCENES, Behavior, BehaviorEvent, Fault, Scenario, base_scenario

FAULTS = ["drop", "freeze", "delay", "pos_perturb"]


def _random_fault(rng: random.Random) -> Fault:
    ft = rng.choice(FAULTS)
    f = Fault(type=ft, start=round(rng.uniform(0, 10), 1), duration=rng.choice([2.0, 5.0, 10.0, None]))
    if ft == "pos_perturb":
        f.bias = round(rng.uniform(-30, 30), 1)
        f.sigma = round(rng.choice([0.0, 0.5, 2.0]), 1)
    if ft == "delay":
        f.delay = round(rng.uniform(0.2, 3.0), 2)
    return f


def _jitter(base: Scenario, rng: random.Random, seed: int) -> Scenario:
    s = Scenario.from_dict(base.to_dict())
    s.ego_dist = round(base.ego_dist + rng.uniform(-20, 20), 1)
    s.cross_dist = round(base.cross_dist + rng.uniform(-20, 20), 1)
    s.cross.v0 = round(base.cross.v0 + rng.uniform(-2, 2), 2)
    s.seed = seed
    return s


def fixed_method(base: str, budget: int, seed: int = 0, **_) -> Iterator[Scenario]:
    rng = random.Random(seed)
    b = base_scenario(base)
    for i in range(budget):
        s = _jitter(b, rng, seed * 1000 + i)
        s.fault = _random_fault(rng)
        s.name, s.source = f"{base}_fixed_{i:03d}", "fixed"
        yield s


def random_method(base: str, budget: int, seed: int = 0, **_) -> Iterator[Scenario]:
    rng = random.Random(seed + 7)
    b = base_scenario(base)
    for i in range(budget):
        s = _jitter(b, rng, seed * 1000 + i)
        s.fault = _random_fault(rng)
        for _ in range(rng.randint(0, 2)):
            s.cross.events.append(BehaviorEvent(t=round(rng.uniform(0, 10), 1),
                                                accel=round(rng.uniform(-4, 3), 2),
                                                dur=round(rng.uniform(1, 4), 1)))
        s.name, s.source = f"{base}_random_{i:03d}", "random"
        yield s


METHODS: Dict[str, Callable] = {"fixed": fixed_method, "random": random_method}
