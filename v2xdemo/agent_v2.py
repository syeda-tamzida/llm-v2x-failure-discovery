"""Version 2: a GROUNDED causal-hypothesis agent ("semantic reasoning proposes, physics verifies").

What changed from v1 (llm_agent.py), based on the first CARLA run:
  1. Grounded timeline: the LLM is told WHEN the cars come within radio range, when each would
     reach the junction, and when the twin starts braking. In v1, 85/90 GPT-4o faults started at
     t=0 and 34/90 ended before the cars were even in radio range, so they could not matter.
  2. Accurate domain notes. v1 told the LLM that drop "usually triggers the stale fail-safe";
     that was misleading. v2 describes the receiver rules exactly, without saying how to break them.
  3. Rich feedback: per run, the LLM sees radio-contact time, first-message time, when each ego
     began braking, gaps, and an automatic diagnosis (e.g. "fault ended before radio contact").
  4. Self-check: every hypothesis is first simulated in the fast kinematic world (the screen);
     results go back to the LLM before the next round. Only the best ones are committed to the
     real world (CARLA). A random baseline gets exactly the same screen + commit budget.
"""
from __future__ import annotations

import json
import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .llm_agent import _extract_json, to_scenario
from .physics import LIMITS, validate
from .runner import run_pair
from .scenario import BASE_SCENES, Scenario, base_scenario
from .world_kinematic import KinematicWorld

SYSTEM_PROMPT_V2 = """You are a transportation safety researcher designing tests for a connected-vehicle
Intersection Movement Assist (IMA) application. Propose CAUSAL HYPOTHESES: plausible ways the traffic
situation could develop (driver behavior + V2X information error) so that the ego vehicle makes an
unsafe decision that an identical ego with perfectly correct information would NOT make.

Scene: unsignalized junction. The ego drives straight toward the conflict point; a crossing car
approaches on a perpendicular lane. The crossing car broadcasts BSMs (position along its lane + speed)
every 0.1 s. A BSM reaches the ego only while the two cars are within 45 m of each other.

Exact receiver and decision rules (this is the code that runs):
- The ego keeps the most recent received BSM and extrapolates it: position_now = s + v * age.
- STOP if the crossing car's predicted arrival at the conflict point is within 3.5 s of the ego's own
  arrival (only ego arrivals within 12 s are considered). Otherwise GO.
- If it has received at least one BSM and the latest one is older than 0.5 s: STOP (stale fail-safe).
- If it has never received any BSM: it has no track of the crossing car and assumes no conflict (GO).
- Reaction delay 1.0 s; comfortable braking 3.5 m/s^2 (up to 7 m/s^2 if needed); once it cannot stop
  it is committed. From 10 m/s it needs roughly 15-25 m to stop.
- The crossing car does not react to the ego; it follows the speed profile you give.
- A finding counts ONLY if the V2X ego collides AND the twin with correct information stays safe.

Fault semantics (exactly one fault per hypothesis; it only acts on messages that arrive while the
window is open, and messages only arrive while the cars are within 45 m):
- drop: messages inside the window are lost.
- freeze: messages keep arriving with fresh timestamps but repeat the content of the first message
  received inside the window.
- delay: each message inside the window is delivered `delay` seconds later (true timestamps).
- pos_perturb: reported position shifted by `bias` m along the lane (negative = reported farther from
  the junction than it really is) plus Gaussian noise `sigma`.
Window: `start` (s from simulation start) and `duration` (s, or null = until the end).

Crossing-car behavior: initial speed v0 plus up to 3 events {{t, accel, dur}} (constant acceleration
from time t for dur seconds). Hard limits: acceleration in [{amin}, {amax}] m/s^2, speed <= {vmax} m/s,
v0 >= {v0min}, event dur <= 10 s, |bias| <= 40, sigma <= 10, delay <= 5, horizon 20 s.
You may change ego_dist / cross_dist by up to +-20 m from the base scene.

Use the TIMELINE you are given: think about where the radio-contact period falls and what the ego
believes during it. Use the FEEDBACK: it tells you what actually happened in each simulated run.

Return ONLY JSON: {{"hypotheses": [ ... ]}}, each item:
{{"mechanism": "<one sentence: behavior -> information error -> wrong decision -> violation>",
  "predicts_failure": true|false,
  "ego_dist": <m>, "cross_dist": <m>,
  "cross": {{"v0": <m/s>, "events": [{{"t": <s>, "accel": <m/s^2>, "dur": <s>}}]}},
  "fault": {{"type": "...", "start": <s>, "duration": <s or null>, "bias": <m>, "sigma": <m>, "delay": <s>}}}}
Prefer DIFFERENT mechanisms; do not repeat ideas the feedback shows failed."""


def system_prompt_v2() -> str:
    L = LIMITS
    return SYSTEM_PROMPT_V2.format(amin=L["accel_min"], amax=L["accel_max"], vmax=L["v_max"], v0min=L["v0_min"])


def base_timeline(base: str) -> str:
    """Simulate the unfaulted base scene and describe its timeline in words."""
    scn = base_scenario(base)
    out = run_pair(scn, KinematicWorld(), keep_trace=True)
    b = BASE_SCENES[base]
    t_ego = b["ego_dist"] / b["ego_speed"]
    t_cross = b["cross_dist"] / b["cross_v0"]
    stop = out["stop_t_gt"]
    return (f"Base scene {base}: ego {b['ego_dist']} m from the conflict point at {b['ego_speed']} m/s "
            f"(would arrive at t={t_ego:.1f} s); crossing car {b['cross_dist']} m away at {b['cross_v0']} m/s "
            f"(would arrive at t={t_cross:.1f} s). {b['note']}\n"
            f"Timeline without faults: the cars come within radio range at t={out['t_in_range']:.1f} s; "
            f"first message received at t={out['first_rx_time']:.1f} s; "
            + (f"the ego decides to stop at t={stop:.1f} s; " if stop is not None else "the ego never needs to stop; ")
            + f"minimum gap {out['gap_gt']} m (no crash).")


def diagnose(scn: Scenario, out: Dict) -> str:
    notes = []
    f = scn.fault
    tr = out.get("t_in_range")
    if f.type != "none" and tr is not None and f.duration is not None and f.start + f.duration <= tr:
        notes.append(f"fault window ended (t={f.start + f.duration:.1f}s) before radio contact (t={tr:.1f}s): no effect")
    if f.type != "none" and tr is None:
        notes.append("cars never came within radio range")
    if out.get("first_rx_time") is None and tr is not None:
        notes.append("V2X ego never received any message")
    if out["v_gt"]:
        notes.append("the correctly-informed twin also crashed: physically unavoidable, not attributable")
    if out.get("stale_fallback") and not out["v_msg"]:
        notes.append("stale fail-safe triggered: V2X ego stopped")
    if out["label"] == "CLEAN" and out["gap_msg"] < 5:
        notes.append("near miss")
    return "; ".join(notes) or "no notable effect"


def feedback_line(scn: Scenario, out: Dict) -> str:
    f = scn.fault
    fmt = lambda x: "never" if x is None else f"{x:.1f}s"
    return (f"[{out['label']}] {scn.hypothesis} | ego_dist={scn.ego_dist} cross_dist={scn.cross_dist} "
            f"v0={scn.cross.v0} events={[(e.t, e.accel, e.dur) for e in scn.cross.events]} | "
            f"fault={f.type} start={f.start} dur={f.duration} bias={f.bias} delay={f.delay} | "
            f"radio contact {fmt(out.get('t_in_range'))}, first msg {fmt(out.get('first_rx_time'))}, "
            f"V2X ego braking from {fmt(out.get('stop_t_msg'))}, twin braking from {fmt(out.get('stop_t_gt'))}, "
            f"min gap V2X {out['gap_msg']} m vs twin {out['gap_gt']} m | {diagnose(scn, out)}")


class GroundedProposer:
    def __init__(self, model="gpt-4o", base_url=None, temperature=0.4):
        import os
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
        self.client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        self.model, self.temperature = model, temperature
        self._timelines: Dict[str, str] = {}

    def propose(self, base: str, n: int, history: List[str]) -> List[dict]:
        if base not in self._timelines:
            self._timelines[base] = base_timeline(base)
        msg = self._timelines[base] + f"\n\nPropose {n} hypotheses."
        if history:
            msg += "\n\nFEEDBACK from your simulated hypotheses so far (most recent last):\n" + "\n".join(history[-15:])
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system_prompt_v2()}, {"role": "user", "content": msg}])
        return list(_extract_json(r.choices[0].message.content).get("hypotheses", []))[:n]


def commit(screened: List[Tuple[Scenario, Dict]], budget: int) -> List[Scenario]:
    """Pick what goes to the real world: screen-admitted first, then closest near misses
    where the twin stayed safe."""
    adm = [s for s, o in screened if o["label"] == "ADMITTED"]
    rest = sorted([(o["gap_msg"], s) for s, o in screened if o["label"] != "ADMITTED" and not o["v_gt"]],
                  key=lambda x: x[0])
    picked = adm[:budget]
    picked += [s for _, s in rest[: budget - len(picked)]]
    return picked


def llm_screened_search(base: str, budget: int, screen: int, proposer, batch: int = 6, seed: int = 0,
                        log: Optional[Callable] = None):
    """Returns (committed scenarios, list of (scenario, screen outcome), n_invalid)."""
    kw = KinematicWorld()
    history: List[str] = []
    screened: List[Tuple[Scenario, Dict]] = []
    idx, invalid, calls = 0, 0, 0
    while len(screened) < screen and calls < 3 * (screen // batch + 1):
        calls += 1
        n = min(batch, screen - len(screened))
        try:
            hyps = proposer.propose(base, n, history)
        except Exception as e:
            if log:
                log(f"  [llm2] proposal failed: {e}")
            continue
        for h in hyps:
            try:
                scn = to_scenario(h, base, idx, seed=seed * 1000 + idx)
            except Exception as e:
                if log:
                    log(f"  [llm2] could not parse hypothesis: {e}")
                continue
            scn.name, scn.source = f"{base}_llm2_{idx:03d}", "llm2"
            idx += 1
            ok, why = validate(scn)
            if not ok:
                invalid += 1
                history.append(f"[PHYSICALLY_INVALID] {scn.hypothesis} | {why[0]}")
                continue
            out = run_pair(scn, kw)
            screened.append((scn, out))
            history.append(feedback_line(scn, out))
            if log:
                log(f"  screen llm2   {scn.name:16s} {scn.mechanism():26s} -> {out['label']}")
            if len(screened) >= screen:
                break
    return commit(screened, budget), screened, invalid


def random_screened_search(base: str, budget: int, screen: int, generator: Callable[..., Iterable[Scenario]],
                           seed: int = 0, log: Optional[Callable] = None):
    kw = KinematicWorld()
    screened = []
    for scn in generator(base, screen, seed=seed + 101):
        scn.name = scn.name.replace("_random_", "_rscr_")
        scn.source = "random_screened"
        out = run_pair(scn, kw)
        screened.append((scn, out))
        if log:
            log(f"  screen rscr   {scn.name:16s} {scn.mechanism():26s} -> {out['label']}")
    return commit(screened, budget), screened, 0


# ---------------------------------------------------------------------------------------------
# Version 3: HYBRID = a proposal (LLM mechanism or random) + numeric timing search.
# Finding from run 2: the grounded LLM chose the right failure mechanism ~4x more often than
# random, but never tuned arrival timing precisely enough (cars must meet within ~0.1-0.2 s).
# Here every promising proposal gets a small search over the crossing car's start distance.
# EVERY simulation (proposal + search) counts toward the same screen budget, and the random
# control (random_hybrid) gets the identical search, so any difference comes from the proposals.
# ---------------------------------------------------------------------------------------------

def info_effect(out: Dict) -> bool:
    """Did the V2X information change the ego's behavior compared with the twin?"""
    a, b = out.get("stop_t_msg"), out.get("stop_t_gt")
    if (a is None) != (b is None):
        return True
    return a is not None and abs(a - b) > 0.5


def promising(out: Dict) -> bool:
    """Neutral rule (same for LLM and random): twin safe, and either close or the info mattered."""
    return (not out["v_gt"]) and out["label"] != "ADMITTED" and (out["gap_msg"] < 15 or info_effect(out))


def timing_search(scn: Scenario, kw, remaining: int, max_d: float = 20.0):
    """Shift only cross_dist (within +-max_d m of the proposal): coarse 4 m grid, then refine
    around the closest V2X-ego gap where the twin stayed safe. Returns (best_scn, best_out, n_sims)."""
    from copy import deepcopy
    tried, n = {}, 0
    base_d = scn.cross_dist

    def run(d):
        nonlocal n
        s = deepcopy(scn)
        s.cross_dist = round(base_d + d, 2)
        if not (LIMITS["dist_min"] <= s.cross_dist <= LIMITS["dist_max"]):
            return None
        o = run_pair(s, kw)
        n += 1
        tried[d] = (s, o)
        return o

    for d in [x for x in range(-int(max_d), int(max_d) + 1, 4) if x != 0]:
        if n >= remaining:
            break
        o = run(d)
        if o and o["label"] == "ADMITTED":
            return tried[d][0], o, n
    cands = [(o["gap_msg"], d) for d, (s, o) in tried.items() if not o["v_gt"]]
    if cands:
        _, best = min(cands)
        for d in (best - 1, best + 1, best - 2, best + 2, best - 3, best + 3):
            if n >= remaining or d in tried or abs(d) > max_d or d == 0:
                continue
            o = run(d)
            if o and o["label"] == "ADMITTED":
                return tried[d][0], o, n
    if not tried:
        return None, None, n
    d = min(tried, key=lambda k: (tried[k][1]["v_gt"], tried[k][1]["gap_msg"]))
    return tried[d][0], tried[d][1], n


def hybrid_search(base: str, budget: int, screen: int, next_candidates: Callable[[List[str]], List[Scenario]],
                  tag: str, log: Optional[Callable] = None):
    """next_candidates(history) -> list of Scenario proposals (LLM batch or random draws)."""
    kw = KinematicWorld()
    history: List[str] = []
    screened: List[Tuple[Scenario, Dict]] = []   # every simulated scenario (proposals + search points)
    found: List[Tuple[Scenario, Dict]] = []
    used, empty_rounds = 0, 0
    while used < screen and empty_rounds < 5:
        props = next_candidates(history)
        if not props:
            empty_rounds += 1
            continue
        for scn in props:
            if used >= screen:
                break
            out = run_pair(scn, kw)
            used += 1
            screened.append((scn, out))
            line = feedback_line(scn, out)
            shown = out["label"]
            if out["label"] == "ADMITTED":
                found.append((scn, out))
            elif promising(out) and used < screen:
                s2, o2, n = timing_search(scn, kw, screen - used)
                used += n
                if s2 is not None:
                    s2.name = scn.name + "_t"
                    screened.append((s2, o2))
                    line += (f" || TIMING SEARCH ({n} sims, moved crossing car start by "
                             f"{round(s2.cross_dist - scn.cross_dist, 1)} m): {o2['label']}, "
                             f"V2X gap {o2['gap_msg']} m vs twin {o2['gap_gt']} m")
                    if o2["label"] == "ADMITTED":
                        found.append((s2, o2))
                        shown = "ADMITTED (after timing search)"
            history.append(line)
            if log:
                log(f"  screen {tag:6s} {scn.name:16s} {scn.mechanism():26s} -> {shown}   [{used}/{screen} sims]")
    return commit(screened, budget), screened, len(found)


def llm_hybrid(base, budget, screen, proposer, batch=6, seed=0, log=None):
    state = {"idx": 0, "invalid": 0}

    def nxt(history):
        try:
            hyps = proposer.propose(base, batch, history)
        except Exception as e:
            if log:
                log(f"  [llm3] proposal failed: {e}")
            return []
        out = []
        for h in hyps:
            try:
                scn = to_scenario(h, base, state["idx"], seed=seed * 1000 + state["idx"])
            except Exception:
                continue
            scn.name, scn.source = f"{base}_llm3_{state['idx']:03d}", "llm3"
            state["idx"] += 1
            ok, why = validate(scn)
            if not ok:
                state["invalid"] += 1
                history.append(f"[PHYSICALLY_INVALID] {scn.hypothesis} | {why[0]}")
                continue
            out.append(scn)
        return out

    committed, screened, n_found = hybrid_search(base, budget, screen, nxt, "llm3", log)
    return committed, screened, state["invalid"]


def random_hybrid(base, budget, screen, generator, seed=0, log=None):
    it = iter(generator(base, 10 * screen, seed=seed + 202))

    def nxt(history):
        out = []
        for _ in range(6):
            try:
                s = next(it)
            except StopIteration:
                break
            s.name, s.source = s.name.replace("_random_", "_rhyb_"), "random_hybrid"
            out.append(s)
        return out

    committed, screened, n_found = hybrid_search(base, budget, screen, nxt, "rhyb", log)
    return committed, screened, 0
