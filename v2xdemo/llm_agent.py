"""LLM causal-hypothesis agent.

The LLM does NOT run the simulation and does NOT decide whether a failure is real.
It reads the scene, imagines plausible ways the situation could evolve
(behavior + communication), and writes each as a causal hypothesis in JSON.
Code then checks physics, simulates the hypothesis with the ground-truth twin,
and feeds the outcomes back so the next round can learn from them.

Providers:
  openai : any OpenAI-compatible API (OpenAI, or a local Qwen served by Ollama / vLLM
           via --base-url, e.g. http://localhost:11434/v1)
  mock   : rule-based stand-in for testing the pipeline offline. NOT a result.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional

from .physics import LIMITS
from .scenario import BASE_SCENES, Behavior, BehaviorEvent, Fault, Scenario

SYSTEM_PROMPT = """You are a transportation safety researcher designing tests for a connected-vehicle
Intersection Movement Assist (IMA) application. Your job is to propose CAUSAL HYPOTHESES: plausible
ways a traffic situation could develop so that faulty V2X information makes the ego vehicle take an
unsafe decision that it would NOT take with correct information.

Scene: an unsignalized junction. The ego drives straight toward the conflict point. A crossing car
approaches from the right on a perpendicular lane. The crossing car broadcasts BSMs (position along
its lane + speed) at 10 Hz; they reach the ego only within 45 m.

Ego IMA logic (exactly what runs in simulation):
- It extrapolates the crossing car's position from the latest BSM: s_now = s + v * age.
- It STOPS if the crossing car's predicted arrival at the conflict point is within 3.5 s of the ego's
  own arrival (ego arrivals within 12 s considered). Otherwise it goes.
- If the latest BSM is older than 0.5 s it treats the information as stale and stops (fail-safe).
- If it has never received a BSM it assumes no conflict and goes.
- Reaction delay 1.0 s; comfortable braking 3.5 m/s^2; it commits once it cannot stop.
- The crossing car does NOT react to the ego; it follows the speed profile you give it.

A test counts as a real finding ONLY if the ego collides when using the V2X messages AND an identical
ego that receives perfectly correct information stays safe. So the danger must come from the
information, not from an unavoidable traffic situation.

Fault vocabulary (exactly one per hypothesis):
- drop: no messages delivered during the window (usually triggers the stale fail-safe)
- freeze: messages keep arriving with fresh timestamps but repeat the content captured when the window
  opened (fresh-but-wrong)
- delay: messages arrive late by `delay` seconds (their timestamps stay true, so they look old)
- pos_perturb: reported position shifted by `bias` m along the lane (negative = reported behind the
  true position, i.e. farther from the junction) plus Gaussian noise `sigma`
Fault window: `start` (s from simulation start) and `duration` (s, or null = until the end).

Crossing-car behavior: initial speed `v0` plus up to 3 events {{t, accel, dur}}: constant acceleration
`accel` (m/s^2) from time t for dur seconds. Think like an engineer: drivers speed up to beat a gap,
slow down for a turning vehicle then go, brake for a pedestrian, etc.

Hard physical limits (violating them wastes your attempt):
acceleration in [{amin}, {amax}] m/s^2, speeds in [0, {vmax}] m/s, v0 >= {v0min}, at most 3 events,
event duration <= 10 s, |bias| <= 40 m, sigma <= 10 m, delay <= 5 s, horizon 20 s.
You may also adjust ego_dist / cross_dist by at most +-20 m from the base scene.

Return ONLY a JSON object: {{"hypotheses": [ ... ]}} where each item is
{{"mechanism": "<one sentence causal chain: behavior -> information error -> wrong decision -> violation>",
  "predicts_failure": true|false,
  "ego_dist": <m>, "cross_dist": <m>,
  "cross": {{"v0": <m/s>, "events": [{{"t": <s>, "accel": <m/s^2>, "dur": <s>}}]}},
  "fault": {{"type": "...", "start": <s>, "duration": <s or null>, "bias": <m>, "sigma": <m>, "delay": <s>}}}}
Prefer DIFFERENT mechanisms across hypotheses rather than variations of one idea."""


def system_prompt() -> str:
    L = LIMITS
    return SYSTEM_PROMPT.format(amin=L["accel_min"], amax=L["accel_max"], vmax=L["v_max"], v0min=L["v0_min"])


def user_prompt(base: str, n: int, history: List[Dict]) -> str:
    b = BASE_SCENES[base]
    msg = (f"Base scene {base}: ego {b['ego_dist']} m from the conflict point at {b['ego_speed']} m/s; "
           f"crossing car {b['cross_dist']} m from the conflict point at {b['cross_v0']} m/s. {b['note']}\n"
           f"Propose {n} hypotheses.")
    if history:
        msg += "\n\nResults of your previous hypotheses (learn from them; avoid repeating failed ideas):\n"
        for h in history[-12:]:
            msg += (f"- [{h['label']}] {h['mechanism']} | fault={h['fault']} behavior={h['behavior']} "
                    f"| min gap with V2X {h['gap_msg']} m vs true info {h['gap_gt']} m"
                    f"{' | rejected by physics: ' + h['why'] if h.get('why') else ''}\n")
    return msg


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in LLM output")
    return json.loads(m.group(0))


def to_scenario(h: dict, base: str, idx: int, seed: int) -> Scenario:
    b = BASE_SCENES[base]
    cross = h.get("cross", {})
    f = h.get("fault", {}) or {}
    dur = f.get("duration")
    return Scenario(
        name=f"{base}_llm_{idx:03d}", base=base,
        ego_dist=float(h.get("ego_dist", b["ego_dist"])), ego_speed=float(b["ego_speed"]),
        cross_dist=float(h.get("cross_dist", b["cross_dist"])),
        cross=Behavior(v0=float(cross.get("v0", b["cross_v0"])),
                       events=[BehaviorEvent(float(e["t"]), float(e["accel"]), float(e["dur"]))
                               for e in cross.get("events", [])]),
        fault=Fault(type=str(f.get("type", "none")), start=float(f.get("start", 0) or 0),
                    duration=None if dur in (None, "null") else float(dur),
                    bias=float(f.get("bias", 0) or 0), sigma=float(f.get("sigma", 0) or 0),
                    delay=float(f.get("delay", 0) or 0)),
        seed=seed, source="llm", hypothesis=str(h.get("mechanism", "")),
        predicts_failure=h.get("predicts_failure"))


class OpenAIProposer:
    def __init__(self, model: str = "gpt-4o", base_url: Optional[str] = None, temperature: float = 0.2):
        from openai import OpenAI   # pip install openai
        key = os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
        self.client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        self.model, self.temperature = model, temperature

    def propose(self, base: str, n: int, history: List[Dict]) -> List[dict]:
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system_prompt()},
                      {"role": "user", "content": user_prompt(base, n, history)}])
        data = _extract_json(r.choices[0].message.content)
        return list(data.get("hypotheses", []))[:n]


class MockProposer:
    """Offline stand-in that writes a few hand-made 'engineer-style' hypotheses.
    Only for checking that the pipeline runs. Do NOT report its numbers."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, base: str, n: int, history: List[Dict]) -> List[dict]:
        b = BASE_SCENES[base]
        lead = (b["cross_dist"] / b["cross_v0"]) - (b["ego_dist"] / b["ego_speed"])  # +: cross arrives later
        out = []
        for _ in range(n):
            j = lambda x, r: round(x + self.rng.uniform(-r, r), 2)
            if lead > 2:     # crossing car is late: it speeds up while the ego is told it is farther away
                events = [{"t": j(2.0, 1), "accel": j(2.0, 0.3), "dur": j(3.0, 0.5)}]
                fault = self.rng.choice([
                    {"type": "freeze", "start": j(5.0, 1.5), "duration": None},
                    {"type": "pos_perturb", "start": j(4.0, 2), "duration": None, "bias": j(-20, 6), "sigma": 0.5}])
                mech = "crossing driver accelerates to beat the gap; ego's belief keeps the car far away -> ego goes"
            elif lead < -2:  # crossing car is early: it slows (yielding) then goes, ego believes it has passed
                events = [{"t": j(1.5, 1), "accel": j(-3.0, 0.5), "dur": j(2.5, 0.5)},
                          {"t": j(5.0, 1), "accel": j(2.0, 0.5), "dur": j(2.0, 0.5)}]
                fault = self.rng.choice([
                    {"type": "pos_perturb", "start": j(3.0, 2), "duration": None, "bias": j(20, 6), "sigma": 0.5},
                    {"type": "freeze", "start": j(6.0, 1.5), "duration": None}])
                mech = "crossing driver hesitates then goes; ego believes it already cleared the junction -> ego goes"
            else:
                events = []
                fault = {"type": "freeze", "start": j(5.0, 2), "duration": None}
                mech = "simultaneous arrival; frozen BSM hides the approaching car -> ego goes"
            out.append({"mechanism": mech, "predicts_failure": True, "ego_dist": b["ego_dist"],
                        "cross_dist": b["cross_dist"], "cross": {"v0": b["cross_v0"], "events": events},
                        "fault": fault})
        return out


def make_proposer(provider: str, model: str, base_url: Optional[str], seed: int):
    if provider == "mock":
        return MockProposer(seed)
    return OpenAIProposer(model=model, base_url=base_url)
