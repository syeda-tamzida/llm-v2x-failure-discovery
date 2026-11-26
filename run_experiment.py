"""Run the comparison: fixed vs random vs LLM hypotheses, same budget per base scene.

Examples
  # 1) no CARLA, offline mock LLM (checks everything works)
  python run_experiment.py --world kinematic --methods fixed random llm --llm mock

  # 2) CARLA (server must be running), real LLM
  python run_experiment.py --world carla --methods fixed random llm --llm openai --model gpt-4o

  # 3) one scenario, watch it live in the CARLA window
  python run_experiment.py --world carla --replay results/runs.jsonl --replay-name S2_llm_003 --watch
"""
from __future__ import annotations

import argparse
import json
import os
import time

from v2xdemo.methods import METHODS
from v2xdemo.physics import validate
from v2xdemo.runner import run_pair
from v2xdemo.scenario import Scenario


def make_world(args):
    if args.world == "carla":
        from v2xdemo.world_carla import CarlaWorld
        return CarlaWorld(args.host, args.port, args.town, watch=args.watch, junction_id=args.junction)
    from v2xdemo.world_kinematic import KinematicWorld
    return KinematicWorld()


def record(fh, method, scn: Scenario, valid, why, out=None):
    row = dict(method=method, base=scn.base, name=scn.name, valid=valid, why="; ".join(why),
               mechanism=scn.mechanism(), hypothesis=scn.hypothesis, predicts_failure=scn.predicts_failure,
               scenario=scn.to_dict())
    if out:
        row.update({k: v for k, v in out.items() if not k.startswith("trace")})
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    return row


def run_simple(method, base, args, world, fh):
    n = 0
    for scn in METHODS[method](base, args.budget, seed=args.seed):
        ok, why = validate(scn)
        if not ok:
            record(fh, method, scn, False, why)
            continue
        out = run_pair(scn, world)
        record(fh, method, scn, True, [], out)
        n += 1
        print(f"  {method:6s} {scn.name:16s} {scn.mechanism():26s} -> {out['label']}")


def run_llm(base, args, world, fh):
    from v2xdemo.llm_agent import make_proposer, to_scenario
    proposer = make_proposer(args.llm, args.model, args.base_url, args.seed)
    history, used, idx, attempts = [], 0, 0, 0
    while used < args.budget and attempts < 3 * args.budget:
        n = min(args.batch, args.budget - used)
        try:
            hyps = proposer.propose(base, n, history)
        except Exception as e:
            print(f"  [llm] proposal failed: {e}")
            attempts += n
            continue
        if not hyps:
            attempts += n
            continue
        for h in hyps:
            attempts += 1
            try:
                scn = to_scenario(h, base, idx, seed=args.seed * 1000 + idx)
            except Exception as e:
                print(f"  [llm] could not parse hypothesis: {e}")
                continue
            idx += 1
            ok, why = validate(scn)
            if not ok:
                record(fh, "llm", scn, False, why)
                history.append(dict(label="PHYSICALLY_INVALID", mechanism=scn.hypothesis, fault=scn.fault.type,
                                    behavior=scn.cross.label(), gap_msg="-", gap_gt="-", why="; ".join(why)))
                print(f"  llm    {scn.name:16s} rejected by physics: {why[0]}")
                continue
            out = run_pair(scn, world)
            record(fh, "llm", scn, True, [], out)
            used += 1
            history.append(dict(label=out["label"], mechanism=scn.hypothesis, fault=scn.fault.type,
                                behavior=scn.cross.label(), gap_msg=out["gap_msg"], gap_gt=out["gap_gt"]))
            print(f"  llm    {scn.name:16s} {scn.mechanism():26s} -> {out['label']}")
            if used >= args.budget:
                break


def run_screened(method, base, args, world, fh):
    """llm2 / random_screened: search in the fast kinematic world (screen budget), then commit the
    best `budget` scenarios to the real world (kinematic or CARLA)."""
    from v2xdemo import agent_v2
    from v2xdemo.methods import random_method
    if method in ("llm2", "llm3"):
        if args.llm == "mock":
            from v2xdemo.llm_agent import MockProposer
            proposer = MockProposer(args.seed)
        else:
            proposer = agent_v2.GroundedProposer(model=args.model, base_url=args.base_url)
        fn = agent_v2.llm_screened_search if method == "llm2" else agent_v2.llm_hybrid
        committed, screened, invalid = fn(base, args.budget, args.screen, proposer,
                                          batch=args.batch, seed=args.seed, log=print)
    elif method == "random_hybrid":
        committed, screened, invalid = agent_v2.random_hybrid(
            base, args.budget, args.screen, random_method, seed=args.seed, log=print)
    else:
        committed, screened, invalid = agent_v2.random_screened_search(
            base, args.budget, args.screen, random_method, seed=args.seed, log=print)
    for scn, out in screened:
        record(fh, method + "@screen", scn, True, [], out)
    for _ in range(invalid):
        pass
    if invalid:
        fh.write(json.dumps(dict(method=method + "@screen", base=base, name=f"{base}_{method}_invalid",
                                 valid=False, why=f"{invalid} physically invalid proposals", mechanism="-",
                                 hypothesis="", predicts_failure=None, scenario={}, n_invalid=invalid)) + "\n")
    print(f"  -> committing {len(committed)} of {len(screened)} screened scenarios to the {args.world} world")
    for scn in committed:
        out = run_pair(scn, world)
        record(fh, method, scn, True, [], out)
        print(f"  {method:6s} {scn.name:16s} {scn.mechanism():26s} -> {out['label']}")


def replay(args, world):
    rows = [json.loads(l) for l in open(args.replay)]
    row = next(r for r in rows if r["name"] == args.replay_name)
    scn = Scenario.from_dict(row["scenario"])
    print(f"Replaying {scn.name}: {scn.hypothesis or scn.mechanism()}")
    out = run_pair(scn, world, keep_trace=True)
    print(f"label={out['label']}  gap with V2X={out['gap_msg']} m  gap with true info={out['gap_gt']} m")
    path = os.path.join(os.path.dirname(args.replay), f"trace_{scn.name}.json")
    json.dump(dict(scenario=scn.to_dict(), outcome={k: v for k, v in out.items()}), open(path, "w"))
    print(f"trace saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["kinematic", "carla"], default="kinematic")
    ap.add_argument("--methods", nargs="+", default=["fixed", "random", "llm"])
    ap.add_argument("--bases", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--budget", type=int, default=30, help="simulated scenario pairs per method per base scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--llm", choices=["openai", "mock"], default="mock")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible URL, e.g. http://localhost:11434/v1 for Ollama")
    ap.add_argument("--batch", type=int, default=6, help="hypotheses per LLM call")
    ap.add_argument("--screen", type=int, default=60, help="fast kinematic checks per base scene (llm2, random_screened)")
    # CARLA
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--junction", type=int, default=None, help="force a specific CARLA junction id")
    ap.add_argument("--watch", action="store_true", help="render and run in real time (for videos)")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--replay-name", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    world = make_world(args)
    try:
        if args.replay:
            replay(args, world)
            return
        path = os.path.join(args.out, f"runs_{args.world}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        with open(path, "w") as fh:
            for base in args.bases:
                for m in args.methods:
                    print(f"\n=== {base} / {m} ===")
                    if m == "llm":
                        run_llm(base, args, world, fh)
                    elif m in ("llm2", "random_screened", "llm3", "random_hybrid"):
                        run_screened(m, base, args, world, fh)
                    else:
                        run_simple(m, base, args, world, fh)
        print(f"\nSaved runs to {path}\nNext: python analyze.py {path}")
    finally:
        world.close()


if __name__ == "__main__":
    main()
