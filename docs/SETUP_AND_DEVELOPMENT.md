# V2X Causal-Reasoning Demo (CARLA)

**Question:** Can an LLM that reasons about *how a traffic situation could evolve* (driver behavior + V2X
information errors) find communication-induced safety failures that fixed-behavior testing misses,
at the same simulation budget?

This builds on the evaluation idea of LAST-V2X (Wu et al., 2026): a failure counts only if the ego that
acts on V2X messages crashes **and** an identical ego with perfect information stays safe. LAST-V2X
varies faults and initial conditions but keeps driver behavior fixed. This demo adds **behavioral
uncertainty** and compares three methods:

| Method | Crossing-car behavior | Fault + initial conditions |
|---|---|---|
| `fixed` | constant speed (LAST-V2X-style) | random |
| `random` | random accelerate/brake events | random |
| `llm` | LLM causal hypotheses, physics-checked, with feedback between rounds | chosen by the LLM |

`random` is the key control: if the LLM can't beat it, the LLM is just generating more scenarios.

---

## What's in the folder

```
run_experiment.py     run the comparison (or replay one scenario)
analyze.py            summary table + CSV
plot_trace.py         figure: ego vs ground-truth twin for one scenario
v2xdemo/
  scenario.py         scenario data: distances, speeds, behavior events, fault; base scenes S1-S3
  physics.py          hard physical limits (semantic ideas must pass physics)
  messages.py         BSM -> channel (45 m range) -> fault engine -> receiver belief
  ima.py              Intersection Movement Assist logic (LAST-V2X Appendix D)
  runner.py           paired run (V2X ego vs ground-truth twin) + causal admission
  methods.py          fixed and random baselines
  llm_agent.py        LLM hypothesis agent (OpenAI API, local Qwen, or offline mock)
  world_kinematic.py  no-CARLA world (two straight crossing lanes) for development
  world_carla.py      CARLA world: finds a real junction, drives cars along map waypoints
```

**Base scenes** (ego always 100 m away at 10 m/s):
- **S1:** the crossing car arrives at about the same time, so the scene is analytically conflicting.
- **S2:** it arrives about 5 s later, so the scene is analytically non-conflicting.
- **S3:** it arrives about 4.5 s earlier, also non-conflicting.

S2 and S3 are where reasoning about behavior should matter.

**Faults:** `drop`, `freeze` (fresh-but-wrong), `delay`, `pos_perturb`.

---

## Step-by-step guide

### Step 1. Check your machine
- An NVIDIA GPU (about 6 GB of VRAM minimum, 8 GB or more is better) with up-to-date drivers.
- About 20 GB of free disk space.
- Windows 10/11 or Ubuntu 20.04/22.04.
- If your laptop has no NVIDIA GPU, use a lab or WPI GPU machine. You can still do Steps 4–5 on any laptop without CARLA.

### Step 2. Install CARLA 0.9.15 (the simulator server)
1. Go to the CARLA GitHub releases page (github.com/carla-simulator/carla/releases) and open **0.9.15**.
2. Download the package for your OS: **CARLA_0.9.15.zip** (Windows) or **CARLA_0.9.15.tar.gz** (Linux). Town05 is included in the main package.
3. Extract it to a folder with no spaces in the path, for example `C:\CARLA_0.9.15` or `~/CARLA_0.9.15`.
4. Start the server:
   - Windows: double-click `CarlaUE4.exe`, or in a terminal run `CarlaUE4.exe -quality-level=Low`
   - Linux: `./CarlaUE4.sh -quality-level=Low` (add `-RenderOffScreen` to run without a window)
5. Wait until the 3D window appears (the first start can take a minute). Leave it running. It listens on port 2000.

### Step 3. Set up Python (the client)
CARLA 0.9.15's Python package works best with **Python 3.8**. Using conda:
```bash
conda create -n v2xdemo python=3.8 -y
conda activate v2xdemo
cd v2x_reasoning_demo
pip install -r requirements.txt
pip install carla==0.9.15
```
If `pip install carla==0.9.15` fails, install the wheel that ships with CARLA instead:
```bash
pip install <CARLA folder>/PythonAPI/carla/dist/carla-0.9.15-cp38-*.whl
```
Check that it works (with the server running):
```bash
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); print(c.get_world().get_map().name)"
```

### Step 4. Run once without CARLA (checks that the logic works, takes seconds)
```bash
python run_experiment.py --world kinematic --llm mock --budget 30
python analyze.py results/runs_kinematic_<timestamp>.jsonl
```
`--llm mock` is a rule-based stand-in so you can test offline. **Never report mock numbers.**

### Step 5. Connect a real LLM
**Option A: OpenAI**
```bash
# Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
python run_experiment.py --world kinematic --llm openai --model gpt-4o --budget 30
```
**Option B: local Qwen (free, no API key) with Ollama**
```bash
ollama pull qwen2.5:7b
python run_experiment.py --world kinematic --llm openai --base-url http://localhost:11434/v1 --model qwen2.5:7b
```
Look at the `physically_valid_pct` column: it tells you how often the LLM proposes physically impossible behavior.

### Step 6. Run in CARLA
With the CARLA server running:
```bash
python run_experiment.py --world carla --town Town05 --llm openai --model gpt-4o --budget 30
python analyze.py results/runs_carla_<timestamp>.jsonl
```
- The script loads Town05 and **automatically finds a junction** with two straight, perpendicular movements and enough approach road. It prints the junction id.
- To fix the junction across runs, use `--junction <id>`.
- Runs are in synchronous mode (50 ms ticks) without rendering, so they're fast. 3 scenes × 3 methods × 30 pairs is about 270 paired runs; expect roughly 20–40 minutes depending on your GPU.
- If CARLA crashes or you press Ctrl+C, the script restores CARLA's normal settings. If the window freezes afterward, restart the server.

### Step 7. Watch a failure and make figures and videos
Pick an `ADMITTED` scenario name from the run log or the jsonl file, then:
```bash
python run_experiment.py --world carla --replay results/runs_carla_<timestamp>.jsonl --replay-name S2_llm_003 --watch
python plot_trace.py results/trace_S2_llm_003.json
```
`--watch` renders in real time with a top-down camera over the junction. Screen-record it (Win+G on Windows, OBS on Linux). The blue car is the ego and the red car is the crossing car. The replay runs the V2X ego first and the ground-truth twin second, so record both. `plot_trace.py` makes a figure comparing the ego, the twin, and the true vs. believed position of the crossing car.

### Step 8. Put it on GitHub
Include: this README, your results table (from `analyze.py`), one trace figure, and one short video. Link the repo in your revised research statement.

---

## How to read the results

| Column | Meaning |
|---|---|
| `admitted` | verified communication-induced failures (V2X ego crashes, twin is safe) |
| `non_attributable` | both crash: the traffic itself was unsafe, not the communication |
| `distinct_mechanisms` | different fault + behavior combinations among admitted failures |
| `runs_to_first_failure` | simulation efficiency |
| `physically_valid_pct` | share of proposals that passed the physics filter (LLM only) |
| `explanation_accuracy_pct` | how often the LLM's `predicts_failure` matched the actual outcome |

To make claims, run several seeds (`--seed 0`, `--seed 1`, ...) and report mean ± std.

## Known simplifications (be upfront about these)
- **Vehicle motion:** vehicles move kinematically along map waypoints (`set_transform`, physics off), like scripted controllers. CARLA supplies the real road geometry, junction and rendering, not vehicle dynamics. A natural next step is a PID controller with physics on.
- **Collision rule:** center-to-center distance < 1.8 m (as in LAST-V2X Appendix D). A bounding-box check would be stricter.
- **Scope:** one application (IMA), one V2X stream (the crossing car's BSMs), and representative fault magnitudes, not measured ones.
- **Crossing-car behavior** is open-loop: it doesn't react to the ego.
- **Silent case:** a `drop` that covers the whole time the cars are in radio range means the ego never hears the crossing car, which is the "silent-by-absence" case. IMA then assumes no conflict, which matches the paper's finding that absent information is dangerous.

---

## Version 2: grounded LLM agent (`llm2`) and fair screened baseline (`random_screened`)

The first CARLA run showed that plain GPT-4o hypotheses (`llm`) did not beat random search (0 vs 2
verified failures in 90 runs). Diagnosis from the run log:
- 85 of 90 GPT-4o faults started at t=0, and 34 of 90 ended before the cars were even within radio
  range, so they could have no effect.
- It only used the three base geometries and repeated about 14 mechanism templates.
- The v1 instructions said `drop` "usually triggers the stale fail-safe", which is misleading.
- All 4 real failures found by the baselines had the same mechanism: the ego received **no message
  during the first 2–2.5 s of radio contact** (delay or drop). Because it never had a track of the
  crossing car, IMA assumed the junction was clear, and it braked too late or never.

`llm2` fixes these:
1. It gives the LLM a grounded timeline (radio contact, arrivals, twin braking).
2. It states the exact receiver rules.
3. It returns rich per-run feedback with an automatic diagnosis.
4. It **self-checks** every hypothesis in the fast kinematic world (the "screen") before committing the
   best ones to the real world.

`random_screened` gets exactly the same screen and commit budgets, so the comparison is fair.

```
python run_experiment.py --world carla --methods fixed random llm2 random_screened --llm openai --model gpt-4o --budget 30 --screen 60
python analyze.py results\runs_carla_<timestamp>.jsonl
```
Rows ending in `@screen` are the fast kinematic checks; rows without it are the committed real-world runs.
Compare `admitted` and `distinct_mechanisms` at the same `--screen` budget.

Caveat to report: the v2 instructions describe the receiver rules precisely, which is domain knowledge.
To measure how much that knowledge helps, run an ablation that removes the rules from the prompt.

---

## Version 3: hybrid (`llm3`) and its control (`random_hybrid`)

Run 2 showed that the grounded LLM chose the right failure mechanism about 4x more often than random
(43% vs 11–20% of runs), but it never tuned arrival timing precisely enough (the cars must meet within
~0.1–0.2 s). In version 3, every *promising* proposal gets a small numeric timing search (it shifts the
crossing car's start by up to ±20 m: a coarse 4 m grid, then refinement).

The "promising" rule is the same for both methods: the twin is safe, and either the gap is under 15 m or
the V2X information changed when the ego braked. Every simulation (proposal + search) counts toward
`--screen`.

- `llm3` = grounded GPT-4o mechanism + timing search
- `random_hybrid` = random proposals + the **identical** timing search (the key control: does the
  LLM add anything beyond the search?)

```
python run_experiment.py --world carla --methods fixed random random_hybrid llm3 --llm openai --model gpt-4o --budget 30 --screen 120
```
