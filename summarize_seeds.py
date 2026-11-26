"""Aggregate multi-seed CARLA runs into the headline table and figure.

Usage: python summarize_seeds.py data/seed_runs/*.jsonl
Writes results_summary.csv and figures/results_5seeds.png
"""
import csv
import json
import statistics as st
import sys

import matplotlib.pyplot as plt

METHODS = ["random", "random_hybrid", "llm3"]
LABELS = {"random": "Random", "random_hybrid": "Random + timing search", "llm3": "Grounded LLM + timing search"}
COLORS = {"random": "#257de0", "random_hybrid": "#f47230", "llm3": "#00b894"}  # fixed categorical order


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main(files):
    files = sorted(files)
    per = {m: dict(adm=[], nonattr=[], scr=[], mech=set(), by_scene={}) for m in METHODS}
    for f in files:
        rows = load(f)
        for m in METHODS:
            rs = [r for r in rows if r["method"] == m and r.get("label")]
            adm = [r for r in rs if r["label"] == "ADMITTED"]
            per[m]["adm"].append(len(adm))
            per[m]["nonattr"].append(sum(r["label"] == "NON_ATTRIBUTABLE" for r in rs))
            per[m]["scr"].append(sum(1 for r in rows if r["method"] == m + "@screen" and r.get("label") == "ADMITTED"))
            for r in adm:
                per[m]["mech"].add((r["base"], r["mechanism"]))
                per[m]["by_scene"][r["base"]] = per[m]["by_scene"].get(r["base"], 0) + 1

    out = []
    for m in METHODS:
        p = per[m]
        conf = f"{sum(p['adm'])}/{sum(p['scr'])}" if sum(p["scr"]) else "-"
        out.append(dict(method=m, seeds=len(files),
                        verified_per_seed_mean=round(st.mean(p["adm"]), 2),
                        verified_per_seed_sd=round(st.stdev(p["adm"]), 2) if len(files) > 1 else 0,
                        non_attributable_per_seed=round(st.mean(p["nonattr"]), 2),
                        screen_confirmed_in_carla=conf,
                        distinct_scene_mechanisms=len(p["mech"]),
                        verified_by_scene=json.dumps(dict(sorted(p["by_scene"].items()))),
                        per_seed=json.dumps(p["adm"])))
    with open("results_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)

    ink, ink2, grid = "#222222", "#6b6b6b", "#e6e6e6"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = range(len(METHODS))
    # (a) verified failures per seed: mean bar + individual seeds
    ax = axes[0]
    for i, m in enumerate(METHODS):
        ax.bar(i, st.mean(per[m]["adm"]), width=0.55, color=COLORS[m], zorder=2)
        jitter = [i - 0.12 + 0.06 * k for k in range(len(per[m]["adm"]))]
        ax.scatter(jitter, per[m]["adm"], s=22, color="white", edgecolor=ink, linewidth=1, zorder=3)
    ax.set_title("Verified failures per seed\n(bar = mean, dots = 5 seeds)", fontsize=10, color=ink)
    # (b) precision: share of fast-check failures confirmed in CARLA (hybrids only)
    ax = axes[1]
    hy = ["random_hybrid", "llm3"]
    vals = [100 * sum(per[m]["adm"]) / max(1, sum(per[m]["scr"])) for m in hy]
    ax.bar(range(2), vals, width=0.55, color=[COLORS[m] for m in hy], zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color=ink)
    ax.set_ylim(0, 110)
    ax.set_xticks(range(2))
    ax.set_xticklabels([LABELS[m].replace(" + ", "\n+ ") for m in hy], fontsize=8, color=ink2)
    ax.set_title("Fast-check failures confirmed\nin CARLA (precision)", fontsize=10, color=ink)
    # (c) coverage: distinct (scene, mechanism) failures over all seeds
    ax = axes[2]
    vals = [len(per[m]["mech"]) for m in METHODS]
    ax.bar(x, vals, width=0.55, color=[COLORS[m] for m in METHODS], zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9, color=ink)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title("Distinct failures found\n(scene × mechanism, all seeds)", fontsize=10, color=ink)
    for ax in (axes[0], axes[2]):
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[m].replace(" + ", "\n+ ") for m in METHODS], fontsize=8, color=ink2)
    for ax in axes:
        ax.grid(axis="y", color=grid, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#bdbdbd")
        ax.tick_params(axis="y", colors=ink2, labelsize=8, length=0)
    axes[0].set_ylim(0, max(max(per[m]["adm"]) for m in METHODS) * 1.15)
    fig.text(0.5, 0.015, "CARLA Town05 · IMA · 3 base scenes · 5 seeds", ha="center", fontsize=9, color=ink2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("figures/results_5seeds.png", dpi=160)
    print("saved figures/results_5seeds.png and results_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
