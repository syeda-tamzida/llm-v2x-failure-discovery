"""Plot a replayed scenario: message-driven ego vs ground-truth twin.

Usage: python plot_trace.py results/trace_S2_fixed_005.json
"""
import json
import sys
import textwrap

import matplotlib.pyplot as plt


def main(path):
    d = json.load(open(path))
    scn, out = d["scenario"], d["outcome"]
    tm, tg = out["trace_msg"], out["trace_gt"]
    ego_cp, cross_cp = scn["ego_dist"], scn["cross_dist"]

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    t = [r["t"] for r in tm]
    ax[0].plot(t, [ego_cp - r["ego_s"] for r in tm], label="ego (acts on V2X messages)", color="#c0392b")
    ax[0].plot([r["t"] for r in tg], [ego_cp - r["ego_s"] for r in tg], "--", label="ground-truth twin", color="#2471a3")
    ax[0].plot(t, [cross_cp - r["cross_s"] for r in tm], label="crossing car (true)", color="#555")
    ax[0].plot(t, [None if r["belief_s"] is None else cross_cp - r["belief_s"] for r in tm], ":",
               label="crossing car (believed by ego)", color="#555")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("distance to conflict point (m)")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, [r["ego_v"] for r in tm], color="#c0392b", label="ego")
    ax[1].plot([r["t"] for r in tg], [r["ego_v"] for r in tg], "--", color="#2471a3", label="twin")
    ax[1].plot(t, [r["cross_v"] for r in tm], color="#555", label="crossing car")
    ax[1].set_ylabel("speed (m/s)")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, [r["gap"] for r in tm], color="#c0392b", label=f"ego, min {out['gap_msg']} m")
    ax[2].plot([r["t"] for r in tg], [r["gap"] for r in tg], "--", color="#2471a3", label=f"twin, min {out['gap_gt']} m")
    ax[2].axhline(1.8, color="k", lw=0.6, ls=":")
    ax[2].set_ylim(0, 60)
    ax[2].set_ylabel("gap between cars (m)")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(fontsize=8)
    f = scn["fault"]
    hyp = textwrap.fill(scn.get('hypothesis') or '', 95)
    fig.suptitle(f"{scn['name']}  [{out['label']}]  fault={f['type']}\n{hyp}", fontsize=9)
    fig.tight_layout()
    png = path.replace(".json", ".png")
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main(sys.argv[1])
