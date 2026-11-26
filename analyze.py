"""Summarize a runs_*.jsonl file into the comparison table (prints Markdown, writes CSV).

Usage: python analyze.py results/runs_kinematic_YYYYmmdd_HHMMSS.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["base"], r["method"])].append(r)
        groups[("ALL", r["method"])].append(r)

    table = []
    for (base, method), rs in sorted(groups.items()):
        sim = [r for r in rs if r["valid"] and "label" in r]
        n_invalid = sum(r.get("n_invalid", 0) for r in rs)
        adm = [r for r in sim if r["label"] == "ADMITTED"]
        first = next((i + 1 for i, r in enumerate(sim) if r["label"] == "ADMITTED"), None)
        preds = [r for r in sim if r.get("predicts_failure") is not None]
        acc = (sum(1 for r in preds if bool(r["predicts_failure"]) == (r["label"] == "ADMITTED")) / len(preds)
               if preds else None)
        row = dict(
            base=base, method=method, proposed=len(sim) + sum(1 for r in rs if not r["valid"] and "n_invalid" not in r) + n_invalid,
            simulated=len(sim),
            physically_valid_pct=0,
            admitted=len(adm),
            non_attributable=sum(1 for r in sim if r["label"] == "NON_ATTRIBUTABLE"),
            clean=sum(1 for r in sim if r["label"] == "CLEAN"),
            distinct_mechanisms=len({r["mechanism"] for r in adm}),
            runs_to_first_failure=first if base != "ALL" else "-",
            explanation_accuracy_pct=None if acc is None else round(100 * acc, 1),
        )
        row["physically_valid_pct"] = round(100 * len(sim) / row["proposed"], 1) if row["proposed"] else 0
        table.append(row)

    cols = list(table[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for t in table:
        print("| " + " | ".join("" if t[c] is None else str(t[c]) for c in cols) + " |")

    out = path.replace(".jsonl", "_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"\nCSV: {out}")

    print("\nAdmitted failure mechanisms (fault + crossing behavior):")
    for (base, method), rs in sorted(groups.items()):
        if base == "ALL":
            continue
        mech = defaultdict(int)
        for r in rs:
            if r.get("label") == "ADMITTED":
                mech[r["mechanism"]] += 1
        if mech:
            print(f"  {base} {method}: " + ", ".join(f"{k} x{v}" for k, v in sorted(mech.items())))


if __name__ == "__main__":
    main(sys.argv[1])
