#!/usr/bin/env python3
"""Combine the two paper layer-selection criteria and pick the Qwen layer.

Reads:
  - results/layer_select/bootstrap_{model}.json         (criterion ii: stability)
  - results/layer_select/selfsteer/selfsteer_lift.json  (criterion i: self-lift, layer x alpha)

The paper (§A.2) selects the layer at alpha=2 whose self-steer lift is "among the
largest in the sweep" AND whose bootstrap stability is high. We take the lift at
--select-alpha (default 2), rank each criterion across layers, min-max normalise to
[0,1], and score each layer by the mean of the two normalised values. The plot shows
lift-vs-layer for EVERY swept alpha (so you can see if the best layer is alpha-robust)
plus the stability curve.

Usage:
    python pipeline/ls_select_layer.py --model-name Qwen2.5-7B-Instruct --select-alpha 2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LS = REPO_ROOT / "results" / "layer_select"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine bootstrap-stability + self-lift, pick the layer")
    p.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    p.add_argument("--select-alpha", type=float, default=2.0, help="alpha used for the pick (paper: 2)")
    p.add_argument("--bootstrap", default=None)
    p.add_argument("--selfsteer", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def norm01(d):
    if not d:
        return {}
    vals = np.array(list(d.values()), float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def main() -> None:
    args = parse_args()
    bpath = Path(args.bootstrap) if args.bootstrap else LS / f"bootstrap_{args.model_name}.json"
    spath = Path(args.selfsteer) if args.selfsteer else LS / "selfsteer" / "selfsteer_lift.json"
    out_dir = Path(args.output_dir) if args.output_dir else LS
    out_dir.mkdir(parents=True, exist_ok=True)

    stab = {r["layer"]: r["stability"] for r in json.load(open(bpath))["rows"]} if bpath.exists() else {}
    # self-lift: rows carry (layer, alpha). group into {alpha: {layer: lift}}
    lift_by_alpha = defaultdict(dict)
    if spath.exists():
        for r in json.load(open(spath))["rows"]:
            if r.get("mean_lift") is not None:
                lift_by_alpha[r["alpha"]][r["layer"]] = r["mean_lift"]
    if not stab:
        print(f"WARN: no bootstrap json at {bpath}")
    if not lift_by_alpha:
        print(f"WARN: no selfsteer json at {spath}")

    sel_lift = lift_by_alpha.get(args.select_alpha, {})
    if not sel_lift and lift_by_alpha:
        # fall back to the closest available alpha
        closest = min(lift_by_alpha, key=lambda a: abs(a - args.select_alpha))
        print(f"NOTE: alpha={args.select_alpha} not in data; using closest alpha={closest}")
        sel_lift = lift_by_alpha[closest]

    layers = sorted(set(stab) | set(sel_lift))
    ns = norm01({L: stab[L] for L in layers if L in stab})
    nl = norm01({L: sel_lift[L] for L in layers if L in sel_lift})

    print(f"\nselection alpha = {args.select_alpha}")
    print(f"{'layer':>5} {'stability':>10} {'self-lift':>10} {'score':>8}")
    combined = {}
    for L in layers:
        parts = [x for x in (ns.get(L), nl.get(L)) if x is not None]
        score = float(np.mean(parts)) if parts else float("nan")
        combined[L] = score
        print(f"{L:5d} {stab.get(L, float('nan')):10.4f} {sel_lift.get(L, float('nan')):+10.3f} {score:8.3f}")

    valid = {L: s for L, s in combined.items() if not np.isnan(s)}
    pick = max(valid, key=valid.get) if valid else None
    if pick is not None:
        print(f"\n>>> selected layer: {pick}  (stability={stab.get(pick)}, "
              f"self-lift@α{args.select_alpha:g}={sel_lift.get(pick):+.3f})")

    json.dump({"model": args.model_name, "select_alpha": args.select_alpha,
               "stability": stab, "self_lift": {a: dict(d) for a, d in lift_by_alpha.items()},
               "combined_score": combined, "selected_layer": pick},
              open(out_dir / f"selected_{args.model_name}.json", "w"), indent=2)

    if layers:
        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        ax2 = ax1.twinx()
        cmap = plt.cm.viridis(np.linspace(0, 0.85, len(lift_by_alpha)))
        for c, a in zip(cmap, sorted(lift_by_alpha)):
            d = lift_by_alpha[a]
            Ls = sorted(d)
            lw = 3 if a == args.select_alpha else 1.6
            ax1.plot(Ls, [d[L] for L in Ls], "o-", color=c, lw=lw,
                     label=f"lift α={a:g}" + (" (select)" if a == args.select_alpha else ""))
        if stab:
            Lb = sorted(stab)
            ax2.plot(Lb, [stab[L] for L in Lb], "s--", color="#d62728", lw=2, label="stability")
        if pick is not None:
            ax1.axvline(pick, color="#2ca02c", ls=":", lw=1.5)
            ax1.annotate(f"selected L{pick}", (pick, ax1.get_ylim()[1]), color="#2ca02c",
                         fontsize=9, fontweight="bold", ha="center", va="top")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("self-steer trait-lift over baseline")
        ax2.set_ylabel("bootstrap stability (pairwise cos)", color="#d62728")
        ax1.set_title(f"Layer selection ({args.model_name}) — paper's two criteria (§A.2), α sweep")
        ax1.grid(alpha=0.25, ls=":")
        ax1.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"layer_select_{args.model_name}.png", dpi=170)
        plt.close(fig)
        print(f"wrote {out_dir / f'layer_select_{args.model_name}.png'}")


if __name__ == "__main__":
    main()
