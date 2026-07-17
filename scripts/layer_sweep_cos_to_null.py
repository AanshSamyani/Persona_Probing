#!/usr/bin/env python3
"""Sweep layers: per-layer cos(v_persona, v_null) for CAA vectors, to pick the
extraction layer.

The right layer is NOT simply the one with the lowest persona cos-to-null (that
just keeps dropping with depth as trait vectors invert / break down). It is the
layer where personas have spread away from null WHILE the nonsense control stays
high — i.e. the layer that maximises the separation

    gap = mean_nonsense_cos - mean_persona_cos

subject to personas not yet sign-flipping (min persona cos not strongly negative).

Reference (paper's published Gemma-2-27B-IT CAA vectors), for comparison:
    Gemma has 45 layers; the paper extracts at layer 22 (~0.49 depth), where
    persona mean=0.706, min=-0.096, nonsense=0.956, gap=+0.25. Gemma's cos-to-null
    slides monotonically with depth just like Qwen — there is no plateau; ~half
    depth is the operating point (spread without sign-flip collapse).

Usage:
    python scripts/layer_sweep_cos_to_null.py --vectors-dir outputs/Qwen2.5-7B-Instruct/caa_vectors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
          "confidence", "deference", "warmth", "impulsivity"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer cos-to-null (persona + nonsense) for CAA vectors")
    ap.add_argument("--vectors-dir", required=True)
    ap.add_argument("--min-floor", type=float, default=-0.15,
                    help="reject a layer as 'collapsed' if min persona cos < this (default -0.15)")
    ap.add_argument("--save-json", default=None,
                    help="also dump the per-layer table as JSON (for plot_layer_comparison.py)")
    ap.add_argument("--model-name", default="model", help="label stored in the JSON")
    args = ap.parse_args()
    vd = Path(args.vectors_dir)

    def load(slug, trait):
        f = vd / f"{slug}_{trait}.pt"
        if not f.exists():
            return None
        return torch.load(f, map_location="cpu", weights_only=True)["vector"].float().numpy()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    ref = load("null", "honesty")
    if ref is None:
        raise SystemExit(f"No vectors found under {vd} (expected null_honesty.pt).")
    n_layers = ref.shape[0]

    print(f"{n_layers} layers.  gap = nonsense - persona (want large, with min not < {args.min_floor})")
    print(f"{'layer':>5} {'pers_mean':>9} {'pers_min':>8} {'nonsense':>9} {'gap':>7}")
    best = (-1.0, None)  # (gap, layer) among non-collapsed layers
    rows = []
    for L in range(n_layers):
        pvals, nvals = [], []
        for t in TRAITS:
            vn = load("null", t)
            if vn is None:
                continue
            for p in PERSONAS:
                vp = load(p, t)
                if vp is not None:
                    pvals.append(cos(vp[L], vn[L]))
            vnon = load("nonsense", t)
            if vnon is not None:
                nvals.append(cos(vnon[L], vn[L]))
        pm, mn = float(np.mean(pvals)), float(np.min(pvals))
        nm = float(np.mean(nvals)) if nvals else float("nan")
        gap = nm - pm
        flag = ""
        if not np.isnan(gap) and mn >= args.min_floor and gap > best[0]:
            best = (gap, L); flag = ""
        if mn < args.min_floor:
            flag = "  (collapsed: sign-flips)"
        print(f"{L:5d} {pm:9.3f} {mn:8.3f} {nm:9.3f} {gap:+7.3f}{flag}")
        rows.append({"layer": L, "depth": round(L / (n_layers - 1), 3),
                     "persona_mean": round(pm, 4), "persona_min": round(mn, 4),
                     "nonsense": None if np.isnan(nm) else round(nm, 4),
                     "gap": None if np.isnan(gap) else round(gap, 4)})

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": args.model_name, "n_layers": n_layers, "rows": rows},
                  open(args.save_json, "w"), indent=2)
        print(f"\nwrote {args.save_json}")

    if best[1] is not None:
        print(f"\n>>> best usable layer: {best[1]}  (gap {best[0]:+.3f}, min persona cos >= {args.min_floor})"
              f"\n    -> rerun fig1_cos_to_null.py --layer {best[1]}")
    else:
        print("\n>>> no layer cleared the min-floor; loosen --min-floor or inspect the table above.")


if __name__ == "__main__":
    main()
