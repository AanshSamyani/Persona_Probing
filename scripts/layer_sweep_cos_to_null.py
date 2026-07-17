#!/usr/bin/env python3
"""Sweep layers: mean cos(v_persona, v_null) per layer for CAA vectors, to pick the
extraction layer with the strongest persona-conditioning (lowest cos = most spread).

Usage:
    python scripts/layer_sweep_cos_to_null.py --vectors-dir outputs/Qwen2.5-7B-Instruct/caa_vectors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
          "confidence", "deference", "warmth", "impulsivity"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer mean cos-to-null for CAA vectors")
    ap.add_argument("--vectors-dir", required=True)
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

    best = (1.0, None)
    for L in range(n_layers):
        vals = []
        for t in TRAITS:
            vn = load("null", t)
            if vn is None:
                continue
            for p in PERSONAS:
                vp = load(p, t)
                if vp is not None:
                    vals.append(cos(vp[L], vn[L]))
        m, mn = float(np.mean(vals)), float(np.min(vals))
        if m < best[0]:
            best = (m, L)
        print(f"layer {L:2d}: mean persona cos-to-null={m:.3f}  min={mn:.3f}")

    print(f"\n>>> most persona-conditioned layer: {best[1]}  (mean {best[0]:.3f}) "
          f"-> rerun fig1_cos_to_null.py --layer {best[1]}")


if __name__ == "__main__":
    main()
