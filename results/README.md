# Results (tracked)

This directory holds the **small, shareable outputs** of the replication runs —
the things that belong on GitHub. Everything heavy (model weights, per-question
activations, per-persona vectors) stays under `outputs/`, which is gitignored and
never pushed.

## `olmo_trajectory/` — Appendix F / Figure 19 (OLMo-2 training trajectory)

Produced on the server by `scripts/run_olmo_trajectory.sh`:

| File | What it is |
|---|---|
| `SUMMARY.md` | Headline tables: mean shared-variance ratio per stage (Fig 19) and transfer-matrix distances between stages (Table 6) |
| `variance_trajectory.json` | Per-stage, per-trait shared-variance ratio ρ (the raw numbers behind Fig 19) |
| `transfer_matrix_distances.json` | Frobenius + Spearman between every pair of stages (Table 6) |
| `vector_alignment.json`, `subspace_overlap.json`, `cluster_stability.json` | Supporting cross-stage geometry metrics |
| `transfer_*.npy` | The 10×10 persona transfer matrix at each stage |
| `trajectory_meta.json` | Layer, persona list, trait list, stage order used |
| `layer_sweep.csv` | Mean shared variance per candidate layer (used to pick the analysis layer) |
| `variance_by_persona.json` | Per-stage, per-persona shared-variance ratio `cos²(v_{c,T}, û_T)` and its mean over traits |
| `figures/fig_variance_trajectory.png` | Paper Figure 19: one line per **trait**, shared variance ρ across stages |
| `figures/fig_variance_by_persona.png` | Companion view: one line per **persona**, averaged over traits (`scripts/variance_by_persona.py`) |
| `figures/*.png` | The other trajectory plots (transfer matrices, alignment, subspace overlap, summary) |
