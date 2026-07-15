# OLMo-2 7B training-trajectory replication

Paper Appendix F — Figure 19 (shared variance ratio) and Table 6 (transfer-matrix distances).

- Analysis layer: **15**
- Personas: 12 — con_artist, drill_sergeant, farmer, kindergarten_teacher, nonsense, null, politician, professor, street_hustler, surgeon, tech_ceo, therapist
- Traits: 8 — assertiveness, confidence, deference, empathy, honesty, impulsivity, risk_taking, warmth

## Shared variance ratio per stage (Figure 19)

| Stage | Mean shared variance |
|---|---|
| Pretrain 1% | 93.7% |
| Pretrain 10% | 96.2% |
| Pretrain 50% | 96.0% |
| Base | 96.5% |
| SFT | 86.2% |
| DPO | 84.9% |
| Instruct | 84.2% |

_Paper targets: 95.7 / 96.5 / 96.0 / 96.8 (pretrain→base) → 85.5 / 84.2 / 83.5 (SFT/DPO/Instruct)._

## Transfer-matrix distances between stages (Table 6)

| Comparison | Frobenius d_F | Spearman rho_s |
|---|---|---|
| Pretrain 1% → Pretrain 10% | 0.41 | 0.49 |
| Pretrain 10% → Pretrain 50% | 0.09 | 0.80 |
| Pretrain 50% → Base | 0.12 | 0.82 |
| Base → SFT | 1.49 | 0.33 |
| Base → Instruct | 1.73 | 0.29 |
| SFT → DPO | 0.18 | 0.99 |
| DPO → Instruct | 0.09 | 1.00 |

