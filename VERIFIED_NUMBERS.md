# RAMR — Verified Numbers Ledger (regenerate via `python verify_numbers.py`)

Standing rule: **verify every measured number against its persisted source before any public citation.** This
ledger ties each citable RAMR number to a result JSON, with the headline RECOMPUTED from the raw arrays (not
trusted from a prose note). Last run: 2026-06-23.

## VERIFIED (traceable to a persisted JSON, headline == recomputed)
| Metric | Value | Source | n |
|---|---|---|---|
| CONVERSION (gold acc) | 1.000 | ramr_v0_result.json | 40 |
| CHAIN-FRAGILITY (gold-partial) | +0.975 | ramr_v0_result.json | 40 |
| contamination (closed-book) | 0.000 (uncontaminated) | ramr_v0_result.json | 40 |
| **CHAIN-FRAGILITY @ n=200, qwen3-coder:30b** | **+1.000 CI [1.000,1.000]** | ramr_scale_cf_result.json | 200 |
| **CHAIN-FRAGILITY @ n=200, glm-5.2** | **+1.000 CI [1.000,1.000]** | ramr_scale_cf_result.json | 200 |
| DISTRACTION@60 (6 models) | +0.15 (llama) .. +0.60 (kimi); qwen +0.35 | ramr_v2c_result.json | 20 |
| OUTCOME-LIFT D=1 | +0.358 CI [+0.299,+0.420] (random-lift -0.30) | outcome_scale_result.json | 12 sets |
| OUTCOME-LIFT D=2 | +0.361 CI [+0.316,+0.406] (random-lift -0.22) | outcome_scale_result.json | 12 sets |
| OUTCOME-LIFT D=4 | +0.469 CI [+0.424,+0.514] (random-lift -0.09) | outcome_scale_result.json | 12 sets |
| OUTCOME-LIFT D=8 | +0.427 CI [+0.385,+0.469] (random-lift -0.07) | outcome_scale_result.json | 12 sets |
| **FACT-RETENTION-LOSS M=48 @400-char, qwen3-coder:30b** | **+0.700 (retention 0.26)** | ramr_factret_result.json runs[] | 5 sets |
| **FACT-RETENTION-LOSS M=48 @400-char, gemma2:9b** | **+0.756 (retention 0.24)** | ramr_factret_result.json runs[] | 5 sets |
| **FACT-RETENTION-LOSS M=48 @400-char, Claude Opus 4.8** | **+0.507 (retention 0.49)** — programmatic packing ceiling, no LLM read-back; NOT method-comparable to the qwen/gemma LLM rows | ramr_factret_result.json runs[] | 3 sets |
| OUTCOME vs INDEPENDENT sklearn baseline @D=8 | +0.469 CI [+0.438,+0.500]; NONE-vs-ext gap 0.000 | ramr_external_baseline_result.json | 4 sets |
| REAL-SYSTEM engines @D=8 (near-dup disambiguation) | keyword FTS5/BM25 0.18, vector 0.09 (chance 0.11); OUTCOME beats best +0.36 CI [+0.29,+0.44] | ramr_real_systems_result.json | 4 sets |
| FORGET-PRECISION (negation / value-update) | 1.00 / 0.97 after supersession (both 0.00 without) | forget_precision_result.json | 6 sets |
| ABSTENTION @ floor 0.6 | abstention-precision 1.00 + in-store recall 1.00 (vs abstention 0.00 at floor 0) | abstention_result.json | 5 seeds |
| CROSS-SCOPE LEAKAGE | scoped 0.00 leakage + 1.00 in-scope recall (vs 0.82 leakage with no scope) | scope_leakage_result.json | 5 seeds |
| SIGNAL-RELIABILITY break-even | recall-lift crosses 0 at the floor 1/(1+D); cal 'full' gain +0.33 / backfire -0.47; 'boost' +0.00; 'gated' +0.19 | ramr_breakeven_result.json | 3 sets |

All flagship numbers are internally consistent (each headline equals the mean recomputed from the stored arrays).

## Previously UNBACKED — now RESOLVED (2026-06-23)
- **FACT-RETENTION cross-model hard-budget table** is now VERIFIED (rows above). Fix applied: `ramr_factret.py`
  now APPENDS per-(model,budget) to a `runs` dict (no overwrite); re-ran qwen3-coder:30b + gemma2:9b (n=5) at
  `RAMR_CBUDGET=400`. The slow/contaminated glm-5.2 cloud run (it returned empty summaries scored as 0 — a
  measurement failure now retried+excluded by the harness) was **replaced by a Claude Opus 4.8 datapoint** (dense
  structured packing, programmatic retention scoring — a separate measurement, not method-comparable to the
  LLM-round-trip rows). Finding holds: compaction is lossy for every model under a fixed budget.

## Note
Every cited number is recomputed from its source JSON by `verify_numbers.py`; a number with no row here pointing at
a persisted file is not cited.
