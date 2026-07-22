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
| FORGET-PRECISION (negation / value-update) | 1.00 / 1.00 after supersession (both 0.00 without) | forget_precision_result.json | 6 sets |
| SUPERSESSION-FALSE-POSITIVE (enumerated) | 0.00 (6/6 coexisting facts survive; was 5/6 deleted before the positional fix); true-update intact | supersession_fp_result.json | n=6 |
| OPERATIONAL-CONTINUITY (idempotent resume) | duplicate-rate decay-ON tracks budget-floor (0.70/0.50/0.00 at k=3/5/10) vs decay-OFF 1.00 at every k; n_old=200 | operational_continuity_result.json | 6 seeds |
| TEMPORAL-AS-OF (bi-temporal, reversed ingest) | now-accuracy 1.00 by valid_from (vs 0.00 by ingest-order); as-of accuracy 1.00 | temporal_asof_result.json | 20 topics x 6 seeds |
| ABSTENTION @ floor 0.6 | abstention-precision 1.00 + in-store recall 1.00 (vs abstention 0.00 at floor 0) | abstention_result.json | 5 seeds |
| CROSS-SCOPE LEAKAGE | scoped 0.00 leakage + 1.00 in-scope recall (vs 0.80 leakage with no scope) | scope_leakage_result.json | 5 seeds |
| SIGNAL-RELIABILITY break-even | recall-lift crosses 0 at the floor 1/(1+D); cal 'full' gain +0.33 / backfire -0.47; 'boost' +0.00; 'gated' +0.19 | ramr_breakeven_result.json | 3 sets |

| COMPRESSION-LIFT (compiled - raw) | +0.00 / -0.40 / -0.55 at K=5/20/50 distractors (compaction is a COST, not a gain, for a capable reader) | compression_oracle_result.json | N=20 |

| **INTEGRITY-RECALL revert** (acc@1: correct CURRENT value after a revert) | **inspeximus 1.00 CI[1,1] · cosine-recency 0.00 · naive 0.55** | ramr_integrity_recall_result.json | 100 |
| **INTEGRITY-RECALL poison** (acc@1: a consumer that branches on a legible warrant tier returns the *corroborated* record over an *uncorroborated* one — **NOT injection detection**; see note) | **inspeximus+warrant 1.00 CI[1,1] · plain inspeximus/recency/naive 0.00** | ramr_integrity_recall_result.json | 100 |
| INTEGRITY-RECALL supersession (acc@1: correct value after an update) | inspeximus 1.00 = cosine-recency 1.00 (a TIE) · naive 0.51 | ramr_integrity_recall_result.json | 100 |

All flagship numbers are internally consistent (each headline equals the mean recomputed from the stored arrays).

### INTEGRITY-CONDITIONED RECALL — honest reading (do not overclaim)
- **revert is the unique win**: only a system with an explicit revert-by-key returns the restored value; a
  recency heuristic returns the retracted one (0.00), naive cosine is a coin-flip among candidates (0.55).
- **poison is a warrant-channel demonstration, NOT injection detection.** Default inspeximus recall scores
  0.00 like everyone else; the *inspeximus+warrant* 1.00 shows the value of a legible warrant tier GIVEN a
  trustworthy warrant — at construction the harness credits the truth with an exogenous `warrant="external"`
  and the poison with none, on distinct keys. It does **not** hold if the attacker can supply the warrant (a
  warrant string is spoofable, per the core's own docs — the same attacker who can inject can attach it).
  Baselines have no warrant channel by design. Unmeasured, and each could flip the story: warranted poison,
  and false-rejection of a legitimate correction that arrives unwarranted. Prior art at this axis: MINJA and
  AgentPoison (memory-injection attacks); AGM belief revision / truth maintenance (the revert operation).
- **supersession is a TIE** with a fair recency baseline (both 1.00); the separation only appears once a
  revert or an injection enters.
- Scope: synthetic controlled scenarios, n=100/scenario, bootstrap 95% CI, same nomic embeddings for every
  system; baselines are plain/recency cosine (what most RAG uses), NOT mem0/Zep. Regenerate:
  `python ramr_integrity_recall.py` (needs a local Ollama serving nomic-embed-text).

### Re-vendor note (v0.4.3)
The vendored `inspeximus` core was updated from the pinned v0.6.10 snapshot to v1.29.0 (matching
`pip install inspeximus`). Every inspeximus-using harness was re-run: all verdicts hold, OUTCOME-LIFT is
unchanged (0.000 drift at every D), and the breakeven cited numbers hold. Two secondary numbers refreshed
against the newer core: CROSS-SCOPE LEAKAGE baseline 0.82 -> 0.80, and the breakeven 'boost' arm shifted by
<=0.03 (its cited headline stays +0.00). Surfaced rather than pinned.

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
