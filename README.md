# RAMR — Retrieval-Augmented Memory Reliability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20818291.svg)](https://doi.org/10.5281/zenodo.20818291)
&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **contamination-resistant synthetic probe** for agentic-RAG / memory systems, plus the findings it produced.

**Cite:** Agora (2026). *RAMR — Retrieval-Augmented Memory Reliability*. https://doi.org/10.5281/zenodo.20818291 (concept DOI, always latest)

> **What this is — and is not.** RAMR v0.1 is a *findings + method* release: a small, reproducible, synthetic
> benchmark that isolates specific failure modes of retrieval-backed memory, and the measurements we got from it.
> It is **not** (yet) a definitive, large-scale, multi-system leaderboard. We lead with the limitations below on
> purpose — every number here is traceable to a persisted source file, and we mark exactly which results are
> statistically firm versus directional.

---

## Limitations (read these first)

- **Synthetic, not real-world.** Items are generated from random tokens (this is a feature — see "contamination
  resistance" — but it means we do **not** measure real-document retrieval or real-conversation memory yet).
- **Scale.** Flagship metrics (CHAIN-FRAGILITY) are measured at n=200 with tight CIs; OUTCOME-RANKED-RECALL at
  n=12 sets; FACT-RETENTION at n=5 sets. CIs are reported throughout; treat small-n magnitudes as directional and
  the orderings as the signal.
- **OUTCOME-RANKED uses one embedder** (local `nomic-embed-text`) and is validated against an *independent*
  standard retriever (scikit-learn cosine) — but not yet against shipped memory products (mem0/Zep/etc.).
- **Answer matching is substring-based** on short synthetic answers; it is exact here because answers are unique
  tokens, but this would be noisy on free-form text.
- **Single covariate construction per metric.** We do not claim these magnitudes transfer unchanged to other task
  shapes; we claim the *relative* effects are robust where the CI says so.

---

## What it measures (8 metrics)

| Metric | Question | How |
|---|---|---|
| **CONVERSION** | Does complete retrieval convert to a correct multi-hop answer? | gold-chain accuracy |
| **CHAIN-FRAGILITY** | How much does ONE missing hop cost? | gold − partial (one hop dropped) |
| **DISTRACTION** | How much do irrelevant/look-alike facts cost? | gold − noisy |
| **FACT-RETENTION** | Does a compiled/summarized memory tier drop facts under a fixed budget? | raw − compiled, at a hard char budget |
| **OUTCOME-RANKED-RECALL** | Does ranking recall by *was-it-right* beat *was-it-recalled*? | outcome-credit vs relevance-only, vs an independent retriever |
| **FORGET-PRECISION** | After a fact is updated, does recall return the CURRENT value or the STALE one? | fraction returning the current fact after a supersession pass |
| **COMPRESSION-vs-RAW** | Does a compiled summary beat the raw (noisy) context, or only lose to it? | acc(compiled) − acc(raw), swept over distractor load |
| **OPERATIONAL-CONTINUITY** | On resume after compaction, does the agent re-execute an already-completed action (a duplicate side-effect)? | duplicate-rate of a budget-limited resume recall, with vs without recency, against accumulated history |

---

## Key findings

_All numbers below are traceable to a persisted result JSON and recomputed by `verify_numbers.py` (see
`VERIFIED_NUMBERS.md`)._

- **CHAIN-FRAGILITY is near-universal.** Dropping a single required hop collapses 3-hop accuracy to near-zero for
  every model tested — 7 models across 6 families (Qwen, Meta, Google, Zhipu, Moonshot, Anthropic), CHAIN-FRAGILITY
  +0.90 to +1.00. The two anchor models were run at **n=200** with paired-bootstrap CIs: qwen3-coder:30b and
  glm-5.2 both **+1.000, CI [+1.000, +1.000]**. The other five were at smaller n (n=20 cross-family zoo: qwen2.5:7b
  +0.95, llama3.1:8b +0.90, gemma2:9b +1.00, kimi-k2.6 +1.00; Claude/Anthropic n=12 blind subset +1.00). (At the
  n=40 v0 pilot it was +0.975 — one partial answered by chance — tightening to +1.000 at n=200.)
- **CONVERSION / contamination.** Complete-chain accuracy = 1.000; closed-book accuracy = 0.000 (random synthetic
  entities -> the data cannot have been memorized).
- **DISTRACTION is variable and model-specific** (and, unlike chain-fragility, *not* universal). DISTRACTION@60
  ranges from a negligible +0.15 (llama3.1:8b) to a substantial +0.60 drop (kimi-k2.6) at 60 distractors (n=20);
  qwen +0.35. (Lexical near-miss distractors did **not** bite beyond raw volume — recorded as honest negatives.)
- **FACT-RETENTION: compaction is lossy under a fixed budget.** At M=48 facts under a hard 400-char budget, every
  model loses facts (the budget bottleneck is real): retention-loss **+0.70** (qwen3-coder:30b, n=5) and **+0.76**
  (gemma2:9b, n=5), both measured via an LLM compress + LLM read-back round-trip; raw store recovers 0.96–1.00. A
  *separate* programmatic packing-density ceiling (Claude Opus 4.8, dense key=value packing, regex-scored, **no LLM
  read-back**, raw set to 1.0 by construction, n=3) retains 0.49 — a method-different **upper bound on what dense
  packing could preserve**, NOT a like-for-like model comparison.
- **OUTCOME-RANKED-RECALL: ranking recall by *was-it-right* beats *was-it-recalled* on a near-duplicate case that
  relevance can't solve.** Outcome-credit reranking beats relevance-only at every ambiguity level: lift **+0.358 /
  +0.361 / +0.469 / +0.427** at D=1/2/4/8 (**n=12 sets**, hardened from n=4; bootstrap CIs all exclude 0, min lower
  bound +0.299). A random-credit control is *negative* (-0.30 → -0.07), so the gain is the outcome signal, not
  reranking noise. The comparison arms (relevance-only /
  FTS5-BM25 / dense-vector / independent sklearn cosine) are denied the outcome label **by design** — this shows
  the *value* of an outcome/credit channel, it is **not** a head-to-head win over shipped products (mem0/Zep were
  not run).
  - The independent scikit-learn `NearestNeighbors(cosine)` retriever (not our code) scores **identically** to
    mnemo-NONE (gap 0.000 at every D), confirming NONE is a faithful standard retriever, not a strawman;
    outcome-ranked beats this independent baseline by **+0.469 at D=8, CI [+0.438, +0.500]**.
- **FORGET-PRECISION: a memory layer's ability to forget is only as good as its update detector.** After a fact is
  superseded, does recall return the current value? With the supersession pass, forget-precision is **1.00** for an
  explicit contradiction ("X holds" → "X *not* holds") and **1.00** for a silent numeric value-update ("…is 5" →
  "…is 12") after the update detector is in place (see Changelog) — both up from 0.00 without supersession (the
  stale, higher-value fact otherwise wins 100%). n=30 topics, 6 seeds. The detector is **two-sided**: it must
  return the current value *without* deleting coexisting records — see SUPERSESSION-FALSE-POSITIVE
  (`ramr_supersession_fp.py`), 0.00 false-positive on a 6-item enumerated store after the v0.1.7 fix.

- **OPERATIONAL-CONTINUITY: recency weighting is necessary AND sufficient for idempotent resume.** On resume, an
  agent must skip already-completed actions; a missed "done" record → a duplicate side-effect. With recency (recent
  completions out-rank old ones), the duplicate-rate tracks the recall-budget floor `max(0, C−k)/C` exactly — robust
  to 200 accumulated old-session completions (e.g. 0.00 at budget k=10 ≥ C=10). WITHOUT recency, current completions
  are indistinguishable from history → duplicate-rate stays **1.00 at every budget** (even k=50): the agent re-runs
  everything. So recency isn't just for salience — it's what keeps current operational state recoverable as history
  grows. (`ramr_operational_continuity.py`, pure-recall proxy, 6 seeds. Metric proposed by @safal207 in
  claude-code#34556; this is a first cut — fixture input welcome.)

See `VERIFIED_NUMBERS.md` for the full ledger (each headline recomputed from its source arrays).

## Changelog

- **v0.1.8** — added an **OPERATIONAL-CONTINUITY** metric (`ramr_operational_continuity.py`) — the idempotent-resume
  property **proposed by @safal207 in [anthropics/claude-code#34556](https://github.com/anthropics/claude-code/issues/34556)**;
  this is a first runnable cut of that idea, and input on the fixtures is welcome. It tests: does the agent
  re-execute a completed action after compaction? Result: recency weighting is
  necessary AND sufficient — with it, duplicate-rate = the recall-budget floor (robust to unlimited history);
  without it, 1.00 at every budget. A new agentic axis beyond fact recall (it measures a duplicate-side-effect cost,
  not fact survival).
- **v0.1.7** — **fixed a supersession false-positive in the reference core (`mnemo`)** surfaced by a new severe
  test (`ramr_supersession_fp.py`): the numeric value-update detector over-fired on ENUMERATED facts
  (`"step 1 takes 5 min"`, `"step 2 takes 8 min"` strip to the same skeleton) and silently superseded coexisting
  records — a 6-item store collapsed to 1/6 active. Fixed by comparing numbers POSITIONALLY (a value update changes
  exactly one number-slot; multiple changed = distinct enumerated facts) → 6/6 survive. Regression-gated: the
  published FORGET-PRECISION number is unchanged (negation 1.00, value-update 0.97→re-measured 1.00). FORGET-
  PRECISION now reported with its dual — supersession should return the current value *without* deleting coexisting
  ones.
- **v0.1.6** — added a **COMPRESSION-vs-RAW** metric (`ramr_compression.py`): tested the hyped
  'compression beats oracle' claim. For a capable reader (qwen3-coder:30b), a compiled summary does NOT
  beat raw context at any noise level (lift +0.00 / -0.40 / -0.55 at K=5/20/50 distractors) -- compaction
  is a cost (structure/budget), not an accuracy gain. Honest negative; the external claim didn't replicate.
- **v0.1.5** — **hardened OUTCOME-RANKED-RECALL from n=4 to n=12 sets** (`outcome_scale_result.json`):
  lift +0.358/+0.361/+0.469/+0.427 at D=1/2/4/8, every bootstrap CI excludes 0 (min lower bound +0.299),
  random-credit control stays negative. The flagship was-it-right>was-it-recalled claim is not small-n noise.
- **v0.1.4** — added a **CROSS-SCOPE LEAKAGE** metric (`ramr_scope_leakage.py`) and `recall(scope=)`
  isolation in the engine: a shared store with two tenants sharing one schema leaks an A-fact into B's
  recall 79% of the time WITHOUT a scope; with `recall(scope='B')` leakage is 0% and in-scope recall 100%.
- **v0.1.3** — added an **ABSTENTION / false-recall** metric (`ramr_abstention.py`) and a relevance-floor
  in the engine: with `recall(min_relevance=0.6)` it abstains on 100% of out-of-store probes (says 'not in
  memory') while keeping 100% in-store recall, vs confabulating a wrong fact 100% of the time at floor 0.
- **v0.1.2** — added a **signal-reliability break-even** harness (`ramr_breakeven.py`): sweeps credit
  reliability p x ambiguity D and shows recall-lift crosses zero at the no-signal floor 1/(1+D),
  validating the law on the engine; also characterizes mnemo's `cal_mode` (full/boost/gated) tradeoff.
- **v0.1.1** — added the **FORGET-PRECISION** metric (`ramr_forget_precision.py`). It surfaced a gap in the
  reference memory core (`mnemo`): supersession only fired on explicit negation, so a *silent numeric value-update*
  was merged as a duplicate and recall kept serving the stale value (forget-precision 0.00). Fixed by detecting a
  near-duplicate pair that differs only in a numeric value as a state-toggle → forget-precision 0.00 → **0.97**.
  (Finding your own gap with a new metric and fixing it is the point.)
- **v0.1.0** — initial release: 5 metrics, frozen dataset, runner, verification ledger.

---

## Design principles

1. **Contamination-resistant.** Entities are random synthetic tokens, so a model cannot have memorized the
   answers — closed-book accuracy is ~0 by construction (we verify this on every run).
2. **Reproducible.** The dataset is frozen to disk with a sha256-pinned manifest (`data/manifest.json`); a single
   runner loads it rather than regenerating. Embeddings are cached to disk so memory-side runs reproduce without
   re-embedding.
3. **Falsifiable.** Every metric ships with a pre-registered falsifier and bootstrap CIs; we record honest
   negatives (e.g. adversarial lexical distractors did NOT bite) and corrections (we caught our own
   summary-budget confound) rather than hiding them.
4. **Independent baselines.** Claims about our own components (mnemo) are checked against standard, independent
   libraries on identical inputs.

---

## How to run

```bash
# 1. freeze (or refresh) the versioned dataset  -> data/ramr_chains_v0.1.0.jsonl + manifest.json
python build_dataset.py

# 2. score a model on the LLM-reader metrics (loads the frozen dataset; never regenerates)
python ramr_run.py --model qwen3-coder:30b --n 200 --dist 30

# 3. memory-side metrics (local embedder)
python ramr_outcome_ranked.py        # OUTCOME-RANKED-RECALL
python ramr_external_baseline.py      # vs independent sklearn/numpy cosine retriever
python ramr_factret.py                # FACT-RETENTION (RAMR_CBUDGET=400 for the fair info-budget mode)

# 4. verify every cited number against its persisted source
python verify_numbers.py
```

Models are reached via an OpenAI-compatible endpoint (local Ollama by default); cloud model tags also work through
the same route.

## Files

- `build_dataset.py` / `data/` — frozen dataset + manifest (sha256-pinned)
- `ramr_run.py` — single runner / scoring CLI (CONVERSION, CHAIN-FRAGILITY, DISTRACTION)
- `ramr_outcome_ranked.py`, `ramr_external_baseline.py`, `ramr_real_systems_baseline.py`, `ramr_factret.py`,
  `ramr_factret_claude.py`, `ramr_forget_precision.py` — memory-side metrics + baselines
- `ramr_scale_chainfragility.py`, `ramr_v2c_zoo.py` — scale + cross-family runners
- `verify_numbers.py` / `VERIFIED_NUMBERS.md` — number-verification audit + ledger

## What we do NOT claim

We do not claim RAMR is the definitive agentic-memory benchmark, that these magnitudes transfer to real corpora,
or that shipped products underperform it (we have not run them). We claim: a reproducible, contamination-resistant
method; a robust cross-model CHAIN-FRAGILITY result; and a set of honestly-caveated findings about where
retrieval-backed memory fails.
