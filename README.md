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

## What it measures (11 metrics)

| Metric | Question | How |
|---|---|---|
| **CONVERSION** | Does complete retrieval convert to a correct multi-hop answer? | gold-chain accuracy |
| **CHAIN-FRAGILITY** | How much does ONE missing hop cost? | gold − partial (one hop dropped) |
| **DISTRACTION** | How much do irrelevant/look-alike facts cost? | gold − noisy |
| **FACT-RETENTION** | Does a compiled/summarized memory tier drop facts under a fixed budget? | raw − compiled, at a hard char budget |
| **OUTCOME-RANKED-RECALL** | Does ranking recall by *was-it-right* beat *was-it-recalled*? | outcome-credit vs relevance-only, vs an independent retriever |
| **FORGET-PRECISION** | After a fact is updated, does recall return the CURRENT value or the STALE one? | fraction returning the current fact after a supersession pass |
| **ECHO-RESISTANCE** | After a correction, if the OLD value is re-stated (verbatim or reworded), does the store keep the corrected value or resurrect the stale one? | fraction whose recall top-1 is the current value AFTER a value-preserving restatement of the retired value |
| **COMPRESSION-vs-RAW** | Does a compiled summary beat the raw (noisy) context, or only lose to it? | acc(compiled) − acc(raw), swept over distractor load |
| **OPERATIONAL-CONTINUITY** | On resume after compaction, does the agent re-execute an already-completed action (a duplicate side-effect)? | duplicate-rate of a budget-limited resume recall, with vs without recency, against accumulated history |
| **TEMPORAL-AS-OF** | When a stale fact arrives LATER than the current one (out-of-order ingest), does supersession resolve by validity-time, not ingest-order? | recall-now accuracy under reversed ingest + recall(as_of=T) returns the value valid at T |
| **INTEGRITY-CONDITIONED RECALL** | After a supersession / revert / poison, does recall return the CORRECT CURRENT value, where a plain cosine store returns the stale or injected one? | acc@1 for naive-cosine, cosine-recency, and inspeximus (± warrant gate), over randomized trials with CIs |

---

## Cross-system integrity + erasure (run against real stores)

The 11 metrics above are the **synthetic, contamination-resistant** core. The [`integrity/`](integrity/) module
is the complementary **cross-system, real-store** cut — the same reliability question asked against the memory
libraries developers actually run, through one shared, ground-truth-blind judge (no home-field instrument).
These cross-system results are rendered as a standing, PR-submittable
**[Agent-Memory Integrity Leaderboard](https://dancenitra.github.io/agora/public/leaderboard/)**
(add your system via [`integrity/SUBMISSION.md`](integrity/SUBMISSION.md)):

- **Value-obscuring REVERT** — the user says *"go back to what we had"* naming no value. Can the store undo a
  correction on that unmarked command? (New here — not one of the synthetic metrics above; mem0/Graphiti have no
  revert operation, so it is a capability gap, not a tuning gap.)
- **ECHO resurrection** — the synthetic ECHO-RESISTANCE metric above, measured cross-system on native configs.
- **Erasure self-check** ([`integrity/erasure_selfcheck.py`](integrity/erasure_selfcheck.py)) — a *run-your-own*
  tool: point it at your installed backend(s); it stores a marker, calls that backend's OWN `delete` + compaction,
  reads the raw store, and reports logical residue. Makes **no vendor claim** — the result is yours. Honest scope
  printed every run (logical vs at-rest residue; audit-log-by-design; coordinated disclosure).
- **Bi-temporal** ([`integrity/temporal_cell.py`](integrity/temporal_cell.py)) — deterministic (unique-token
  ground truth, no judge): reversed-ingest now-accuracy, `as_of(valid-time)` point queries, and a transaction-time
  back-fill (a later correction must not leak into the earlier belief). **This is a parity-with-leaders cell, not
  an inspeximus win** — bi-temporal modelling is the *documented design* of the graph-memory leaders **Zep**
  ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), `t_valid/t_invalid` vs `t′created/t′expired`) and
  **Graphiti** ([getzep/graphiti](https://github.com/getzep/graphiti), `valid_at/invalid_at` + `created_at`), which
  are **not run here** (they need a live Neo4j + LLM pipeline) and are listed, not scored. Measured this cycle:
  **inspeximus 4/4**; **mem0** (default vector store) has no valid-time channel (reversed-ingest returns the stale
  value; no `as_of`). Result: inspeximus *matches* the bi-temporal leaders and *leads* plain vector stores — it does
  not beat Zep/Graphiti on this axis, and we don't claim it does.

```bash
pip install inspeximus
python integrity/run.py                      # revert + echo, local free Ollama judge
python integrity/erasure_selfcheck.py        # your stack's erasure receipt
python integrity/temporal_cell.py            # bi-temporal cell -> results/temporal.json
```

Method, the fairness fix that dropped inspeximus's revert headline from a flattering 1.00 to 0.75, and how to add
your system: [`integrity/METHODOLOGY.md`](integrity/METHODOLOGY.md) · [`integrity/SUBMISSION.md`](integrity/SUBMISSION.md).

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
    inspeximus-NONE (gap 0.000 at every D), confirming NONE is a faithful standard retriever, not a strawman;
    outcome-ranked beats this independent baseline by **+0.469 at D=8, CI [+0.438, +0.500]**.
- **FORGET-PRECISION: a memory layer's ability to forget is only as good as its update detector.** After a fact is
  superseded, does recall return the current value? With the supersession pass, forget-precision is **1.00** for an
  explicit contradiction ("X holds" → "X *not* holds") and **1.00** for a silent numeric value-update ("…is 5" →
  "…is 12") after the update detector is in place (see Changelog) — both up from 0.00 without supersession (the
  stale, higher-value fact otherwise wins 100%). n=30 topics, 6 seeds. The detector is **two-sided**: it must
  return the current value *without* deleting coexisting records — see SUPERSESSION-FALSE-POSITIVE
  (`ramr_supersession_fp.py`), 0.00 false-positive on a 6-item enumerated store after the v0.1.7 fix.

- **ECHO-RESISTANCE: a correction that sticks can still be undone by simply re-stating the old value.**
  FORGET-PRECISION shows the correction holds (1.00). But when the retired value is re-asserted afterwards — a
  benign restatement or an attacker re-injecting it — a last-writer-wins / validity-recency store treats the echo as
  the newest assertion and **resurrects the stale value: echo-resistance 0.00** (verbatim AND reworded, n=30). A
  superseded-object ledger (`inspeximus` `echo_guard`, object-keyed) refuses to let an already-retired value be revived
  by a mere restatement → **echo-resistance 1.00**, with FORGET-PRECISION unchanged (the correction still sticks). A
  *genuine* reversal back to the old value needs an explicit `reaffirm` signal. Honest scope: this tests
  **value-preserving** restatements (the value token is present) — a value-*obscuring* / coreferent echo ("go back to
  the old one") carries no value to key on and is out of scope for any object-level defense. `ramr_echo_resistance.py`.
  The adversarial post-correction restatement is unmeasured in prior benchmarks (STALE / LongMemEval run a single
  correction, no re-injection).

  **Cross-backend, ANSWER-LEVEL (the fair comparison).** A top-1 retrieval metric is fair to a supersession store
  (which *removes* the stale value) but a strawman for an *add-based* store like **mem0**, whose design keeps both
  values and reconciles at read time (top-k handed to an answering LLM) — so we also score echo-resistance at the
  answer level: recall top-k → judge LLM → is the CURRENT value returned? Fair to both designs. Measured (n=30,
  judge = a small instruct model, `ramr_echo_resistance_backends.py`):

  | backend | forget-precision | echo-resistance |
  |---|---|---|
  | inspeximus (echo_guard off) | 1.00 | 0.00 |
  | **mem0 2.0.11** (add-based, real system) | 0.87 | **0.53** (95% CI 0.37–0.70) |
  | **Zep/Graphiti** (Neo4j + OpenAI, real runtime) | 0.87 | **0.87** — echo-attributable **1.00** † |
  | inspeximus (echo_guard on) | 1.00 | **1.00** |

  † Graphiti's raw 0.87 is not an echo failure. Disaggregating pre-echo vs post-echo per case (n=30): in the
  **26/30 cases where the correction actually registered, the echo flipped exactly 0 of them** (echo-attributable
  resurrection 0/26 = 0.00). The residual 13% is 4 cases where Graphiti's LLM extraction never wrote the correction
  at all (pre-echo already stale) — an upstream extraction miss, unrelated to the attack. So Graphiti's bi-temporal
  invalidation **fully defends** against the echo; unlike mem0's 0.53, whose resurrection is echo-driven (14/30 flips
  ≫ its 4/30 pre-echo misses).

  At the answer level (recall top-k → judge LLM → is the current value returned?), under a value-preserving reworded
  echo **real mem0, run in its own recommended config (gpt-4o-mini + text-embedding-3-small + Chroma), resurrects
  the retired value ~47% of the time** — echo-resistance 0.53, 95% CI on resurrection [0.30, 0.63], n=30. It
  reproduces an all-local Ollama-config run (0.57), so the effect is config-robust, not an artifact of one judge.
  This isn't a "mem0 bug": an add-based store keeps both values and reconciles at read time, and the reader
  sometimes returns the retired one (its extractor writes a "reverted back to <old>" record — observed in inspected
  cases). inspeximus's default keyed supersession is fully vulnerable (0.00, store-deterministic); `echo_guard` closes it
  (1.00). Honest scope: a small synthetic probe (n=30), value-preserving echoes, single judge — a demonstration, not
  a definitive benchmark; report the CI, not a point estimate. Add a backend via a 3-method adapter to benchmark
  your own store.

  **Zep/Graphiti — measured at runtime, and it defends.** We ran Graphiti end-to-end (real `graphiti_core`,
  Neo4j backend, OpenAI `gpt-4o-mini` + `text-embedding-3-small` for extraction/embedding, n=30, same
  correction→echo→recall→judge protocol). Confirming the code-path reading: when the correction registers,
  the echo resurrects the stale value **0 out of 26 times** — Graphiti's bi-temporal invalidation surfaces the
  already-invalidated stale edge as a dedup candidate (`get_between_nodes` has no validity filter), so a verbatim
  echo folds onto it via the `_normalize_string_exact` fast-path and a reworded one is handed to the resolver LLM
  with the stale edge present to dedup against. The only staleness (4/30) is Graphiti's extraction pipeline never
  writing the correction in the first place (a `Target entity not found` extraction miss), which the echo neither
  causes nor exploits. **Takeaway: this is not "inspeximus defends, Graphiti doesn't" — a real bi-temporal store and an
  object-keyed ledger both defend, structurally.** inspeximus's edge is being a single zero-dependency file with no graph
  DB or LLM-extraction step (which is exactly where Graphiti's 13% leaks), and the still-open frontier both share:
  a value-*obscuring* echo ("go back to the old one") carries no value to invalidate against and defeats object-level
  and edge-level defenses alike. `graphiti_echo_run.py`.

- **OPERATIONAL-CONTINUITY: recency weighting is necessary AND sufficient for idempotent resume.** On resume, an
  agent must skip already-completed actions; a missed "done" record → a duplicate side-effect. With recency (recent
  completions out-rank old ones), the duplicate-rate tracks the recall-budget floor `max(0, C−k)/C` exactly — robust
  to 200 accumulated old-session completions (e.g. 0.00 at budget k=10 ≥ C=10). WITHOUT recency, current completions
  are indistinguishable from history → duplicate-rate stays **1.00 at every budget** (even k=50): the agent re-runs
  everything. So recency isn't just for salience — it's what keeps current operational state recoverable as history
  grows. (`ramr_operational_continuity.py`, pure-recall proxy, 6 seeds. Metric proposed by @safal207 in
  claude-code#34556; this is a first cut — fixture input welcome.)

- **TEMPORAL-AS-OF: supersession must resolve by validity-time, not ingest-order.** When a stale fact about an
  earlier state arrives LATER than the current one (back-fill / replayed log / multi-source merge), an ingest-order
  "last-write-wins" rule keeps the stale record (it has the later `ts`) — `now_accuracy` 0.00 by construction.
  Resolving by `valid_from` (when the fact is TRUE) instead serves the **current** value (now_accuracy **1.00**) and
  `recall(as_of=T)` returns the value that was valid at T (as_of_accuracy **1.00**). The reference engine now carries
  `valid_from` / `invalidated_at` (defaulting to ingest-time, so ordered streams are unchanged).
  (`ramr_temporal_asof.py`, deterministic, 20 topics × 6 seeds.)

See `VERIFIED_NUMBERS.md` for the full ledger (each headline recomputed from its source arrays).

## Changelog

- **v0.4.3** — **INTEGRITY-CONDITIONED RECALL** metric (`ramr_integrity_recall.py`): after a supersession /
  revert / poison, does recall return the correct current value? On this constructed scenario, `revert` is
  the unique win (inspeximus 1.00 vs cosine-recency 0.00, naive 0.55 — recency has no revert operation);
  `supersession` ties a fair recency baseline (both 1.00). The `poison` row (inspeximus+warrant 1.00 vs
  everyone-else 0.00) is a **warrant-channel demonstration, not injection detection** — it lets a consumer
  branch on an externally-assigned trust label; it does not detect the injection and does not hold if the
  attacker can supply the warrant (spoofable, per the core's docs). Warranted-poison and false-rejection of
  unwarranted legitimate corrections are unmeasured. Prior art: MINJA / AgentPoison, AGM belief revision. n=100/scenario, bootstrap CIs, raw
  per-trial arrays persisted so `verify_numbers.py` recomputes each acc@1. Also **re-vendored the `inspeximus`
  core to v1.29.0** (matching `pip install inspeximus`, replacing the pinned v0.6.10 snapshot): every
  inspeximus-using harness was re-run — all verdicts hold and OUTCOME-LIFT is unchanged; two secondary numbers
  refreshed against the newer core (CROSS-SCOPE-LEAKAGE baseline 0.82→0.80; breakeven 'boost' arm ≤0.03, its
  cited headline stays +0.00).
- **v0.4.2** — renamed the vendored memory library to `inspeximus` throughout (module, class, and every
  reference), aligning the repo with the maintained package name. Rename-only: every number reproduces
  against the unchanged vendored code.
- **v0.4.1** — Folklore Meter fix: `ask_clean`/`run_claim` now honour a claim's custom **`extractor`** (it was read but ignored — `extract_int` was hardcoded), so the meter works for string / abstention / multiple-choice claims, not just integer answers. Unit-tested with a string extractor.
- **v0.4.0** — **Folklore Meter** (`ramr_folklore_meter.py`): a reusable tool that asks, of any AI-engineering
  "folklore" mechanism, *does it actually help or is it a weak-model crutch?* It measures a claim the
  contamination-resistant RAMR way — a CONTRARIAN, **judge-free exact-match** task, run across a **capability
  gradient** (local Ollama models of increasing size + a pluggable frontier anchor) — and issues a verdict:
  **REAL** (advantage persists at the frontier) / **WEAK-MODEL ARTIFACT** (helps weak models, ~0 at the frontier) /
  **REGIME-SPECIFIC** / **NULL**. Robust answer extraction (only the `ANSWER:` line, retry-until-clean — no
  last-number fallback, which silently confounds messier conditions). Two worked verdicts from this method
  (2026-06-24): **decision-trace / "why"-memory → WEAK-MODEL ARTIFACT** (storing the rationale helps sub-frontier
  models +0.1–0.3 but adds +0.00 for a frontier model, which re-derives it from the bare outcome); **multi-agent
  vote-ensemble → REGIME-SPECIFIC** (an inverted-U in single-shot reliability: helps only sub-reliable models, +0.00
  once a model solves the task reliably alone). Recurring lesson: *memory/agent-mechanism sophistication tends to be
  a weak-model crutch — presence of the relevant fact matters, not the mechanism's cleverness.*
- **v0.3.1** — **RESUME verdict semantics pinned** (with [safal207/LS#654](https://github.com/safal207/LS/issues/654),
  which verified the v0.3.0 canonical sources and aligned LS to the mapping). In the shared `ramr-ls-evidence-v0.1`
  standard, **`RESUME` means the tested continuation invariant passed — NOT global execution authorization**: it does
  not bypass downstream policy, approval, or effect gates. Semantic clarification only — fixture bytes and sha256
  digests are unchanged, so existing pins (incl. LS#654) stay valid. The envelope is now aligned end-to-end; any
  future semantic change is a new envelope version.
- **v0.3.0** — **RAMR↔LS fixture set now spans all four continuation verdicts.** Added 3 more canonical
  `ramr-ls-evidence-v0.1` evidence fixtures (each frozen + sha256 digest for LS to pin):
  `superseded_approval` → REJECT, `incomplete_dependency_chain` → ABSTAIN, `target_state_drift` → REVALIDATE
  (with `duplicate_successful_outcome` → REJECT from v0.2.0). Plus `run_ramr_ls_fixtures.py`, a deterministic
  conformance runner that scores each fixture's RAMR-side measured quantity (recovered_side_effect /
  recovered_current_approval / full_chain_recovered / target_current) against the frozen `expected` — all PASS.
  Boundary stays: RAMR measures retrieval reliability, LS owns the verdict; *a retrieval miss is a reliability
  failure, not execution permission*. `superseded_approval` + `target_state_drift` ride inspeximus's bi-temporal
  `valid_from`/`invalidated_at`; `incomplete_dependency_chain` rides CHAIN-FRAGILITY.
- **v0.2.0** — **RAMR↔LS interoperability** (collaboration with [safal207/LS](https://github.com/safal207/LS),
  [anthropics/claude-code#34556](https://github.com/anthropics/claude-code/issues/34556)). RAMR hosts the canonical
  `ramr-ls-evidence-v0.1` evidence fixture (`fixtures/ramr_ls/duplicate_successful_outcome.json`, frozen + sha256
  digest) and a reliability-layer reference harness (`ramr_ls_evidence.py`) that emits the envelope from a memory
  store as a thin projection of native fields (bi-temporal `valid_from`/`invalidated_at`, `provenance`, Beta
  `reliability_signal`, recall `budget`) and scores `recovered_side_effect`. Boundary: **RAMR measures retrieval
  reliability; LS evaluates the deterministic continuation verdict.** Invariant: *a retrieval miss is a reliability
  failure, not execution permission* — so a duplicate completed side effect is REJECTed whether or not the record
  was recovered. Also lands two `inspeximus` upgrades used by the envelope (both regression-gated, no metric change):
  **source-span provenance** (`remember(source=)`, surfaced in `recall()`) and a **poison-propagation guard**
  (episodic→semantic graduation now requires corroboration — provenance, a positive outcome, or a corroborating
  link — so a confabulation can't become durable on recall-frequency alone).
- **v0.1.9** — added a **TEMPORAL-AS-OF** metric (`ramr_temporal_asof.py`) and **bi-temporal validity** in the
  reference core (`inspeximus`): `remember(valid_from=)`, supersession resolves by validity-time not ingest-order, and
  `recall(as_of=T)`. Result: under reversed ingest (stale fact arrives later), the old ingest-order rule serves the
  STALE value (now_accuracy 0.00) while validity-time serves the CURRENT one (1.00) and as-of recall returns the
  historical value (1.00). FORGET-PRECISION + SUPERSESSION-FP + OPERATIONAL-CONTINUITY unchanged (valid_from
  defaults to ingest-time, so ordered streams are byte-identical).
- **v0.1.8** — added an **OPERATIONAL-CONTINUITY** metric (`ramr_operational_continuity.py`) — the idempotent-resume
  property **proposed by @safal207 in [anthropics/claude-code#34556](https://github.com/anthropics/claude-code/issues/34556)**;
  this is a first runnable cut of that idea, and input on the fixtures is welcome. It tests: does the agent
  re-execute a completed action after compaction? Result: recency weighting is
  necessary AND sufficient — with it, duplicate-rate = the recall-budget floor (robust to unlimited history);
  without it, 1.00 at every budget. A new agentic axis beyond fact recall (it measures a duplicate-side-effect cost,
  not fact survival).
- **v0.1.7** — **fixed a supersession false-positive in the reference core (`inspeximus`)** surfaced by a new severe
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
  validating the law on the engine; also characterizes inspeximus's `cal_mode` (full/boost/gated) tradeoff.
- **v0.1.1** — added the **FORGET-PRECISION** metric (`ramr_forget_precision.py`). It surfaced a gap in the
  reference memory core (`inspeximus`): supersession only fired on explicit negation, so a *silent numeric value-update*
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
4. **Independent baselines.** Claims about our own components (inspeximus) are checked against standard, independent
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
