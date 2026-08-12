# ERRATA

Defects found in published RAMR artifacts, what changed, and what each does and does not invalidate.
Newest first. Every entry names the hashes so you can tell which files you hold.

The norm this file exists to meet: a revised public dataset gets a **published diff**, not a silent
in-place edit. We did the silent edit first and are correcting that here.

---

## 2026-08-12 — trace artifacts: two generations, a false manifest pin, and a third shortcut

### 1. `ramr_traces_v0.1.jsonl` and the v0.2 pair came from different generator runs

`ramr_traces_v0.2_blind.jsonl` / `_labels.jsonl` are derived from `ramr_traces_v0.1.jsonl`, but the
v0.1 that shipped was **not** the v0.1 the v0.2 pair was built from. Same 300 queries; **300 of 300
traces carried different `record_id`s**. Joining v0.2 back to v0.1 matched nothing, silently, in every
trace.

Cause: `build_trace()` takes ids from `store.remember()`, which mints `uuid.uuid4().hex[:10]`. Re-running
the export therefore produces an artifact that is semantically identical and byte-wise unjoinable, and
no number moves to signal it.

| file | before | after |
|---|---|---|
| `ramr_traces_v0.1.jsonl` | `e6a8aced18e9c14c…` | `99b5fb61f38f85db…` |
| `ramr_traces_v0.2_blind.jsonl` | `1155c1161d93916d…` | unchanged |
| `ramr_traces_v0.2_labels.jsonl` | `ecb28e9ab040ccb0…` | unchanged |

**Rewritten in place, at the same path, with no version bump** — that was the wrong call and this
entry is the correction. If you took v0.1 before 2026-08-12 you hold `e6a8aced…`.

**Does not invalidate:** anything scored on `v0.2_blind` + `v0.2_labels` alone. Those two are
byte-identical before and after, and self-consistent with each other. The positional invariant
reported by an external reviewer (planted record precedes the record it replaces) reproduces at
**300/300 on both** the old and the new v0.1 — it is a property of the generator, not of the sample.

**May invalidate:** anything that joined v0.2 back to v0.1. Only you can tell whether you did.

**Still open:** the ids are not content-addressed, so the artifact still cannot be regenerated
byte-for-byte. That is the real fix and it is not done.

Guard: `ramr_traces_pair_consistency.py`, in CI on every push. It reads the files on disk rather than a
fresh run — a fresh run rebuilds both sides from one generation and they agree by construction, which
is exactly how this stayed invisible.

### 2. The manifest pinned one file, and the pin never matched it

`README` advertises "the dataset is frozen to disk with a sha256-pinned manifest". In fact
`data/manifest.json` pinned a single file, `ramr_chains_v0.1.0.jsonl`, at
`00f1587680672690…` — which matches **no committed version of that file**: not the blob, not a CRLF or
LF variant, not a whitespace-stripped one. The file has been committed exactly once, in June 2026, as
`036983181534ed72…`. The pin was wrong from the first commit and nothing ever checked it. The three
trace files were not pinned at all.

Fixed: the manifest now records all four shipped artifacts with verified hashes, `.gitattributes` pins
`*.jsonl`/`*.json` to `eol=lf` so the working copy equals the blob on every platform, and
`verify_manifest.py` runs in CI. A hash nobody verifies is a claim, not a pin.

### 3. A third shortcut, which survives v0.2

Two shortcuts were already known and fixed in v0.2: the planted record always preceded the record it
replaced (300/300), and every item shipped its own `kind` label. Both fixes hold.

The third is in the **content**, so neither fix touches it. The generator builds the plant by swapping
a fact's final token for one borrowed from the distractor pool, so the stale value connects to nothing
while the real value is usually the subject of another fact in the same chain. Measured on v0.2:

* the current value recurs elsewhere in the trace in **206/300** traces (68.7%), the stale value in
  **5/300** (1.7%);
* "whichever value echoes elsewhere is the current one" is right on **204 of 207** decisive traces,
  **98.6%**, with no semantics and no memory;
* by template: `works at` **105/105** and `is headquartered in` **99/99** — 100% — while
  `currency of X is the` fires on 3 of 94 and is wrong on all 3;
* a capitalisation tie-break (the capitalised value is the stale one) is **59/59**; echo then case
  reaches **260/263 = 98.9%** at **87.7%** coverage.

Runnable: `ramr_traces_value_echo_shortcut.py`, in CI. It is expected to FIND something and to keep
saying so until the generator is re-cut.

**What this means for any score on RAMR traces.** Two string rules reach 98.9% at 87.7% coverage.
Until the generator draws the stale value as well-connected as the current one, a number on this
fixture is not evidence of staleness-detection capability. Report the shortcut baseline beside any
score; that baseline is the ceiling on what the benchmark can currently license.

**Caution on the sign.** In production the correlation may run the other way: stale values often keep
echoing in cached summaries, quoted threads and copied notes, while the correction arrives once. A
detector tuned to "recurring means current" on this fixture could be learning a rule that is
backwards in the field. We have not measured that, and it is the reason the re-cut matters.
