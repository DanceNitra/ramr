Title: [R] We audited our own agent-memory benchmark before anyone else could: 97.2% of it was solvable by surface rules. Here's the tool, the number, and the re-cut (40%).

RAMR is a small synthetic benchmark for retrieval-backed agent memory (random-token entities, so closed-book accuracy is 0.000 by construction). Eleven metrics, each with a pre-registered falsifier and bootstrap CIs; every cited number is recomputed from its persisted result file by `verify_numbers.py`.

The part I think is actually interesting: it ships `memaudit.py`, a partial-input baseline battery (position, length, casing, token recurrence, query overlap, stray fields, id shape) that reports the **shortcut floor** — the accuracy reachable by rules that never read the content — scored against permuted labels so each probe has a measured chance level rather than an assumed one.

Run on our own traces first:

- v0.2: **97.2%** (position + a stray field)
- v0.3 "fix": 96.8% — the length cue wasn't fixed, it was inverted
- v0.5, connectivity-balanced: **40.0%** at 1.7% coverage
- LoCoMo under our framing, for comparison: 36.3%, no positional artifact, no stray field — cleaner than ours

It caught us twice (an inverted cue that one-directional scoring reads as chance, and an id-shape leak no probe was looking at). Both are in ERRATA.md, not deleted.

Other findings that survived their falsifiers: chain-fragility +1.000 CI[1,1] at n=200 on two model families (one missing hop → 0.000); outcome-ranked recall lift +0.36…+0.47 with a negative random-credit control; echo-resistance measured cross-system at the answer level (keyed store 0.00, mem0 0.53, Graphiti 0.87 raw with 0/26 registered corrections flipped, object-keyed ledger 1.00). There's also a four-gate preflight (budget parity / retrieval / liveness / parameter efficacy) that aborts by naming the dead layer — motivated by a comparison where a k=20 memory arm faced session-level BM25 at 9.03x the context.

Limitations first, on the page: synthetic, uneven n, one embedder, substring matching, and the traces leak by a published amount.

Site (every number is a clickable receipt to its source file, and there's a "shortcut mode" toggle that shows you what a surface-rule reader sees): https://dancenitra.github.io/ramr/
Code: https://github.com/DanceNitra/ramr · Write-up PDF + DOI: 10.5281/zenodo.20818291

Happy to be told which of these numbers don't hold up. That's what the ledger is for.
