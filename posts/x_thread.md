1/ We audited our own agent-memory benchmark before anyone else could.
97.2% of it was solvable by rules that never read a single word.
We published that number first. Then re-cut to 40%.
https://dancenitra.github.io/ramr/

2/ The auditor is memaudit.py: position, length, casing, stray fields, id shape — scored against permuted labels so every probe has a *measured* chance level.
0 of 8 probes fire on clean data. A planted cue reads 100% vs 0% null. Both asserted in CI.

3/ It caught us twice.
A "fix" that inverted a cue (73%→83% the other way, reads as chance if you score one direction).
An id-shape leak no probe was looking at.
Both in ERRATA.md. Not deleted.

4/ One missing hop collapses a multi-hop answer. Every model, every family.
gold chain 1.000 → drop one hop 0.000
+1.000, CI [1,1], n=200. Seven models, six families, +0.90 to +1.00.

5/ Echo-resistance: correct a fact, then re-state the old value. Does it come back?
keyed store 0.00 · mem0 0.53 · Graphiti 0.87 raw — and 0 of 26 registered corrections flipped, so that 13% is an extraction miss, not the echo · object-keyed ledger 1.00
We say so because the table alone would mislead you.

6/ Before any of that: was the comparison even admissible?
A memory arm at k=20 (1,323 chars) vs session BM25 (11,941). 9.03x. BM25 "won".
Matched: 0.283 → 0.593, ranking flipped.
Four gates. An abort names the dead layer.

7/ Every number on the page is a clickable receipt to the file it was recomputed from.
There's a "shortcut mode" toggle: read the page the way a surface-rule reader would.
Code + PDF + DOI: github.com/DanceNitra/ramr
