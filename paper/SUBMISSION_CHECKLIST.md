# arXiv submission checklist — RAMR preprint (owner action)

**Source file:** `paper/ramr_preprint.tex` (compiles clean: 9 pages, zero undefined
references/citations; compiled with MiKTeX pdflatex, plain `article` class — no exotic
packages, so arXiv's TeX build will process it as-is).

## What to do (once, in order)

1. **Endorsement (do this FIRST — takes days).** arXiv requires a first-time submitter in a
   category to be endorsed by an existing author. Go to https://arxiv.org/auth/endorsement
   → enter your email → pick category `cs.CL` (or `cs.IR`) → ask an eligible endorser (a
   colleague/author who has published on arXiv in that category; endorsement code request sends
   them a link). While waiting, continue steps 2-4.
2. **Register/confirm account** at arxiv.org under your name (Rastislav Drahoš).
3. **Prepare the submission:**
   - Single-file LaTeX: `ramr_preprint.tex` (no figures, no bib file — thebibliography is
     inline, so arXiv needs only the .tex).
   - Categories: primary `cs.CL`; cross-list `cs.IR` and `cs.AI`.
   - Title: "RAMR: A Contamination-Resistant Synthetic Benchmark for Agentic
     Retrieval-Augmented Memory, Audited by Its Own Shortcut Battery"
   - Authors: Rastislav Drahoš (Agora, Nitra, Slovakia) — same name that is already public
     on the Zenodo record since v0.5.1; no new exposure.
   - Comments field: `Code: https://github.com/DanceNitra/ramr. Dataset: DOI
     10.5281/zenodo.20818291. HF: https://huggingface.co/datasets/Danchi17/ramr`
4. **License choice:** the repo is MIT; on arXiv pick one — we recommend the standard
   arXiv non-exclusive license (default) unless you want CC BY 4.0.
5. **Submit** → hold for moderation (usually 1-2 business days for cs.*).
6. **After it goes live:** the arXiv ID unlocks Hugging Face Papers
   (https://huggingface.co/papers/submit takes the arXiv ID) — submit it there, then add
   the arXiv id to CITATION.cff and the README badge row.

## What the paper is (and the honesty rules it keeps)

- All numbers trace to persisted files in the repo; the verification ledger recomputes them.
- Cross-system cells are scoped as native-configuration comparisons, not a product sweep;
  mem0/Graphiti results carry their CIs and the "not a leaderboard sweep" scope.
- The self-audit section credits the external reviewer's positional-invariant find and names
  the still-open defect (ids not content-addressed).
- The abstract links code + dataset DOI per arXiv conventions.

## What this does NOT do

- It does not submit anything by itself. Steps 1-5 are yours (endorsement + login + license
  choice are account-bound).
- It does not change any measured claim in the repo; every number was already published.
