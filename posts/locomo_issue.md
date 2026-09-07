Title: Partial-input baseline result on LoCoMo (our framing): 36.3% floor, no positional artifact — sharing the numbers, not a verdict

Hi — I maintain RAMR, a small synthetic agent-memory benchmark. We built a partial-input baseline battery (`memaudit.py`, after Feng, Wallace & Boyd-Graber, ACL 2019) to measure how much of *our own* benchmark is solvable by surface-form rules alone, with a permuted-label null per probe. Our first traces read 97.2%; we published that and re-cut to 40%.

For a comparison point we ran the same battery on LoCoMo, under **our** framing (every dialogue turn as a candidate, single-evidence questions only). Result:

- shortcut floor **36.3%** over 44.3% of questions
- **no positional artifact** (position:first / pair-order at chance)
- **no stray label field**
- the coverage comes from content-level cues (value echo / query overlap), which are much harder to call artifacts than the structural ones we found in our own data

I want to be explicit that this is a floor for our framing, not a verdict on LoCoMo — a passing probe proves nothing (Feng et al.), and LoCoMo is clean on exactly the two axes where our data leaked. We don't redistribute anything; the adapter takes a path to the user's own copy.

Reproduce: `python memaudit.py --adapter locomo --path <your LoCoMo copy>` from https://github.com/DanceNitra/ramr. Method and the null construction: https://dancenitra.github.io/ramr/#floor

If you'd like the per-probe table or think the framing is unfair to the dataset, I'll adjust and re-run. Filing as an issue only because I couldn't find a discussions tab; feel free to close.
