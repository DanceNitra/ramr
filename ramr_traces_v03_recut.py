"""v0.3 -- the structural re-cut. The stale value is drawn to look like the current one.

WHY A RE-CUT AND NOT A FILTER. memaudit put a number on it: v0.2's shortcut floor is 97.2% over 96.0%
of traces, from three surface rules. Filtering the cues that produce that buys one round -- the history
of adversarial filtering is that the shortcut relocates. The intervention with a track record is
changing what the label DEPENDS ON, so that is what this does.

WHERE THE LEAK CAME FROM. A chain is "Eskander works at Nimcorp." / "Nimcorp is headquartered in
Drmark." / "The currency of Drmark is the fenek." The value of each fact is the SUBJECT of the next
one -- that is what makes it a chain. The plant replaced a fact's final token with the final token of a
distractor, which is an OBJECT and therefore the subject of nothing. So the current value was connected
to the rest of the trace and the stale value was connected to nothing, and "whichever value echoes
elsewhere is the current one" scored 98.6%. Casing and length leaked for the same reason: an object
drawn from a different slot has different capitalisation and a different length distribution.

WHAT v0.3 CHANGES. The stale value is drawn from the SUBJECTS of the distractors that will be in the
store, so it is the subject of a fact the retriever can also return -- as connected as the value it
replaces. Among those candidates it takes the one matching the current value's capitalisation and
closest to its length. All three cues are addressed at the source rather than filtered afterwards.

It uses the SHIPPED export path (`build_trace` from ramr_trace_export) with only the plant chooser
swapped, so this measures the generator we actually run rather than a reconstruction of it.

WHAT IT CANNOT DO. It cannot make the fixture clean, and this file does not claim to: run memaudit
afterwards and publish whatever floor remains. A re-cut that is not measured is a hope.

    python ramr_traces_v03_recut.py && python memaudit.py --adapter ramr \\
        --blind ramr_traces_v0.3_blind.jsonl --labels ramr_traces_v0.3_labels.jsonl
"""
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_trace_export as X  # noqa: E402

BLIND = "ramr_traces_v0.3_blind.jsonl"
LABELS = "ramr_traces_v0.3_labels.jsonl"
SEED = 20260812


def _subjects_all(chain):
    """Subjects of the WHOLE distractor pool, with a flag for the ones that will be in the store.

    Two goals pull against each other. A donor whose fact is stored is CONNECTED, which is what kills
    the echo cue; but the stored slice is only 12 facts and its subjects skew short, so restricting to
    it makes the length cue impossible to balance -- measured: the "longer" branch was empty, every
    fallback went short, and the inverted length cue survived at 83.3%. So candidates are drawn from
    the whole pool and stored donors win ties. Neither cue is fully removable from this vocabulary;
    the residue is measured and published rather than assumed away.
    """
    stored = set(_subjects(chain))
    out = []
    for d in (chain.get("distractor_pool") or []):
        w = d.rstrip(".").split()
        if not w:
            continue
        if w[0].lower() == "the" and len(w) > 3:
            s = w[2] if w[1].lower() == "currency" and len(w) > 2 else w[1]
        else:
            s = w[0]
        if s:
            out.append((s, s in stored))
    return out


def _subjects(chain, n_distractors=12):
    """The subject of every distractor that will be placed in the store.

    'The currency of Faovia is the polek.' -> Faovia, not polek: the article is skipped and the
    grammatical subject taken, because it is the subject that other facts point at.
    """
    out = []
    for d in (chain.get("distractor_pool") or [])[:n_distractors]:
        w = d.rstrip(".").split()
        if not w:
            continue
        if w[0].lower() == "the" and len(w) > 3:
            out.append(w[2] if w[1].lower() == "currency" and len(w) > 2 else w[1])
        else:
            out.append(w[0])
    return [s for s in out if s]


def stale_variant_balanced(fact, chain):
    """A superseded value that is as CONNECTED, as capitalised and as long as the one it replaces.

    LENGTH IS BALANCED, NOT MINIMISED, and the first attempt got this wrong in an instructive way.
    Taking the nearest length looks like the careful choice and is not: the donor subjects skew short,
    so "nearest" landed shorter 83.3% of the time and memaudit read a STRONGER cue than before, merely
    pointing the other way. Removing a bias and inverting one are different operations. So: prefer an
    exact length match, and when there is none, alternate which side of the current value's length we
    take, keyed on the chain id so the corpus is balanced and the choice stays deterministic.
    """
    words = fact.rstrip(".").split()
    if len(words) < 3:
        return None
    cv = words[-1]
    cands = [(s, st) for s, st in _subjects_all(chain)
             if s != cv and s[:1].isupper() == cv[:1].isupper()]
    if not cands:
        return None

    # Exact length first, stored donors preferred within it -- that satisfies both cues at once.
    exact = sorted([c for c in cands if len(c[0]) == len(cv)], key=lambda c: (not c[1], c[0]))
    if exact:
        return " ".join(words[:-1] + [exact[0][0]]) + "."

    longer = sorted([c for c in cands if len(c[0]) > len(cv)], key=lambda c: (len(c[0]), not c[1], c[0]))
    shorter = sorted([c for c in cands if len(c[0]) < len(cv)], key=lambda c: (-len(c[0]), not c[1], c[0]))
    prefer_long = sum(ord(c) for c in str(chain.get("id", ""))) % 2 == 0
    order = (longer, shorter) if prefer_long else (shorter, longer)
    pick = (order[0] or order[1])[0][0]
    return " ".join(words[:-1] + [pick]) + "."


def main():
    n = int(os.environ.get("TRACES", "300"))
    chains = X._chains(n)

    # Swap ONLY the plant chooser on the shipped export path.
    original = X._stale_variant
    X._stale_variant = stale_variant_balanced
    try:
        traces = [X.build_trace(c) for c in chains]
    finally:
        X._stale_variant = original

    planted = sum(1 for t in traces if any(i["kind"] == "planted" for i in t["retrieved"]))
    print("chains: %d | traces whose plant surfaced in top-%d: %d" % (len(traces), X.K, planted))

    rnd = random.Random(SEED)
    blind, labels = [], []
    for t in traces:
        items = [{"record_id": i["record_id"], "text": i["text"]} for i in t["retrieved"]]
        rnd.shuffle(items)
        for r, i in enumerate(items):
            i["rank"] = r
        blind.append({"chain_id": t["chain_id"], "query": t["query"],
                      "retrieved": [{"rank": i["rank"], "record_id": i["record_id"],
                                     "text": i["text"]} for i in items]})
        labels.append({"chain_id": t["chain_id"], "ground_truth": t["ground_truth"]})

    for path, rows in ((BLIND, blind), (LABELS, labels)):
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The three cues, measured directly on what was just written, so the claim in the docstring is
    # checkable without running the whole battery.
    lab = {r["chain_id"]: r["ground_truth"] for r in labels}
    echo_c = echo_s = case_split = longer = differing = pairs = 0
    for t in blind:
        p = (lab[t["chain_id"]].get("planted") or {})
        if not p.get("current_text"):
            continue
        byid = {i["record_id"]: i["text"] for i in t["retrieved"]}
        a, b = byid.get(p.get("record_id")), byid.get(p.get("replaces"))
        if a is None or b is None:
            continue
        pairs += 1
        cv = p["current_text"].rstrip(".").split()[-1]
        sv = p["stale_text"].rstrip(".").split()[-1]
        rest = " ".join(v for k, v in byid.items()
                        if k not in (p.get("record_id"), p.get("replaces"))).split()
        echo_c += cv in rest
        echo_s += sv in rest
        case_split += cv[:1].isupper() != sv[:1].isupper()
        if len(a) != len(b):
            differing += 1
            longer += len(a) > len(b)

    print("\ncue balance on v0.3 (%d pairs):" % pairs)
    print("  current value echoes elsewhere : %d (%.1f%%)" % (echo_c, 100.0 * echo_c / pairs))
    print("  stale   value echoes elsewhere : %d (%.1f%%)" % (echo_s, 100.0 * echo_s / pairs))
    print("  pairs split by capitalisation  : %d (%.1f%%)" % (case_split, 100.0 * case_split / pairs))
    print("  stale longer, of those differing: %d/%d (%.1f%%)"
          % (longer, differing, 100.0 * longer / differing if differing else 0.0))
    print("\nwrote %s and %s" % (BLIND, LABELS))
    print("now run:  python memaudit.py --adapter ramr --blind %s --labels %s" % (BLIND, LABELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
