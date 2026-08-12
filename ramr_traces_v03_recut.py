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

WHAT IT CANNOT DO, and this is diagnosed rather than shrugged at. Echo survives at 100% accuracy over
47.7% of traces and no choice of donor fixes it, because the cue is a property of RETRIEVAL, not of the
plant. The echo probe reads the retrieved set; retrieval is lexical on the question, so it returns the
gold chain and rarely the distractor a donor came from. A stored donor is therefore usually not a
retrieved one, and only the chain's own gold entities are reliably present -- but a chain holds exactly
one company, one country and one currency, so there is no same-type entity available to donate. Drawing
across types ("Eskander works at Drmark") would balance echo and introduce a type cue in its place,
which is trading a measured leak for an unmeasured one.

Balancing echo therefore needs the CORPUS to give each chain more than one entity per type, so a
same-type, same-slot, retrieved donor exists. That is a change to build_dataset.py and it is not done
here. Until it is, the floor is published as it stands.

It cannot make the fixture clean, and this file does not claim to: run memaudit afterwards and publish
whatever floor remains. A re-cut that is not measured is a hope.

    python ramr_traces_v03_recut.py && python memaudit.py --adapter ramr \\
        --blind ramr_traces_v0.3_blind.jsonl --labels ramr_traces_v0.3_labels.jsonl
"""
import hashlib
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_trace_export as X  # noqa: E402

VER = os.environ.get("RAMR_TRACE_VER", "0.3")
BLIND = "ramr_traces_v%s_blind.jsonl" % VER
LABELS = "ramr_traces_v%s_labels.jsonl" % VER
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
        s = _subject(d)
        if s:
            out.append((s, s in stored))
    return out


def _subject(sentence):
    """The grammatical subject. 'The currency of Faovia is the polek.' -> Faovia.

    THIS WAS OFF BY ONE AND IT SHIPPED. The first version took the token after 'currency', which is
    the word 'of', so 94 of 300 v0.3 plants read 'The currency of Kamark is the of.' -- 31% of the
    corpus was grammatical nonsense carrying a two-character value. It did not announce itself as a
    parsing bug; it surfaced as the LENGTH cue inverting to 83%, because a lowercase current value
    like 'fenek' was being matched against a donor pool consisting almost entirely of 'of'. A defect
    that arrives disguised as a statistic is why the re-cut gets audited instead of trusted.
    """
    w = (sentence or "").rstrip(".").split()
    if not w:
        return None
    if w[0].lower() in ("the", "a", "an"):
        if "of" in w:                       # 'The currency of X is the Y' -> X
            i = w.index("of")
            return w[i + 1] if i + 1 < len(w) else None
        return w[1] if len(w) > 1 else None
    return w[0]


def _subjects(chain, n_distractors=12):
    """The subject of every distractor that will be placed in the store.

    'The currency of Faovia is the polek.' -> Faovia, not polek: the article is skipped and the
    grammatical subject taken, because it is the subject that other facts point at.
    """
    return [s for s in (_subject(d) for d in (chain.get("distractor_pool") or [])[:n_distractors]) if s]


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
        # NO SUBJECT FITS, and for one slot that is structural rather than accidental: a currency is
        # never the subject of anything in this corpus, so a lowercase value has no connected donor at
        # all. Fall back to the OBJECT of a same-shaped fact -- another currency -- which balances case
        # and length but cannot balance echo. That residue is real and is counted in the report below
        # rather than hidden by silently dropping the plant.
        objs = []
        for d in (chain.get("distractor_pool") or []):
            w = d.rstrip(".").split()
            if len(w) == len(words) and w[-1] != cv and w[-1][:1].isupper() == cv[:1].isupper():
                objs.append((w[-1], False))
        cands = objs
    if not cands:
        return None

    # CONNECTEDNESS OUTRANKS LENGTH, and the first ordering had it backwards. Ranked by measured lift:
    # echo carries 55.2pp and length 16.6pp, so a donor that is STORED -- and therefore echoes like the
    # value it replaces -- is worth more than one that merely matches character count. Preferring exact
    # length first left echo at 50.7% coverage and 100.0% accuracy, the single largest surviving cue.
    stored_exact = sorted([c for c in cands if c[1] and len(c[0]) == len(cv)], key=lambda c: c[0])
    if stored_exact:
        return " ".join(words[:-1] + [stored_exact[0][0]]) + "."
    stored = sorted([c for c in cands if c[1]], key=lambda c: (abs(len(c[0]) - len(cv)), c[0]))
    if stored:
        return " ".join(words[:-1] + [stored[0][0]]) + "."

    exact = sorted([c for c in cands if len(c[0]) == len(cv)], key=lambda c: c[0])
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
    corpus = os.environ.get("RAMR_CORPUS")
    if corpus:
        X.CORPUS = corpus
        print("corpus: %s" % corpus)
    chains = X._chains(n)

    # Swap ONLY the plant chooser on the shipped export path.
    original = X._stale_variant
    X._stale_variant = stale_variant_balanced
    try:
        traces = [X.build_trace(c) for c in chains]
    finally:
        X._stale_variant = original

    if os.environ.get("RAMR_INJECT_SIBLING") == "1":
        # CONNECTIVITY BY CONSTRUCTION, because retrieval will not supply it. Measured on v0.4: 0 of
        # 300 sibling facts were ever retrieved, while all 3 gold facts were retrieved every time.
        # Retrieval is lexical on the question and the question names only the person, so the trace is
        # always the gold chain plus the plant -- and the current value is a link in that chain while
        # the stale value can never be. The echo cue was therefore never a property of the PLANT; it
        # was a property of what the question pulls back, and no choice of donor could reach it.
        #
        # A real memory store would hold other facts about a superseded value -- that is why it was
        # written down in the first place -- so a trace containing one is more faithful, not less. The
        # corroborating fact is added to the trace explicitly and marked as a distractor.
        injected = 0
        for t, c in zip(traces, chains):
            gt = t.get("ground_truth") or {}
            p = gt.get("planted") or {}
            if not p.get("stale_text"):
                continue
            sv = p["stale_text"].rstrip(".").split()[-1]
            present = " ".join(i["text"] for i in t["retrieved"]).split()
            if sv in [w.rstrip(".") for w in present if w.rstrip(".") != sv] or present.count(sv) > 1:
                continue
            sib = next((d for d in (c.get("distractor_pool") or []) if _subject(d) == sv), None)
            if not sib:
                continue
            # The id must be INDISTINGUISHABLE from the store's own. The first version used
            # "sib-<chain_id>", which a scorer reading nothing but the id format could spot -- and the
            # subject of that record IS the stale value, so it solved 205 of 300 traces outright. A
            # fix that plants a brighter cue than the one it removes is not a fix. Content-addressed,
            # so it is also reproducible, which uuid4 ids are not.
            rid = hashlib.sha256((t["chain_id"] + sib).encode("utf-8")).hexdigest()[:10]
            t["retrieved"].append({"rank": len(t["retrieved"]), "record_id": rid,
                                   "text": sib, "kind": "distractor"})
            injected += 1
        print("corroborating facts injected for the stale value: %d" % injected)

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
