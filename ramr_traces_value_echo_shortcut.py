"""A third shortcut in the RAMR traces, ours, and found before anyone exploited it.

TWO WERE ALREADY KNOWN. Marat Sultanov (deepseek-ai/DeepSeek-V3#1466) found that the planted record
precedes the gold record it replaces in 300 of 300 traces. Separately, every item in v0.1 shipped its
own `kind` field. v0.2 killed both -- shuffled order, labels in a separate file.

THIS ONE SURVIVES v0.2, because it is in the CONTENT and neither fix touched content. The generator
builds the plant by taking a gold fact and swapping its final token for one borrowed from the
distractor pool. The gold fact's real value, meanwhile, is usually the subject of some OTHER fact in
the same chain -- that is what makes the chain a chain. The stale value is borrowed and connects to
nothing. So the two halves of every contradiction pair are distinguishable by a rule that needs no
semantics and no memory: whichever value ECHOES elsewhere in the retrieved set is the current one.

That matters for anyone scoring a semantic detector here (an embedding model was proposed for exactly
this benchmark): a strong number may be reading this shortcut rather than the defect, and it will not
survive contact with real traces, where a superseded value is usually just as well connected as the
one that replaced it -- it used to be true.

The honest fix is a generator change: draw the stale value so that it is as connected as the current
one, and re-cut. Until then this file states the size of the hole.

    python ramr_traces_value_echo_shortcut.py
"""
import io
import json
import os
import sys

BLIND = "ramr_traces_v0.2_blind.jsonl"
LABELS = "ramr_traces_v0.2_labels.jsonl"
OUT = "ramr_traces_value_echo_shortcut_result.json"


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def main():
    missing = [p for p in (BLIND, LABELS) if not os.path.exists(p)]
    if missing:
        print("MISSING: %s" % ", ".join(missing))
        return 2

    traces = load(BLIND)
    label_of = {t["chain_id"]: t["ground_truth"] for t in load(LABELS)}

    n = cur = stale = both = neither = 0
    for t in traces:
        p = (label_of[t["chain_id"]].get("planted") or {})
        if not p.get("current_text") or not p.get("stale_text"):
            continue
        n += 1
        cv = p["current_text"].rstrip(".").split()[-1]
        sv = p["stale_text"].rstrip(".").split()[-1]
        # Everything EXCEPT the contradiction pair itself -- otherwise each value trivially "echoes"
        # its own record and the measurement answers a question nobody asked.
        rest = " ".join(i["text"] for i in t["retrieved"]
                        if i["record_id"] not in (p.get("record_id"), p.get("replaces"))).split()
        c, s = cv in rest, sv in rest
        cur += c
        stale += s
        both += c and s
        neither += (not c) and (not s)

    decisive = (cur - both) + (stale - both)
    right = cur - both
    print("traces carrying a plant                      : %d" % n)
    print("CURRENT value recurs elsewhere in the trace  : %d (%.1f%%)" % (cur, 100.0 * cur / n))
    print("STALE   value recurs elsewhere in the trace  : %d (%.1f%%)" % (stale, 100.0 * stale / n))
    print("both %d | neither %d | decisive (exactly one) %d" % (both, neither, decisive))
    print()
    print("RULE: \"the value that echoes elsewhere is the current one\" -- no semantics, no memory.")
    print("  correct on %d of %d decisive traces = %.1f%%" % (right, decisive, 100.0 * right / decisive))
    print("  coverage: %d of %d traces = %.1f%% (the rest are silent, not wrong)"
          % (decisive, n, 100.0 * decisive / n))

    io.open(OUT, "w", encoding="utf-8").write(json.dumps({
        "traces_with_plant": n, "current_value_echoes": cur, "stale_value_echoes": stale,
        "both": both, "neither": neither, "decisive": decisive, "correct_on_decisive": right,
        "accuracy_on_decisive": right / decisive if decisive else None,
        "coverage": decisive / n if n else None,
        "note": "survives v0.2: the shuffle fixed order and the split fixed labels; this is content",
    }, indent=2))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
