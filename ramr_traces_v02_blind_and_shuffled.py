"""v0.2 of the RAMR traces: the label removed from the artifact, and the order randomised.

TWO DEFECTS IN v0.1, one reported and one not.

REPORTED. Marat Sultanov (DeepSeek-V3#1466) found that the planted record always ranks before the
gold record it replaces. Measured here on the shipped file: **300 of 300, 100%**. His v0.7 scorer
reaches 100% detection / 1.3% FP by exploiting exactly that, and he said so himself rather than
banking the number -- which is the only reason anyone knew.

NOT REPORTED, and worse. Every item in the shipped `ramr_traces_v0.1.jsonl` carries its own
`kind` field: `planted` | `gold` | `distractor`. The in-repo harness strips it before handing a
scorer the trace (`blind = [{rank, record_id, text}]`), so the INTERFACE was always right -- but the
exported artifact, the thing a third party actually receives, shipped the answer key inside the
question. Any number computed from that file is only as trustworthy as the scorer's discipline in
not reading a field that was sitting there. Marat's clearly did not read it; that is luck plus his
integrity, not a property of the benchmark.

WHAT v0.2 CHANGES.

  * `traces_blind.jsonl` carries {chain_id, query, retrieved:[{rank, record_id, text}]} and NOTHING
    else. The label cannot be read because it is not present.
  * `labels.jsonl` carries the ground truth, keyed by chain_id, in a separate file.
  * The retrieved list is SHUFFLED with a recorded seed, so the planted record is as likely to
    follow the gold record as to precede it. The positional rule stops working by construction, not
    by request.

AND THE CONTROLS, because a fix nobody can see fail is not a fix. Two adversarial scorers run
against both versions:

    label_reader   -- returns kind == "planted". Must score 100% on v0.1 and be UNABLE to run on v0.2.
    positional     -- flags the earlier item of each contradiction pair. Must score ~100% on v0.1 and
                      ~50% (chance within a pair) on v0.2.

If either still works on v0.2, the fix did not land and this file says so.

    python ramr_traces_v02_blind_and_shuffled.py
"""
import io
import json
import os
import random

SRC = "ramr_traces_v0.1.jsonl"
BLIND = "ramr_traces_v0.2_blind.jsonl"
LABELS = "ramr_traces_v0.2_labels.jsonl"
SEED = 20260812


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def positional_invariant(traces, label_of):
    """How often does the planted record precede the gold record it replaces?"""
    before = pairs = 0
    for t in traces:
        gt = label_of[t["chain_id"]]
        pid, rep = gt["planted"]["record_id"], gt["planted"].get("replaces")
        idx = {i["record_id"]: i["rank"] for i in t["retrieved"]}
        if pid in idx and rep in idx:
            pairs += 1
            before += idx[pid] < idx[rep]
    return before, pairs


def label_reader_score(traces):
    """Reads `kind` if it is there. Returns None when the field is absent -- which is the point."""
    hits = total = 0
    for t in traces:
        for it in t["retrieved"]:
            if "kind" not in it:
                return None
            total += 1
            hits += it["kind"] == "planted"
    return hits, total


def positional_score(traces, label_of):
    """Flags the EARLIER member of each contradiction pair as planted. Marat's v0.7, reproduced."""
    right = pairs = 0
    for t in traces:
        gt = label_of[t["chain_id"]]
        pid, rep = gt["planted"]["record_id"], gt["planted"].get("replaces")
        idx = {i["record_id"]: i["rank"] for i in t["retrieved"]}
        if pid not in idx or rep not in idx:
            continue
        pairs += 1
        guess = pid if idx[pid] < idx[rep] else rep
        right += guess == pid
    return right, pairs


def main():
    src = load(SRC)
    label_of = {t["chain_id"]: t["ground_truth"] for t in src}
    print("source: %s -- %d traces" % (SRC, len(src)))

    b, p = positional_invariant(src, label_of)
    print("\nv0.1, as shipped:")
    print("  positional invariant       : %d/%d = %.1f%%" % (b, p, 100 * b / p))
    lr = label_reader_score(src)
    print("  label_reader scorer        : %s" % (
        "reads `kind` directly -- %d planted found in %d items, a FREE 100%%" % lr if lr else "field absent"))
    pr = positional_score(src, label_of)
    print("  positional scorer (v0.7)   : %d/%d = %.1f%% correct" % (pr[0], pr[1], 100 * pr[0] / pr[1]))

    rnd = random.Random(SEED)
    blind, labels = [], []
    for t in src:
        items = [{"record_id": i["record_id"], "text": i["text"]} for i in t["retrieved"]]
        rnd.shuffle(items)
        for r, i in enumerate(items):
            i["rank"] = r
        blind.append({"chain_id": t["chain_id"], "query": t["query"],
                      "retrieved": [{"rank": i["rank"], "record_id": i["record_id"], "text": i["text"]}
                                    for i in items]})
        labels.append({"chain_id": t["chain_id"], "ground_truth": t["ground_truth"]})

    with io.open(BLIND, "w", encoding="utf-8") as fh:
        for r in blind:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(LABELS, "w", encoding="utf-8") as fh:
        for r in labels:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    b2, p2 = positional_invariant(blind, label_of)
    lr2 = label_reader_score(blind)
    pr2 = positional_score(blind, label_of)
    print("\nv0.2, blind + shuffled (seed %d):" % SEED)
    print("  positional invariant       : %d/%d = %.1f%%" % (b2, p2, 100 * b2 / p2))
    print("  label_reader scorer        : %s" % ("STILL READS IT -- fix failed" if lr2 else
                                                 "cannot run: `kind` is not in the artifact"))
    print("  positional scorer (v0.7)   : %d/%d = %.1f%% correct" % (pr2[0], pr2[1], 100 * pr2[0] / pr2[1]))

    ok = (lr2 is None) and (0.35 < b2 / p2 < 0.65) and (0.35 < pr2[0] / pr2[1] < 0.65)
    print("\nVERDICT: %s" % ("both exploits are dead -- the label is gone and the order carries no signal"
                            if ok else "AT LEAST ONE EXPLOIT SURVIVES -- do not ship this"))
    out = {"source_traces": len(src), "seed": SEED,
           "v01": {"positional_invariant": [b, p], "label_field_present": lr is not None,
                   "positional_scorer": list(pr)},
           "v02": {"positional_invariant": [b2, p2], "label_field_present": lr2 is not None,
                   "positional_scorer": list(pr2)},
           "both_exploits_dead": bool(ok)}
    io.open("ramr_traces_v0.2_result.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print("wrote %s, %s and ramr_traces_v0.2_result.json" % (BLIND, LABELS))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
