"""Was the twin baseline measured on the SAME shuffled data as the detector it is compared against?

MARAT SULTANOV ASKED THIS (deepseek-ai/DeepSeek-V3#1466), and it is the right question. The v0.9
detector was scored on v0.2 -- blind and shuffled. If the "contradiction twin" baseline it is compared
against was scored on v0.1 -- labelled and in generator order, where the planted record precedes the
gold record it replaces in 300 of 300 traces -- then the comparison is between two different datasets
and the confound dominates any margin either way.

The answer cannot be argued from the shape of the scorer. It is one run:

  * rebuild BOTH datasets in the same harness -- v0.1 as shipped, and v0.2 from the blind artifact
    plus its separate label file, which is what a third party actually receives;
  * assert they are the same items, per trace, so that ORDER is the only thing that differs;
  * score the twin baseline on each with the SAME evaluate() the export uses -- imported, not
    reimplemented, because a reconstruction measures the reconstruction;
  * and carry a POSITIVE CONTROL that must break: the positional scorer, which reads order and
    nothing else. It has to score ~100% on v0.1 and ~50% on v0.2. If it does not, the two datasets in
    front of me are not actually ordered differently, and any "identical" verdict below would be an
    artifact of loading the same thing twice.

    python ramr_traces_order_invariance.py
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramr_trace_export import evaluate, twin_score  # noqa: E402

V01 = "ramr_traces_v0.1.jsonl"
BLIND = "ramr_traces_v0.2_blind.jsonl"
LABELS = "ramr_traces_v0.2_labels.jsonl"
OUT = "ramr_traces_order_invariance_result.json"


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def kind_of(rid, gt):
    planted = gt.get("planted") or {}
    if rid == planted.get("record_id"):
        return "planted"
    return "gold" if rid in (gt.get("gold_ids") or []) else "distractor"


def relabel(blind, label_of):
    """Put the v0.2 artifact back into the harness shape, taking labels from the SEPARATE file.

    This is exactly what an external scorer's evaluator has to do, and it is the only place the label
    may enter: the blind file does not carry one.
    """
    out = []
    for t in blind:
        gt = label_of[t["chain_id"]]
        out.append({"chain_id": t["chain_id"], "query": t["query"],
                    "retrieved": [{"rank": i["rank"], "record_id": i["record_id"], "text": i["text"],
                                   "kind": kind_of(i["record_id"], gt)} for i in t["retrieved"]]})
    return out


def positional(traces, label_of):
    """POSITIVE CONTROL. Reads rank order and nothing else -- must die when the order is shuffled."""
    right = pairs = 0
    for t in traces:
        gt = label_of[t["chain_id"]]
        pid = (gt.get("planted") or {}).get("record_id")
        rep = (gt.get("planted") or {}).get("replaces")
        idx = {i["record_id"]: i["rank"] for i in t["retrieved"]}
        if pid not in idx or rep not in idx:
            continue
        pairs += 1
        right += (pid if idx[pid] < idx[rep] else rep) == pid
    return right, pairs


def pct(x):
    return "n/a" if x is None else "%.2f%%" % (100 * x)


def main():
    for p in (V01, BLIND, LABELS):
        if not os.path.exists(p):
            print("MISSING: %s -- run ramr_traces_v02_blind_and_shuffled.py first" % p)
            return 2

    v01 = load(V01)
    label_of = {t["chain_id"]: t["ground_truth"] for t in v01}
    v02 = relabel(load(BLIND), label_of)
    print("loaded: v0.1 %d traces, v0.2 %d traces" % (len(v01), len(v02)))

    # SAME ITEMS, DIFFERENT ORDER -- asserted, not assumed. Without this the comparison could be
    # between two different samples and every number below would be meaningless.
    a = {t["chain_id"]: sorted(i["record_id"] for i in t["retrieved"]) for t in v01}
    b = {t["chain_id"]: sorted(i["record_id"] for i in t["retrieved"]) for t in v02}
    same_items = a == b
    order_differs = sum(
        1 for t1, t2 in zip(v01, v02)
        if [i["record_id"] for i in t1["retrieved"]] != [i["record_id"] for i in t2["retrieved"]])
    print("same items per trace : %s" % same_items)
    print("traces whose ORDER differs : %d of %d" % (order_differs, len(v01)))
    if not same_items:
        print("ABORT: the two versions do not hold the same items; this is not an order comparison.")
        return 2

    ctl01, ctl02 = positional(v01, label_of), positional(v02, label_of)
    r01, r02 = ctl01[0] / ctl01[1], ctl02[0] / ctl02[1]
    print("\nPOSITIVE CONTROL -- positional scorer (order-only):")
    print("  v0.1 %d/%d = %.1f%%   v0.2 %d/%d = %.1f%%" % (ctl01 + (100 * r01,) + ctl02 + (100 * r02,)))
    control_ok = r01 > 0.95 and 0.35 < r02 < 0.65
    print("  control %s" % ("HOLDS -- the shuffle is real and this harness can see it"
                            if control_ok else "FAILED -- the datasets are not ordered differently"))

    t01 = evaluate(v01, twin_score, "contradiction twin on v0.1 (labelled, generator order)")
    t02 = evaluate(v02, twin_score, "contradiction twin on v0.2 (blind, shuffled)")
    print("\nTWIN BASELINE:")
    hdr = "%-52s %10s %10s %8s %7s" % ("dataset", "detection", "false-pos", "planted", "clean")
    print(hdr)
    print("-" * len(hdr))
    for r in (t01, t02):
        print("%-52s %10s %10s %8d %7d" % (r["scorer"], pct(r["detection_rate"]),
                                           pct(r["false_positive_rate"]), r["planted_items"],
                                           r["clean_items"]))

    # TIE THIS ROW TO THE NUMBER HE ACTUALLY SAW. The comparison posted on #1466 quoted a baseline
    # precision of 0.463, and a claim that "the baseline is the twin" is worth nothing unless the
    # twin's own cells reproduce it. Precision from the confusion cells, recomputed here, not copied.
    prec = {}
    for tag, r in (("v01", t01), ("v02", t02)):
        tp = r["detection_rate"] * r["planted_items"]
        fp = r["false_positive_rate"] * r["clean_items"]
        prec[tag] = tp / (tp + fp) if (tp + fp) else None
    print("twin precision (TP/(TP+FP)) : v0.1 %.4f   v0.2 %.4f" % (prec["v01"], prec["v02"]))
    print("  the baseline quoted at 0.463 on #1466 is THIS row: %s"
          % ("matches" if abs(prec["v02"] - 0.463) < 0.001 else
             "DOES NOT MATCH -- the baseline in that comparison was some other scorer"))

    identical = (t01["detection_rate"] == t02["detection_rate"]
                 and t01["false_positive_rate"] == t02["false_positive_rate"]
                 and t01["planted_items"] == t02["planted_items"]
                 and t01["clean_items"] == t02["clean_items"])
    print()
    if not control_ok:
        verdict = "INCONCLUSIVE -- the control did not reproduce the shuffle"
    elif identical:
        verdict = ("NO CONFOUND -- the twin baseline scores identically on both, so which version it "
                   "was measured on cannot move the comparison")
    else:
        verdict = ("CONFOUNDED -- the twin baseline moves with the shuffle, so it must be re-measured "
                   "on the same file as any detector it is compared against")
    print("VERDICT: %s" % verdict)

    io.open(OUT, "w", encoding="utf-8").write(json.dumps({
        "question": "was the twin baseline measured on the same shuffled data as the detector?",
        "asked_by": "Marat Sultanov, deepseek-ai/DeepSeek-V3#1466",
        "same_items_per_trace": same_items, "traces_whose_order_differs": order_differs,
        "control_positional": {"v01": list(ctl01), "v02": list(ctl02), "holds": bool(control_ok)},
        "twin": {"v01": t01, "v02": t02}, "twin_precision": prec,
        "identical": bool(identical), "verdict": verdict,
    }, indent=2))
    print("wrote %s" % OUT)
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
