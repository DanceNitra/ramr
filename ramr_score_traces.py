"""Score a stale-record detector on RAMR traces -- against the reference rows AND the shortcut floor.

WHAT THIS IS FOR. `ramr_trace_export.py` builds traces and scores them in one pass, which is fine for
the generator and useless for anyone else: an external detector needs to be scored on the traces as
PUBLISHED, from the blind file plus the separate labels, exactly as a third party receives them. This
runner does that, for any trace version.

THREE REFERENCE ROWS, and each fails in a different direction, which is the point:

    null      flags nothing. Must read 0.00 / 0.00 or the harness is inventing credit.
    oracle    reads the label. Must read 1.00 / 0.00 or the key itself is broken.
    twin      a real structural detector: two retrieved records agreeing on every word but the last.
              It cannot say WHICH half of the pair is stale, so it flags both -- full detection paid
              for with a false positive on every plant. That is the honest shape of the task.

AND THE FLOOR, which is what makes any of it interpretable. memaudit reports how much of the fixture
falls to rules that read only surface form. A detector scoring at or below that number has not shown
it can do anything the benchmark could not do by itself. On v0.2 the floor was 97.2% over 96% of
traces, so no score on v0.2 meant anything; on v0.5 it is 40.0% over 1.7%, which is why scoring
started here.

    python ramr_score_traces.py --version 0.5

Bring your own detector: a callable taking (query, [{rank, record_id, text}]) and returning one bool
per item. Point --scorer at "module:function".
"""
import argparse
import importlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramr_trace_export import null_score, oracle_score, twin_score  # noqa: E402


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def kind_of(rid, gt):
    p = gt.get("planted") or {}
    if rid == p.get("record_id"):
        return "planted"
    return "gold" if rid in (gt.get("gold_ids") or []) else "distractor"


def rebuild(blind_path, labels_path):
    """The traces as a third party gets them: blind file, labels joined from the separate file."""
    labels = {r["chain_id"]: r["ground_truth"] for r in load(labels_path)}
    out = []
    for r in load(blind_path):
        gt = labels.get(r["chain_id"]) or {}
        out.append({"chain_id": r["chain_id"], "query": r.get("query"),
                    "retrieved": [{"rank": i["rank"], "record_id": i["record_id"],
                                   "text": i.get("text", ""),
                                   "kind": kind_of(i["record_id"], gt)} for i in r["retrieved"]],
                    "ground_truth": gt})
    return out


def evaluate(traces, scorer, name, oracle=False):
    tp = fp = planted_total = clean_total = skipped = 0
    for t in traces:
        items = t["retrieved"]
        if not items:
            skipped += 1
            continue
        labels = [i["kind"] == "planted" for i in items]
        blind = [{"rank": i["rank"], "record_id": i["record_id"], "text": i["text"]} for i in items]
        if oracle:
            for b, i in zip(blind, items):
                b["_kind_for_oracle"] = i["kind"]
        flags = scorer(t["query"], blind)
        if len(flags) != len(items):
            raise ValueError("%s returned %d flags for %d items" % (name, len(flags), len(items)))
        for f, lab in zip(flags, labels):
            if lab:
                planted_total += 1
                tp += bool(f)
            else:
                clean_total += 1
                fp += bool(f)
    det = (tp / planted_total) if planted_total else None
    fpr = (fp / clean_total) if clean_total else None
    prec = (tp / (tp + fp)) if (tp + fp) else None
    f1 = (2 * prec * det / (prec + det)) if (prec and det) else None
    return {"scorer": name, "detection": det, "false_positive": fpr, "precision": prec, "f1": f1,
            "planted": planted_total, "clean": clean_total, "skipped": skipped}


def pct(x):
    return "   n/a" if x is None else "%5.1f%%" % (100 * x)


def num(x):
    return "  n/a" if x is None else "%.3f" % x


def main(argv=None):
    ap = argparse.ArgumentParser(description="score a detector on published RAMR traces")
    ap.add_argument("--version", default="0.5")
    ap.add_argument("--scorer", default=None, help='"module:function" -- your detector')
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)

    blind = "ramr_traces_v%s_blind.jsonl" % a.version
    labels = "ramr_traces_v%s_labels.jsonl" % a.version
    for p in (blind, labels):
        if not os.path.exists(p):
            print("MISSING: %s" % p)
            return 2

    traces = rebuild(blind, labels)
    print("RAMR traces v%s -- %d traces, %d candidates median\n"
          % (a.version, len(traces),
             sorted(len(t["retrieved"]) for t in traces)[len(traces) // 2]))

    rows = [evaluate(traces, null_score, "null (flags nothing)"),
            evaluate(traces, oracle_score, "oracle (reads the label)", oracle=True),
            evaluate(traces, twin_score, "contradiction twin (structural)")]

    if a.scorer:
        mod, _, fn = a.scorer.partition(":")
        rows.append(evaluate(traces, getattr(importlib.import_module(mod), fn), a.scorer))

    hdr = "%-34s %9s %9s %9s %7s" % ("scorer", "detection", "false-pos", "precision", "F1")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-34s %9s %9s %9s %7s" % (r["scorer"], pct(r["detection"]), pct(r["false_positive"]),
                                         num(r["precision"]), num(r["f1"])))

    # The harness checks ITSELF before it reports anyone. A null that scores above zero or an oracle
    # that cannot reach one means the numbers above are about the harness, not the detectors.
    bad = []
    if rows[0]["detection"] or rows[0]["false_positive"]:
        bad.append("the NULL scorer scored above zero -- the harness credits flags nobody made")
    if rows[1]["detection"] != 1.0 or rows[1]["false_positive"] != 0.0:
        bad.append("the ORACLE did not reach 1.00/0.00 -- the label join is broken")
    print()
    if bad:
        for b in bad:
            print("HARNESS FAILURE: %s" % b)
        return 1
    print("harness controls hold: null 0.00/0.00, oracle 1.00/0.00")
    print("\nRead every row against the shortcut floor for this version:")
    print("  python memaudit.py --adapter ramr --blind %s --labels %s" % (blind, labels))

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"version": a.version, "traces": len(traces), "rows": rows}, indent=2) + "\n")
        print("wrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
