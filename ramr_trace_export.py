"""RAMR: export per-query retrieval TRACES so an external diagnostic can be scored on them.

Marat Sultanov asked (DeepSeek-V3#1466) whether RAMR gives the raw retrieval trace at the level of
individual records or only an aggregated score, because TAT's structural detectors work on ORDERED
SEQUENCES. Measured before answering: the traces exist at runtime -- every metric calls
`store.recall(...)` and gets a ranked list -- but nothing persists them, so every `*_result.json` in
this repo is an aggregate. This module closes that gap.

WHAT IT EXPORTS, one object per query, ordered:

    {"chain_id", "query", "answer",
     "retrieved": [{"rank", "record_id", "text", "kind"}...],   # kind: gold | distractor | planted
     "ground_truth": {"gold_ids": [...], "planted": {"kind", "record_id", "replaces"}}}

`kind` is the label a scorer must NOT see. It is carried in a separate `ground_truth` block and the
scorer is handed only `query` + `retrieved` (with `kind` stripped), which is what makes a detection
rate mean anything.

THE DEFECT PLANTED HERE is echo resurrection: a gold fact is corrected, and the SUPERSEDED value is
left in the store so a plain retriever can return it. It is the cleanest of RAMR's defect types to
label unambiguously per record, which is what a per-item flag needs.

SCORER INTERFACE -- two functions, so nobody has to read this repo to be measured in it:

    score(query: str, retrieved: list[dict]) -> list[bool]     # one flag per retrieved item
    name() -> str

A scorer flags an item True when it believes that item is a planted defect. Detection rate is over
planted items; false-positive rate is over clean ones. Both are reported, never one alone.

CONTROLS, and the run refuses without them:
  * a NULL scorer that always returns False -- must score 0.00 detection and 0.00 FP. If it scores
    anything, the harness is crediting flags nobody made.
  * an ORACLE scorer that reads ground truth -- must score 1.00 / 0.00. If it cannot, the labelling is
    broken and every number below is measured against a corrupt key.
  * exclusions are counted, never dropped.

    python ramr_trace_export.py            # writes ramr_traces_v0.1.jsonl + runs the reference scorers
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "ramr_chains_v0.1.0.jsonl")
OUT = os.path.join(HERE, "ramr_traces_v0.1.jsonl")
K = 8


def _chains(limit):
    rows = []
    with io.open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def _stale_variant(fact, chain):
    """A superseded version of `fact`: same subject, a value taken from another fact of the same shape.

    No marker, no tell. The only way to know which of the two is current is to know the correction --
    which is exactly the capability under test. Returns None when no same-shaped donor exists, and the
    caller then plants nothing rather than planting something malformed."""
    words = fact.rstrip(".").split()
    if len(words) < 3:
        return None
    tail = words[-1]
    for d in (chain.get("distractor_pool") or []):
        dw = d.rstrip(".").split()
        if len(dw) == len(words) and dw[-1] != tail and dw[1:-1] == words[1:-1]:
            return " ".join(words[:-1] + [dw[-1]]) + "."
    for d in (chain.get("distractor_pool") or []):
        dw = d.rstrip(".").split()
        if len(dw) >= 3 and dw[-1] != tail:
            return " ".join(words[:-1] + [dw[-1]]) + "."
    return None


def build_trace(chain, n_distractors=12):
    """One chain -> one ordered retrieval trace with a planted echo defect."""
    store = Inspeximus(path=None, embed=None)
    store.semantic_threshold = 10 ** 9                      # lexical: no embedder needed to reproduce
    gold_ids, planted = [], None

    for i, fact in enumerate(chain["gold_facts"]):
        gid = store.remember(fact, tags=["gold"], value=1.0)
        gold_ids.append(gid)
        if i == chain.get("drop_index", 0):
            # THE PLANT, and the first version of it was worthless. It appended "(superseded)" to the
            # text, so the defect carried a LABEL and a one-line grep scored 100% -- identical to the
            # oracle. A benchmark whose planted defect announces itself measures string matching, not
            # detection. The stale record now takes the SAME subject with a DIFFERENT value, borrowed
            # from the distractor pool so it is well-formed and indistinguishable in shape.
            stale = _stale_variant(fact, chain)
            if stale and stale != fact:
                pid = store.remember(stale, tags=["stale"], value=1.0)
                planted = {"kind": "echo_resurrection", "record_id": pid, "replaces": gid,
                           "current_text": fact, "stale_text": stale}

    for d in (chain.get("distractor_pool") or [])[:n_distractors]:
        store.remember(d, tags=["distractor"], value=0.6)

    hits = store.recall(chain["question"], k=K, mode="lexical") or []
    retrieved = []
    for rank, h in enumerate(hits):
        rid = h.get("id")
        kind = ("planted" if planted and rid == planted["record_id"]
                else "gold" if rid in gold_ids else "distractor")
        retrieved.append({"rank": rank, "record_id": rid, "text": h.get("text", ""), "kind": kind})

    return {"chain_id": chain["id"], "query": chain["question"], "answer": chain.get("answer"),
            "retrieved": retrieved,
            "ground_truth": {"gold_ids": gold_ids, "planted": planted}}


# ── reference scorers ────────────────────────────────────────────────────────────────────────────────
def null_score(query, retrieved):
    """Flags nothing. Must produce 0.00 / 0.00 or the harness is inventing credit."""
    return [False] * len(retrieved)


def oracle_score(query, retrieved, _truth=None):
    """Reads the label. Must produce 1.00 / 0.00 or the key itself is broken."""
    return [item.get("_kind_for_oracle") == "planted" for item in retrieved]


def twin_score(query, retrieved):
    """A weak but GENUINE structural detector, and the row that shows what the task actually costs.

    An echo leaves a fingerprint no label is needed to see: two retrieved records that agree on every
    word except the last. That is a contradiction pair -- the same subject asserted with two values --
    and spotting it needs no ground truth.

    What it CANNOT do is say which half of the pair is current. It flags both, so it reaches full
    detection and pays for it in false positives on the twin. That is the honest shape of the problem
    and the reason this row is here: noticing the contradiction is easy, and RESOLVING it is the
    capability under test. A detector that beats this row is doing something real.
    """
    keys = []
    for item in retrieved:
        w = (item.get("text") or "").rstrip(".").split()
        keys.append(tuple(w[:-1]) if len(w) >= 2 else None)
    out = []
    for i, k in enumerate(keys):
        twin = k is not None and any(
            j != i and keys[j] == k
            and (retrieved[j].get("text") or "") != (retrieved[i].get("text") or "")
            for j in range(len(keys)))
        out.append(twin)
    return out


def evaluate(traces, scorer, name, oracle=False):
    tp = fp = planted_total = clean_total = 0
    skipped = 0
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
    return {"scorer": name,
            "detection_rate": (tp / planted_total) if planted_total else None,
            "false_positive_rate": (fp / clean_total) if clean_total else None,
            "planted_items": planted_total, "clean_items": clean_total, "skipped_traces": skipped}


def main():
    n = int(os.environ.get("TRACES", "300"))
    traces = [build_trace(c) for c in _chains(n)]
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    planted_present = sum(1 for t in traces if any(i["kind"] == "planted" for i in t["retrieved"]))
    print("traces exported : %d -> %s" % (len(traces), os.path.basename(OUT)))
    print("traces whose planted record actually surfaced in top-%d: %d" % (K, planted_present))
    print("  (a trace where the plant never surfaced cannot test detection -- counted, not dropped)")
    print()

    rows = [evaluate(traces, null_score, "null (flags nothing)"),
            evaluate(traces, oracle_score, "oracle (reads the label)", oracle=True),
            evaluate(traces, twin_score, "contradiction twin (real)")]

    hdr = "%-30s %10s %10s %9s %8s" % ("scorer", "detection", "false-pos", "planted", "clean")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-30s %9s%% %9s%% %9d %8d"
              % (r["scorer"],
                 "n/a" if r["detection_rate"] is None else "%.1f" % (100 * r["detection_rate"]),
                 "n/a" if r["false_positive_rate"] is None else "%.1f" % (100 * r["false_positive_rate"]),
                 r["planted_items"], r["clean_items"]))
    print()

    null_r, oracle_r = rows[0], rows[1]
    bad = []
    if null_r["detection_rate"] or null_r["false_positive_rate"]:
        bad.append("the NULL scorer scored above zero -- the harness credits flags nobody made")
    if oracle_r["detection_rate"] != 1.0 or oracle_r["false_positive_rate"] != 0.0:
        bad.append("the ORACLE scorer did not reach 1.00/0.00 -- the labelling itself is broken")
    if bad:
        for b in bad:
            print("CONTROL FAILED: %s" % b)
        print("Every number above is measured against a corrupt key. Reporting nothing.")
        return 2
    print("controls: null 0.00/0.00 and oracle 1.00/0.00  [OK] -- the key is sound and the harness")
    print("          credits only flags a scorer actually made.")
    json.dump({"k": K, "traces": len(traces), "planted_surfaced": planted_present, "rows": rows},
              io.open(os.path.join(HERE, "trace_export_result.json"), "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
