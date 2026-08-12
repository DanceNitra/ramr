"""Do the PUBLISHED trace artifacts agree with each other? They did not, and nothing was checking.

THE DEFECT. `ramr_traces_v0.2_blind.jsonl` is produced from `ramr_traces_v0.1.jsonl` by shuffling each
trace and moving the labels into a separate file. But v0.1 was regenerated locally and never
committed, so the pair that shipped came from two different generations: same 300 queries, and NOT ONE
of the 300 traces holding the same record_ids. A third party joining v0.2 back to v0.1 -- to recover a
`kind`, to check the plant, to reproduce the shuffle -- matches nothing, in every trace.

WHY IT COULD HAPPEN AT ALL. `build_trace()` mints record_ids through `store.remember()`, which issues a
fresh id per run. The texts and the queries are read from the corpus in order and are stable; the IDS
ARE NOT. So re-running the export produces an artifact that is semantically the same and byte-wise
unjoinable to the one before it, and no number changes to tell you. For a project whose stated moat is
determinism, an artifact that cannot be regenerated is the wrong kind of quiet.

WHAT THIS CHECKS, and it is deliberately about the FILES ON DISK rather than about a fresh run: a fresh
run cannot detect this, because a fresh run rebuilds both sides from one generation and they agree by
construction. That is exactly how it stayed invisible.

    python ramr_traces_pair_consistency.py        # exit 0 = the published pair joins

Run it before publishing any trace artifact.
"""
import io
import json
import os
import sys

V01 = "ramr_traces_v0.1.jsonl"
BLIND = "ramr_traces_v0.2_blind.jsonl"
LABELS = "ramr_traces_v0.2_labels.jsonl"


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def main():
    missing = [p for p in (V01, BLIND, LABELS) if not os.path.exists(p)]
    if missing:
        print("MISSING: %s" % ", ".join(missing))
        return 2

    v01, blind, labels = load(V01), load(BLIND), load(LABELS)
    label_of = {t["chain_id"]: t["ground_truth"] for t in labels}
    print("v0.1 %d traces | v0.2 blind %d | v0.2 labels %d" % (len(v01), len(blind), len(labels)))

    problems = []

    ids01 = {t["chain_id"]: sorted(i["record_id"] for i in t["retrieved"]) for t in v01}
    ids02 = {t["chain_id"]: sorted(i["record_id"] for i in t["retrieved"]) for t in blind}
    if set(ids01) != set(ids02):
        problems.append("v0.1 and v0.2 cover different chain_ids (%d vs %d)"
                        % (len(ids01), len(ids02)))
    shared = set(ids01) & set(ids02)
    mismatched = sorted(c for c in shared if ids01[c] != ids02[c])
    print("traces whose record_ids do NOT join v0.1 -> v0.2 : %d of %d" % (len(mismatched), len(shared)))
    if mismatched:
        problems.append("%d of %d traces hold different record_ids in v0.1 and v0.2; the published "
                        "pair came from two different generations and cannot be joined"
                        % (len(mismatched), len(shared)))
        print("  first: %s" % mismatched[0])

    # The labels must reference ids that EXIST in the blind file, or an external scorer cannot be
    # graded at all. This is the half that was self-consistent, and it is checked so it stays that way.
    dangling = 0
    for t in blind:
        present = {i["record_id"] for i in t["retrieved"]}
        gt = label_of.get(t["chain_id"]) or {}
        wanted = [(gt.get("planted") or {}).get("record_id")] + list(gt.get("gold_ids") or [])
        dangling += sum(1 for w in wanted if w and w not in present and w != (gt.get("planted") or {}).get("replaces"))
    print("label ids referencing a record absent from the blind file : %d" % dangling)
    if dangling:
        problems.append("%d label references point at records not present in the blind artifact"
                        % dangling)

    q01 = {t["chain_id"]: t["query"] for t in v01}
    q02 = {t["chain_id"]: t["query"] for t in blind}
    same_q = all(q01.get(c) == q02.get(c) for c in shared)
    print("queries identical across the pair : %s" % same_q)
    if not same_q:
        problems.append("the two versions do not even pose the same questions")

    print()
    if problems:
        print("FAIL -- the published pair does not join:")
        for p in problems:
            print("  * %s" % p)
        return 1
    print("OK -- v0.1 and v0.2 are the same generation and the labels resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
