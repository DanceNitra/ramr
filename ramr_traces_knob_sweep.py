"""Sweep TRACES, the one exposed knob the published run never varied.

The construction gate refused the trace export with UNSWEPT-KNOB: `TRACES` is settable and every
published number came from a single value of it. That refusal is correct even when the intuition is
right -- "the positional invariant is a property of the generator, not of the sample" is a belief
until it is three measurements.

Sweeping it also found a defect the intuition would never have surfaced: the corpus holds 300 chains,
so TRACES=600 silently produced 300. A knob that returns less than it was asked for, without saying
so, is the same silent-success shape this repository keeps finding.

    python ramr_traces_knob_sweep.py
"""
import io
import json
import os
import shutil
import subprocess
import sys

SIZES = (50, 150, 300, 600)
OUT = "ramr_traces_v0.1.jsonl"
BACKUP = OUT + ".sweep-backup"


def invariant(path):
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    before = pairs = 0
    for r in rows:
        p = r["ground_truth"]["planted"]
        idx = {i["record_id"]: i["rank"] for i in r["retrieved"]}
        if p["record_id"] in idx and p.get("replaces") in idx:
            pairs += 1
            before += idx[p["record_id"]] < idx[p["replaces"]]
    leak = any("kind" in i for r in rows for i in r["retrieved"])
    return len(rows), before, pairs, leak


def main():
    if not os.path.exists(OUT):
        print("no %s to protect; run the export first" % OUT)
        return 1
    shutil.copy2(OUT, BACKUP)          # the published artifact is restored in `finally`, always
    rows = []
    try:
        for n in SIZES:
            env = {**os.environ, "TRACES": str(n), "PYTHONIOENCODING": "utf-8"}
            subprocess.run([sys.executable, "-X", "utf8", "ramr_trace_export.py"],
                           capture_output=True, text=True, timeout=600, env=env, check=True)
            got, before, pairs, leak = invariant(OUT)
            pct = 100.0 * before / pairs if pairs else 0.0
            short = got < n
            print("  TRACES=%-4d produced %-4d  positional-invariant %3d/%-3d = %5.1f%%  "
                  "kind-leak=%s%s" % (n, got, before, pairs, pct, leak,
                                      "   <-- CORPUS EXHAUSTED" if short else ""))
            rows.append({"asked": n, "produced": got, "invariant": [before, pairs],
                         "pct": pct, "kind_field_present": leak, "corpus_exhausted": short})
    finally:
        shutil.copy2(BACKUP, OUT)
        os.remove(BACKUP)
        restored, _b, _p, _l = invariant(OUT)
        print("\n  published artifact restored: %s holds %d traces" % (OUT, restored))

    pcts = {r["pct"] for r in rows}
    stable = len(pcts) == 1 and pcts == {100.0}
    print("\nFINDINGS")
    print("  the invariant is INDEPENDENT of sample size: %s" % ("yes, 100%% at every size" if stable
                                                                 else "no -- it varies: %s" % sorted(pcts)))
    exhausted = [r for r in rows if r["corpus_exhausted"]]
    if exhausted:
        print("  the knob CAPS: %s asked for more than the corpus holds and got %d without saying so"
              % ([r["asked"] for r in exhausted], exhausted[0]["produced"]))
        print("  (the export now prints a NOTE when this happens -- that is the fix this sweep bought)")
    io.open("ramr_traces_knob_sweep_result.json", "w", encoding="utf-8").write(
        json.dumps({"sizes": SIZES, "rows": rows, "invariant_independent_of_n": stable}, indent=2))
    print("\nwrote ramr_traces_knob_sweep_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
