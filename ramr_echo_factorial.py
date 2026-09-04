"""What actually produces the 100%? A factorial ablation of our own headline number.

THE DEFECT THIS REMOVES. `ramr_echo_resistance_backends.py` reports inspeximus at 100% and a guard-off
control at 0%, and we have been reading that as "recording the supersession link is what defends".
It is not what the harness shows. Both arms pass `key=f"{ent}::r"` on every write, so provenance is
HELD CONSTANT and the only variable is `echo_guard`. A third factor sits in the stored text: the
superseding record is written as `correction: {ent} region is {new}`, and a literal tag telling the
reader which record is the correction is worth 10.7 points to a model choosing between the two
(measured, ramr_echo_prefix_ablation.py).

So the published number is a sum of three effects reported as one, and the same is true of every
end-to-end score in this category. Nobody publishes the decomposition, including us.

THE DESIGN. Three binary factors, eight cells, one judge held identical across all of them:

    key      supersession key passed on write   -> the store can resolve; recall collapses to one record
    guard    echo_guard on                      -> a re-asserted stale value is suppressed
    tag      "correction:" prefix in the text   -> the reader is told which one is the correction

Everything else -- entities, values, order, judge model, temperature, query -- is fixed. What varies is
only what we claim to be measuring, which is the property the original harness lacked.

WHAT A RESULT LOOKS LIKE. If `key` alone carries the effect, the mechanism is supersession and the
guard is insurance. If `guard` carries it, our defence is the echo suppressor and the link is
necessary-but-not-sufficient. If `tag` carries a large share, then a chunk of every number in this
space -- ours and our competitors' -- is a model reading a label, not a store resolving a conflict.

    python ramr_echo_factorial.py --key-file ../agora/server/.env --n 30
"""
import argparse
import io
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus  # noqa: E402
import ramr_echo_resistance_backends as E  # noqa: E402
from ramr_content_resolver import DEFAULT_BASE, DEFAULT_MODEL, read_key  # noqa: E402

OUT = "ramr_echo_factorial_result.json"


def build(ent, old, new, use_key, guard, tag, echo=True):
    """One trial's store, then the texts a reader would see. Deterministic, no network."""
    m = Inspeximus(path=None, embed=None)
    m.semantic_threshold = 10 ** 9                       # lexical: reproducible without an embedder
    m.echo_guard = guard
    k = ("%s::r" % ent) if use_key else None
    cur = ("correction: %s region is %s" % (ent, new)) if tag else ("%s region is %s" % (ent, new))
    m.remember("%s region is %s" % (ent, old), key=k, object=old if use_key else None, value=5.0)
    m.remember(cur, key=k, object=new if use_key else None, value=1.0)
    if echo:                                             # the attack: re-assert the stale value, loudly
        m.remember("just to confirm, %s stays in %s" % (ent, old),
                   key=k, object=old if use_key else None, value=5.0)
    return [h.get("text", "") for h in (m.recall("%s region" % ent, k=5, mode="lexical") or [])]


def main(argv=None):
    ap = argparse.ArgumentParser(description="factorial ablation of the echo-resistance number")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)

    api = read_key(a.key_file)
    if not api:
        print("no API key found -- pass --key-file")
        return 2
    E.KEY, E.BASE, E.JUDGE = api, a.base.rstrip("/"), a.model

    trials = [(E.ENTS[i], E.OLD[i], E.NEW[i]) for i in range(min(a.n, len(E.ENTS)))
              if E.OLD[i] != E.NEW[i]]
    cells = list(itertools.product([True, False], repeat=3))    # key, guard, tag
    print("%d trials x %d cells = %d judged reads, model %s\n" % (len(trials), len(cells),
                                                                  len(trials) * len(cells), a.model))

    jobs = [(k, g, t, ent, old, new) for (k, g, t) in cells for (ent, old, new) in trials]
    done = [0]
    t0 = time.time()

    def run(job):
        use_key, guard, tag, ent, old, new = job
        texts = build(ent, old, new, use_key, guard, tag)
        # THE SAME JUDGE IN EVERY CELL. Swapping in a deterministic scorer where recall returns one
        # record would change the instrument between conditions -- the exact confound being removed.
        verdict = E.judge(ent, texts, old, new)
        done[0] += 1
        if done[0] % 40 == 0:
            print("  %d/%d  %.0fs" % (done[0], len(jobs), time.time() - t0), flush=True)
        return (use_key, guard, tag), verdict, len(texts)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run, jobs))

    agg = {}
    for cell, verdict, nrec in results:
        d = agg.setdefault(cell, {"new": 0, "old": 0, "other": 0, "none": 0, "records": []})
        d[verdict if verdict in ("new", "old", "other") else "none"] += 1
        d["records"].append(nrec)

    print("\n  key  guard  tag |  keeps correction | records surfaced")
    print("  " + "-" * 58)
    rows = []
    for cell in cells:
        d = agg.get(cell, {})
        n = sum(d.get(x, 0) for x in ("new", "old", "other", "none"))
        if not n:
            continue
        rate = d["new"] / n
        recs = sum(d["records"]) / len(d["records"])
        rows.append({"key": cell[0], "guard": cell[1], "tag": cell[2],
                     "keeps_correction": round(rate, 3), "mean_records": round(recs, 2), "n": n})
        print("  %-4s %-6s %-4s| %14.1f%% | %.2f"
              % (str(cell[0])[0], str(cell[1])[0], str(cell[2])[0], 100 * rate, recs))

    def main_effect(idx, name):
        on = [r["keeps_correction"] for r in rows if list(r.values())[idx]]
        off = [r["keeps_correction"] for r in rows if not list(r.values())[idx]]
        if on and off:
            eff = 100 * (sum(on) / len(on) - sum(off) / len(off))
            print("  %-6s main effect: %+.1f pp" % (name, eff))
            return eff
        return None

    print()
    effects = {n: main_effect(i, n) for i, n in ((0, "key"), (1, "guard"), (2, "tag"))}
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"model": a.model, "trials": len(trials), "cells": rows,
                    "main_effects_pp": effects}, indent=2) + "\n")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
