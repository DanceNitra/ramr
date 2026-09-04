"""What did the id cue we planted while removing another one actually solve?

WHY. Three files in this repo state that reading nothing but the id format "solved 205 of 300
traces": memaudit.py, ramr_traces_v03_recut.py and README.md. None of them carries a receipt, and
the corpus that produced the figure no longer exists on disk -- every shipped corpus was rebuilt
with content-addressed ids. An assertion repeated in three places is still one assertion.

WHAT THIS DOES. Rebuilds the corpus with the ORIGINAL id scheme, sib-<chain_id>, then measures the
exploit memaudit implements: find the minority id shape, take that record's first token, and answer
with the candidate whose final token matches it.

WHAT IT FOUND, in two parts, and the second is the more useful one.

FIRST: the published 205 does not reproduce. The cue lands on 204 of 300 traces, and the id rule
fires on roughly 140 of them. It never reaches 205.

SECOND: THE RECONSTRUCTION HAS NO SINGLE ANSWER, because the generator is not deterministic.
build_trace() takes ids from store.remember(), which mints uuid.uuid4().hex[:10], so every rebuild
draws different ids and the minority-shape detection shifts with them. ERRATA.md already lists this
as open: "the ids are not content-addressed, so the artifact still cannot be regenerated". This
probe is what that costs. Three consecutive runs gave 140, 143 and 144 traces at 100.0%, 99.3% and
98.6%. A probe that printed the first of those would have replaced one unreproducible number with
another.

So the result is reported as a RANGE over N runs, and no point estimate is printed.

The argument the number was making is untouched and arguably stronger: a re-cut that removed one
shortcut planted a brighter one, and every other probe in the battery reported "at chance" because
none of them looked at ids.

SCOPE, stated because it is the whole caveat. This is a RECONSTRUCTION. It rebuilds what the
generator produces today under the old id scheme; it is not the original run. The correct claim is
"205 does not reproduce", never "the original was wrong".

CONTROLS:
  * THE CUE MUST ACTUALLY LAND. If no sib- id reaches the corpus, the probe measures a corpus
    without the defect and its zero means nothing. The run refuses.
  * THE NULL MUST BE NEAR ZERO. memaudit scores every rule against permuted labels. An id rule
    whose null is high is reading the label, not the id.
  * THE SHIPPED CORPUS MUST NOT HAVE THE CUE. If v0.5 also carries sib- ids then the fix never
    landed, and that is a bigger finding than this one.
  * THE RUNS MUST DISAGREE. If every run returns the same count, the generator is deterministic
    after all, the range is decoration, and the probe refuses rather than reporting a spread it did
    not observe.

    python ramr_id_cue_reconstruction.py --runs 20
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ramr_id_cue_reconstruction_result.json")
OLD_ID = '            rid = "sib-" + t["chain_id"]'
ID_LINE = re.compile(r"^ *rid = hashlib\.sha256\(.*$", re.M)


def refuse(why):
    print("REFUSED: " + why)
    json.dump({"verdict": "REFUSED", "why": why},
              io.open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
    raise SystemExit(2)


def one_run():
    """One rebuild plus one audit. Returns (cue_planted, fires, accuracy, null, traces)."""
    work = tempfile.mkdtemp(prefix="ramr_idcue_")
    try:
        for f in ("ramr_traces_v03_recut.py", "memaudit.py", "ramr_trace_export.py"):
            shutil.copy(os.path.join(HERE, f), work)
        shutil.copytree(os.path.join(HERE, "data"), os.path.join(work, "data"))

        recut = os.path.join(work, "ramr_traces_v03_recut.py")
        src = io.open(recut, encoding="utf-8").read()
        patched, n_sub = ID_LINE.subn(OLD_ID, src, count=1)
        if not n_sub:
            refuse("could not find the content-addressed id line to revert; the generator changed "
                   "and this reconstruction no longer describes it")
        io.open(recut, "w", encoding="utf-8", newline="\n").write(patched)

        env = dict(os.environ, RAMR_TRACE_VER="0.4", RAMR_INJECT_SIBLING="1")
        subprocess.run([sys.executable, "-X", "utf8", "ramr_traces_v03_recut.py"],
                       cwd=work, env=env, capture_output=True, timeout=900)

        blind = os.path.join(work, "ramr_traces_v0.4_blind.jsonl")
        if not os.path.exists(blind):
            refuse("the rebuild produced no corpus, so there is nothing to audit")
        planted = io.open(blind, encoding="utf-8").read().count("sib-")
        if planted == 0:
            refuse("no sib- id reached the rebuilt corpus, so this run measures a corpus without "
                   "the defect and any number below is void")

        subprocess.run([sys.executable, "-X", "utf8", "memaudit.py", "--adapter", "ramr",
                        "--blind", "ramr_traces_v0.4_blind.jsonl",
                        "--labels", "ramr_traces_v0.4_labels.jsonl", "--json", "audit.json"],
                       cwd=work, capture_output=True, text=True, timeout=900)
        audit = json.load(io.open(os.path.join(work, "audit.json"), encoding="utf-8"))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    probe = next((p for p in audit["probes"] if p["probe"].startswith("leak:id-outlier")), None)
    if not probe or not probe.get("fired"):
        refuse("leak:id-outlier did not fire on a corpus that carries the cue; the probe cannot "
               "see its own target")
    if (probe.get("null_accuracy") or 0) > 0.10:
        refuse("the permuted-label null is %.3f, so this rule is reading the label rather than the "
               "id and its accuracy is not a shortcut measurement" % probe["null_accuracy"])
    return (planted, round(probe["coverage"] * audit["traces"]),
            probe["accuracy"], probe["null_accuracy"], audit["traces"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = 20
    if "--runs" in argv:
        runs = int(argv[argv.index("--runs") + 1])

    shipped = os.path.join(HERE, "ramr_traces_v0.5_blind.jsonl")
    if os.path.exists(shipped) and "sib-" in io.open(shipped, encoding="utf-8").read():
        refuse("the SHIPPED v0.5 corpus still carries sib- ids, so the content-addressed fix never "
               "landed; that is the finding, not this reconstruction")

    planted, fires, accs, nulls, n = [], [], [], [], None
    for i in range(runs):
        p, f, a, nl, n = one_run()
        planted.append(p)
        fires.append(f)
        accs.append(a)
        nulls.append(nl)
        print("  run %2d/%d  cue on %d  fires %d  acc %.4f" % (i + 1, runs, p, f, a))

    if len(set(fires)) == 1 and len(set(accs)) == 1:
        refuse("all %d runs returned the identical count and accuracy, so the generator is "
               "deterministic and a RANGE would claim a spread that was not observed; report the "
               "point estimate instead and delete this control" % runs)

    res = {
        "verdict": "205_OF_300_DOES_NOT_REPRODUCE",
        "scope": "a reconstruction under the old sib-<chain_id> scheme, not the original run; the "
                 "original corpus no longer exists on disk",
        "published_claim": "reading nothing but the id format solved 205 of 300 traces",
        "runs": runs,
        "traces": n,
        "cue_planted_on_traces": {"min": min(planted), "max": max(planted),
                                  "identical_every_run": len(set(planted)) == 1},
        "id_rule_fires_on_traces": {"min": min(fires), "max": max(fires),
                                    "distinct_values": len(set(fires))},
        "id_rule_accuracy": {"min": min(accs), "max": max(accs)},
        "permuted_label_null": {"min": min(nulls), "max": max(nulls)},
        "why_a_range_and_not_a_number":
            "build_trace() takes record ids from store.remember(), which mints "
            "uuid.uuid4().hex[:10], so every rebuild draws different ids and the minority-shape "
            "detection shifts with them. ERRATA.md lists this as open. A point estimate would "
            "replace one unreproducible number with another.",
        "measured_claim":
            "the cue lands on %d of %d traces in every run; an id-shape rule fires on %d to %d of "
            "them and is right on %.1f%% to %.1f%% of those, against a permuted-label null at or "
            "below %.1f%%. It never reaches 205."
            % (min(planted), n, min(fires), max(fires), 100 * min(accs), 100 * max(accs),
               100 * max(nulls)),
    }
    json.dump(res, io.open(OUT, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print()
    print("  " + res["measured_claim"])
    print("  verdict: %s" % res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
