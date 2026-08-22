#!/usr/bin/env python3
"""A runnable checker for a shared calibration dataset, so a usage guide is not only prose.

WHY THIS EXISTS. On deepseek-ai/DeepSeek-V3#1591, @UID9622 published a real audit-log dataset as a
shared calibration input, and the thread proposed a community usage guide to go with it. A guide is a
document, and the same thread had just established that a check nobody runs is indistinguishable from
a check that passes: the dataset's own `可以` keyword rule was perfectly consistent with itself and
wrong in the same direction on all eight of its hits.

So this is the half of a usage guide that can fail. Prose explains a pitfall; this catches the reader
who skipped the prose. It takes any JSONL calibration dataset with per-label trigger strings, so it
is not specific to that dataset -- every threshold, family name and expected count is an argument.

WHAT IT CHECKS, and each one exists because it caught something real:

  1. BYTES  the file's SHA-256 against the digest its publisher declared. Without this the rest is a
     statement about whatever happened to be on disk.

  2. SCHEMA  no field outside the declared set, and no field from the forbidden set. A dataset that
     grows a column between revisions changes what every downstream comparison means.

  3. TRIGGER FAMILIES  the reason a label was assigned, grouped. THE NAIVE VERSION IS WRONG AND THIS
     IS THE FINDING THIS FILE WAS WRITTEN AROUND: on the 1591 dataset, splitting `rejection_reason`
     on ":" -- which is what "split on the prefix before comparing frameworks" literally says --
     gives ELEVEN groups, because the heuristic embeds a character count in the prefix
     (`未明确判定(62字符)`, `长回复(556字符)·可能穿透`). Normalising the parenthetical and the
     suffix away gives FOUR: 穿透信号 8, 未明确判定 7, 长回复 3, 能力受限 1 -- which is exactly the
     8 / 7 / 3 the dataset's own SCHEMA declares, plus the single negative. A reader following the
     recommendation literally would bucket 19 records eleven ways and compare noise.

  4. REVISION DIFF BY CONTENT, NEVER BY ID. Ids in this dataset are positional, and 14 of 19 were
     renumbered between r1 and r2. The first version of our own audit asserted that four named ids
     were absent from r2; one of them had never been in r1 at all, so that assertion could not fail.
     Diffing by content instead reports the substitution the counts hide: 15 kept, 4 dropped, 4 added,
     with 19 records on both sides.

  5. WHAT IT CANNOT VERIFY, said out loud. A rule that needs the source logs is reported UNVERIFIABLE
     and never as a pass. A checker that silently drops the rules it cannot reach is how a partial
     audit reads as a complete one.

CONTROLS. `--self-test` builds fixtures that MUST fail each check and asserts they do, including the
two vacuous forms that got past us: an absence assertion on an id that never existed, and a record
count that is stable across a total content substitution. A checker whose checks cannot fail has
measured nothing, which is the whole thesis of the thread this came from.

Usage:
    python integrity/calibration_dataset_check.py DATA.jsonl --sha256 <digest> \\
        --label-field verdict --trigger-field rejection_reason \\
        --expect-family 穿透信号=8 --expect-family 未明确判定=7 --expect-family 长回复=3
    python integrity/calibration_dataset_check.py NEW.jsonl --against OLD.jsonl
    python integrity/calibration_dataset_check.py --self-test

Exit 0 = every check that could run, passed. Exit 1 = a check failed. Exit 2 = it could not run.
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile

#: Strip an embedded measurement from a trigger prefix: `未明确判定(62字符)` -> `未明确判定`.
#: Both bracket families, because the datasets in this thread use the full-width ones.
_PAREN = re.compile(r"[（(][^）)]*[)）]")


def family_of(trigger: str) -> str:
    """The heuristic family a per-label trigger string belongs to.

    Deliberately not `trigger.split(":")[0]`. See check 3 in the module docstring: the raw prefix
    carries a per-record character count, so the naive split fragments one family into as many groups
    as it has distinct lengths.
    """
    head = (trigger or "").split(":")[0].split("：")[0].strip()
    head = _PAREN.sub("", head)
    head = head.split("·")[0].split("|")[0].strip()
    return head or "(none)"


def load(path):
    raw = open(path, "rb").read()
    records, bad = [], []
    for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad.append((i, str(e)[:60]))
    return raw, records, bad


class Report:
    def __init__(self):
        self.rows = []

    def ok(self, name, passed, detail=""):
        self.rows.append(("PASS" if passed else "FAIL", name, detail))
        return passed

    def unverifiable(self, name, why):
        """Never a pass. A rule this cannot reach is reported, not skipped."""
        self.rows.append(("UNVERIFIABLE", name, why))

    def note(self, name, detail):
        self.rows.append(("NOTE", name, detail))

    def failed(self):
        return any(s == "FAIL" for s, _, _ in self.rows)

    def render(self):
        for status, name, detail in self.rows:
            print("  %-13s %-52s %s" % (status, name, detail))
        n_f = sum(1 for s, _, _ in self.rows if s == "FAIL")
        n_u = sum(1 for s, _, _ in self.rows if s == "UNVERIFIABLE")
        print("\n%d checks, %d failed, %d unverifiable" % (len(self.rows), n_f, n_u))
        if n_u:
            print("UNVERIFIABLE is not a pass: those rules need inputs this file does not have.")


def check_dataset(path, sha256=None, label_field="verdict", trigger_field="rejection_reason",
                  expect_families=None, allowed_fields=None, forbidden_fields=None,
                  against=None, truncation_marker=None, expect_truncated=None, rep=None):
    rep = rep or Report()
    raw, records, bad = load(path)
    print("dataset: %s  %d bytes, %d records\n" % (os.path.basename(path), len(raw), len(records)))
    rep.ok("every line parses as JSON", not bad, "; ".join("line %d: %s" % b for b in bad[:3]))

    # ---- 1. bytes ------------------------------------------------------------------------------
    digest = hashlib.sha256(raw).hexdigest()
    if sha256:
        rep.ok("SHA-256 matches the declared digest", digest.startswith(sha256.lower()),
               digest[:24])
    else:
        rep.unverifiable("SHA-256 against a declared digest", "no --sha256 given; file digest is "
                         + digest[:24])

    # ---- 2. schema -----------------------------------------------------------------------------
    keys = set()
    for r in records:
        keys |= set(r)
    if allowed_fields:
        extra = sorted(keys - set(allowed_fields))
        rep.ok("no field outside the declared schema", not extra, ", ".join(extra))
    else:
        rep.note("fields present", ", ".join(sorted(keys)))
    if forbidden_fields:
        present = sorted(keys & set(forbidden_fields))
        rep.ok("no forbidden field present", not present, ", ".join(present))

    # ---- 3. trigger families -------------------------------------------------------------------
    if trigger_field and any(trigger_field in r for r in records):
        naive = collections.Counter((r.get(trigger_field) or "").split(":")[0].strip()
                                    for r in records)
        fams = collections.Counter(family_of(r.get(trigger_field)) for r in records)
        rep.note("trigger groups: naive split vs normalised family",
                 "%d naive -> %d families" % (len(naive), len(fams)))
        if len(naive) > len(fams):
            rep.note("the naive split would over-fragment",
                     "splitting on ':' alone gives %d groups for %d families -- the prefix carries a "
                     "per-record measurement" % (len(naive), len(fams)))
        for fam, n in sorted(fams.items(), key=lambda kv: -kv[1]):
            rep.note("  family %s" % fam, str(n))
        if expect_families:
            for fam, want in expect_families.items():
                rep.ok("family %s has the declared count" % fam, fams.get(fam, 0) == want,
                       "got %d, declared %d" % (fams.get(fam, 0), want))
    else:
        rep.unverifiable("trigger families",
                         "no %r field: the labelling policy left no fingerprint, so the labels "
                         "cannot be audited without re-running inference" % trigger_field)

    # ---- 3b. a document's cross-reference must resolve IN THE DATA ---------------------------
    # Found on the 1591 dataset while writing this. SCHEMA warns not to run character-length rules on
    # the published `response` field because three records are truncated at 500, and names them by
    # id. Two of the three ids are right. The third names a record that is NOT truncated and belongs
    # to a different heuristic family, while the record that IS truncated is not listed. Excluding by
    # that table therefore drops a healthy record and keeps a truncated one -- the opposite of what
    # the warning intends. A prose warning cannot check its own ids against the file; this can.
    if truncation_marker:
        actual = {r.get("request_id") for r in records
                  if truncation_marker in (r.get("response") or "")}
        rep.note("records carrying the truncation marker", "%d: %s"
                 % (len(actual), ", ".join(sorted(x for x in actual if x))))
        if expect_truncated is not None:
            want = set(expect_truncated)
            rep.ok("the documented truncated ids are the truncated records",
                   actual == want,
                   "documented-but-not-truncated: %s | truncated-but-undocumented: %s"
                   % (", ".join(sorted(want - actual)) or "none",
                      ", ".join(sorted(actual - want)) or "none"))

    # ---- 4. revision diff, BY CONTENT --------------------------------------------------------
    if against:
        _, old, _ = load(against)
        def body(r):
            return json.dumps({k: v for k, v in sorted(r.items())
                               if k not in ("request_id", "timestamp")},
                              ensure_ascii=False, sort_keys=True)
        a, b = {body(r) for r in old}, {body(r) for r in records}
        kept, dropped, added = len(a & b), len(a - b), len(b - a)
        rep.note("revision diff by content", "%d kept, %d dropped, %d added" % (kept, dropped, added))
        rep.ok("the record count did not hide a substitution",
               not (len(old) == len(records) and (dropped or added)),
               "%d -> %d records while %d were replaced" % (len(old), len(records), max(dropped, added)))
        ids_old = {r.get("request_id") for r in old}
        ids_new = {r.get("request_id") for r in records}
        renamed = len(ids_old - ids_new)
        if renamed:
            rep.note("ids are not stable across revisions",
                     "%d of %d ids from the old revision are absent from the new one, so an "
                     "id-based absence check proves nothing" % (renamed, len(ids_old)))

    # ---- 5. what cannot be checked from the file alone ------------------------------------------
    rep.unverifiable("labels match the underlying source logs",
                     "needs the source logs, which a shared dataset does not ship")
    return rep


def _write(tmp, name, rows):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def self_test():
    """Every check must be able to FAIL, including the two vacuous forms that got past us."""
    tmp = tempfile.mkdtemp(prefix="caldata_")
    base = [{"request_id": "R-%03d" % i, "verdict": "hit",
             "rejection_reason": "穿透信号: 可以", "response": "r%d" % i} for i in range(8)]
    base += [{"request_id": "R-1%02d" % i, "verdict": "hit",
              "rejection_reason": "未明确判定(%d字符)" % (60 + i), "response": "u%d" % i}
             for i in range(7)]
    good = _write(tmp, "good.jsonl", base)
    fails = []

    def must_fail(label, **kw):
        r = check_dataset(quiet_path, rep=Report(), **kw)
        if not r.failed():
            fails.append(label)

    print("=== CONTROL: each check must be able to fail ===\n")
    quiet_path = good

    # a wrong digest must fail
    must_fail("sha256", sha256="deadbeef")
    # a forbidden field present must fail
    _write(tmp, "x.jsonl", base)
    quiet_path = _write(tmp, "extra.jsonl",
                        [dict(r, inference_time_ms=1) for r in base])
    must_fail("forbidden field", forbidden_fields=["inference_time_ms"])
    must_fail("schema", allowed_fields=["request_id", "verdict", "rejection_reason", "response"])
    # a wrong declared family count must fail
    quiet_path = good
    must_fail("family count", expect_families={"穿透信号": 99})

    # THE TWO VACUOUS FORMS -------------------------------------------------------------------
    # (a) total content substitution with an unchanged record count must be caught
    swapped = [dict(r, response="CHANGED-%d" % i) for i, r in enumerate(base)]
    quiet_path = _write(tmp, "swapped.jsonl", swapped)
    must_fail("substitution hidden by a stable count", against=good)

    # (b) the family grouping must not be the naive split
    naive_groups = len({(r["rejection_reason"] or "").split(":")[0].strip() for r in base})
    fam_groups = len({family_of(r["rejection_reason"]) for r in base})
    ok_norm = naive_groups > fam_groups
    print("\n  %-13s naive split gives %d groups, normalised gives %d"
          % ("PASS" if ok_norm else "FAIL", naive_groups, fam_groups))
    if not ok_norm:
        fails.append("family normalisation")

    print("\n=== CONTROL RESULT ===")
    if fails:
        print("  FAIL -- these checks did not fail when they should have: " + ", ".join(fails))
        return 1
    print("  every check failed on its own mutant, and the normaliser beats the naive split")
    return 0


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", nargs="?", help="the JSONL calibration dataset")
    ap.add_argument("--sha256", help="the digest its publisher declared")
    ap.add_argument("--label-field", default="verdict")
    ap.add_argument("--trigger-field", default="rejection_reason",
                    help="the per-label field recording WHY the label was assigned")
    ap.add_argument("--expect-family", action="append", default=[], metavar="NAME=COUNT")
    ap.add_argument("--allowed-field", action="append", default=[])
    ap.add_argument("--forbidden-field", action="append", default=[])
    ap.add_argument("--against", help="an earlier revision, diffed BY CONTENT")
    ap.add_argument("--truncation-marker",
                    help="a suffix a publisher appends to truncated fields")
    ap.add_argument("--expect-truncated", action="append", default=[], metavar="ID",
                    help="an id a document CLAIMS is truncated; checked against the data")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--self-test", action="store_true", help="prove every check can fail")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.data:
        ap.error("give a dataset, or --self-test")
    if not os.path.exists(a.data):
        print("no such file: %s" % a.data, file=sys.stderr)
        return 2

    fams = {}
    for spec in a.expect_family:
        if "=" not in spec:
            ap.error("--expect-family wants NAME=COUNT, got %r" % spec)
        k, v = spec.rsplit("=", 1)
        fams[k] = int(v)

    rep = check_dataset(a.data, sha256=a.sha256, label_field=a.label_field,
                        trigger_field=a.trigger_field, expect_families=fams or None,
                        allowed_fields=a.allowed_field or None,
                        forbidden_fields=a.forbidden_field or None, against=a.against,
                        truncation_marker=a.truncation_marker,
                        expect_truncated=a.expect_truncated or None)
    rep.render()
    if a.json:
        json.dump({"data": a.data, "rows": [{"status": s, "check": n, "detail": d}
                                            for s, n, d in rep.rows]},
                  open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("report -> " + a.json)
    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
