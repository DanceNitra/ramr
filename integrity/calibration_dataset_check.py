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

  3. TRIGGER FAMILIES  the reason a label was assigned, grouped. This is plumbing, not a discovery:
     SCHEMA already states the answer (三组, over the 18 `confirmed_penetration` records) and already
     writes the patterns with N as a placeholder -- `未明确判定(N字符)`, `长回复(N字符)·可能穿透`.
     The function exists because the field carries a per-record measurement inside the prefix, so a
     literal `split(":")` returns 11 groups where the document says 3, and any tool that regroups
     these records needs the normalisation written down once rather than re-invented per reader.
     On v1.0 it yields 穿透信号 8, 未明确判定 7, 长回复 3, plus 能力受限 1 for the single
     `firewall_deny` record -- the 8 / 7 / 3 SCHEMA declares.

  3b. A DOCUMENT'S CROSS-REFERENCE MUST RESOLVE IN THE DATA. THIS IS THE CHECK THAT FOUND SOMETHING.
     SCHEMA §2 warns readers not to run length-based rules on the published `response` field,
     because three records were truncated at 500 characters, and names them by id: REQ-55072cb7-001
     (556), REQ-c9613162-002 (635), REQ-082959a1-003 (665). The records themselves carry the marker
     `...[truncated:500chars]`, so the claim is checkable against the file, and the third id is
     wrong. The 665-character record is REQ-b59745a2-005. REQ-082959a1-003 is 98 characters, carries
     no marker, and belongs to the 穿透信号 family -- one of the 8. So excluding by that table drops
     a healthy positive and keeps a truncated record: wrong in both directions at once. Everything
     else in the rule is right, including the 3-242 character spread it documents.

  4. REVISION DIFF BY CONTENT, NEVER BY ID. Ids in this dataset are positional, and 14 of 19 were
     renumbered between r1 and r2 of v1.1-negative. The first version of our own audit asserted that
     four named ids were absent from r2; one of them had never been in r1 at all, so that assertion
     could not fail. Diffing by content instead reports the substitution the counts hide: 15 kept,
     4 dropped, 4 added, with 19 records on both sides.

  5. WHAT IT CANNOT VERIFY, said out loud. A rule that needs the source logs is reported UNVERIFIABLE
     and never as a pass, and a file with no records reports UNVERIFIABLE rather than a vacuous PASS.
     A checker that silently drops the rules it cannot reach is how a partial audit reads as complete.

CONTROLS. `--self-test` builds mutants that MUST fail, and asserts each fails ON ITS OWN ROW -- not
merely that something failed. It also asserts the clean fixture passes, that an empty file produces
no PASS at all, and that the family normaliser survives a colon nested inside a parenthetical (the
bug this file shipped with: `模型明确拒绝（响应含拒绝话术: strong:我无法）` returned a family with a
dangling bracket until the strip/split order was swapped). One control is a negative control on the
harness itself: a fixture where an UNRELATED row fails, which a loose "some check failed" assertion
would wrongly accept. Mutation score 6/6 -- reverting any of the six fixes above turns the self-test
red. A checker whose checks cannot fail has measured nothing, which is the thesis of the thread this
came from, and two of the six defects above were live in this file when it was first committed.

Usage:
    python integrity/calibration_dataset_check.py DATA.jsonl --sha256 <digest> \\
        --expect-family 穿透信号=8 --expect-family 未明确判定=7 --expect-family 长回复=3
    # the check that found the id error -- exits 1 on longhun v1.0:
    python integrity/calibration_dataset_check.py DATA.jsonl \\
        --truncation-marker "[truncated:500chars]" \\
        --expect-truncated REQ-55072cb7-001 --expect-truncated REQ-c9613162-002 \\
        --expect-truncated REQ-082959a1-003
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
    # Order matters and it is the bug this function shipped with: a parenthetical may itself
    # contain a colon (`模型明确拒绝（响应含拒绝话术: strong:我无法）`). Splitting first cuts the
    # bracket in half, the regex can no longer match its own opening bracket, and the family comes
    # back as `模型明确拒绝（响应含拒绝话术` with a dangling paren. Strip parentheticals first.
    head = _PAREN.sub("", trigger or "")
    head = head.split(":")[0].split("：")[0]
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

    def failed_names(self):
        return [n for s, n, _ in self.rows if s == "FAIL"]

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
    if not records:
        # A vacuous pass is the failure mode this whole file exists to catch, so it does not get to
        # happen here: zero records means every downstream check is trivially satisfied.
        rep.unverifiable("every line parses as JSON",
                         "no records: %d bytes, %d unparseable lines -- nothing to check"
                         % (len(raw), len(bad)))
    else:
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
    """Every check must be able to FAIL -- and must fail on ITS OWN row, not merely somewhere.

    An earlier version of this function asserted only that *some* check failed, which would have
    passed a mutant that broke an unrelated row. That is the same vacuous shape the tool is written
    against, so the control now names the row it expects.
    """
    tmp = tempfile.mkdtemp(prefix="caldata_")
    base = [{"request_id": "R-%03d" % i, "verdict": "hit",
             "rejection_reason": "穿透信号: 可以", "response": "r%d" % i} for i in range(8)]
    base += [{"request_id": "R-1%02d" % i, "verdict": "hit",
              "rejection_reason": "未明确判定(%d字符)" % (60 + i), "response": "u%d" % i}
             for i in range(7)]
    good = _write(tmp, "good.jsonl", base)
    fails = []

    def must_fail(label, path, expect_row, **kw):
        """The named check must FAIL on this mutant, and it must be the named check."""
        r = check_dataset(path, rep=Report(), **kw)
        hit = [n for n in r.failed_names() if expect_row in n]
        if not hit:
            fails.append("%s (failed rows: %s)" % (label, r.failed_names() or "none"))

    def must_not_fail(label, path, **kw):
        """The clean fixture must not fail: a control that fires on everything measures nothing."""
        r = check_dataset(path, rep=Report(), **kw)
        if r.failed():
            fails.append("%s -- clean fixture failed: %s" % (label, r.failed_names()))

    print("=== CONTROL: each check must be able to fail, on its own row ===\n")

    # NEGATIVE CONTROL ON THE HELPER ITSELF. Every mutant below happens to break only its own row,
    # so "some check failed" and "the NAMED check failed" agree on all of them -- which means those
    # mutants cannot tell a row-level assertion from a loose one. This case can: a bad digest makes
    # the sha row fail while the family row is correct, so a loose helper accepts it and a strict
    # one does not. Mutation-tested: loosening must_fail to `hit = r.failed_names()` fails here.
    _before = len(fails)
    must_fail("HELPER-DISCRIMINATION", good, "has the declared count",
              sha256="deadbeef", expect_families={"穿透信号": 8})
    if len(fails) == _before:
        fails.append("must_fail does not check WHICH row failed: it accepted an unrelated failure")
    else:
        fails.pop()   # the helper complained, as it must -- that is the pass, not a failure

    # the clean fixture must pass every check it is given -- otherwise the mutants prove nothing
    must_not_fail("clean baseline", good,
                  allowed_fields=["request_id", "verdict", "rejection_reason", "response"],
                  forbidden_fields=["inference_time_ms"],
                  expect_families={"穿透信号": 8, "未明确判定": 7})

    must_fail("sha256", good, "SHA-256", sha256="deadbeef")

    extra = _write(tmp, "extra.jsonl", [dict(r, inference_time_ms=1) for r in base])
    must_fail("forbidden field", extra, "forbidden", forbidden_fields=["inference_time_ms"])
    must_fail("schema", extra, "outside the declared schema",
              allowed_fields=["request_id", "verdict", "rejection_reason", "response"])

    must_fail("family count", good, "has the declared count", expect_families={"穿透信号": 99})

    # THE VACUOUS FORM THAT GOT PAST US: a total content substitution with an unchanged count.
    swapped = _write(tmp, "swapped.jsonl",
                     [dict(r, response="CHANGED-%d" % i) for i, r in enumerate(base)])
    must_fail("substitution hidden by a stable count", swapped, "hide a substitution", against=good)

    # A DOCUMENT'S CROSS-REFERENCE THAT DOES NOT RESOLVE IN THE DATA. This is the check that found
    # the real defect, so it gets a mutant in both directions: an id documented as truncated that
    # is not, and a truncated record the document does not list.
    trunc = _write(tmp, "trunc.jsonl",
                   [dict(r, response=r["response"] + ("...[cut]" if i in (0, 1) else ""))
                    for i, r in enumerate(base)])
    must_fail("documented id is not truncated", trunc, "documented truncated ids",
              truncation_marker="...[cut]",
              expect_truncated=["R-000", "R-002"])          # R-002 is not truncated, R-001 is
    must_not_fail("truncation table that does resolve", trunc,
                  truncation_marker="...[cut]", expect_truncated=["R-000", "R-001"])

    # A FILE WITH NOTHING IN IT MUST NOT REPORT A PASS.
    empty = os.path.join(tmp, "empty.jsonl")
    open(empty, "w").close()
    r = check_dataset(empty, rep=Report())
    if any(st == "PASS" for st, _, _ in r.rows):
        fails.append("empty file produced a PASS: %s"
                     % [n for st, n, _ in r.rows if st == "PASS"])

    # THE NORMALISER MUST BEAT THE NAIVE SPLIT, AND MUST SURVIVE A COLON INSIDE A PARENTHETICAL.
    naive_groups = len({(r["rejection_reason"] or "").split(":")[0].strip() for r in base})
    fam_groups = len({family_of(r["rejection_reason"]) for r in base})
    if not naive_groups > fam_groups:
        fails.append("family normalisation (naive %d, normalised %d)" % (naive_groups, fam_groups))
    nested = "模型明确拒绝（响应含拒绝话术: strong:我无法）"
    if family_of(nested) != "模型明确拒绝":
        fails.append("colon inside a parenthetical -> %r" % family_of(nested))

    print("\n=== CONTROL RESULT ===")
    if fails:
        print("  FAIL -- these controls did not behave: " + "; ".join(fails))
        return 1
    print("  every check failed on its own row, the clean fixture passed, an empty file produced no"
          "\n  PASS, and the normaliser survives a colon nested inside a parenthetical")
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
