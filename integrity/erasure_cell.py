#!/usr/bin/env python3
"""Erasure-correctness cells: does a right-to-erasure request reach exactly the right records?

    python integrity/erasure_cell.py --systems inspeximus --n 20
    python integrity/erasure_cell.py --systems inspeximus --versions 1.86.0,1.87.0,1.88.0

TWO CELLS, and they are the two directions of one soundness/completeness pair. A store can fail either way
and the failures look nothing alike:

  OVER-ERASURE   a request naming a subject the store has never seen deletes somebody else's records.
                 The victim is a bystander who never asked for anything.
  UNDER-ERASURE  a request naming a real subject leaves their data behind -- and the residue that matters
                 is the CURRENT value, because a store that corrects a fact and then erases only the
                 superseded copy has kept the live one.

WHY DETERMINISTIC. Both cells read whether a planted marker is present in the store's own recall surface
after that store's own erasure call. No LLM decides the outcome, so the result cannot be moved by choosing
a better or worse judge -- the same property the revert and echo cells were built for, and easier to hold
here because absence is checkable.

WHY THROUGH THE PUBLIC API. Every adapter must write and erase through the interface the project documents
for its users, never through internals. That is not a stylistic preference: measured on inspeximus 1.86.0,
a store written through its own MCP surface answered `would_erase = 0` to every phrasing of the subject
while the identical write through the library answered 1. A harness that reaches past the public surface
measures a component nobody ships.

WHAT THIS IS NOT. It does not test at-rest security, free-space residue, or backups -- see
erasure_selfcheck.py for the on-disk question. It reads the RECALL SURFACE, which is what an application
serves to its users and its model.

PRIOR WORK, cited because this is not the first look at the problem: DELF (Cohn-Gordon et al., USENIX
Security 2020) built deletion annotations at Facebook precisely because "developer mistakes lead to ...
retaining data that should be deleted, deleting the wrong data"; GateMem (arXiv:2606.18829) and MemLeak
(arXiv:2606.29788) measure residual recoverability after deletion requests. Both are preprints and both
score METHODS. What is not published anywhere is a scored comparison of named shipping PRODUCTS, which is
the only thing these cells add.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

#: (subject, their value, an unrelated bystander subject, the bystander's value)
CASES = [
    ("hr/alice", "5 Elm Street", "hr/bob", "9 Oak Avenue"),
    ("crm/dana", "dana@example.com", "crm/erik", "erik@example.com"),
    ("hr/frida", "+44 7700 900123", "hr/gustav", "+44 7700 900456"),
    ("crm/hana", "account 4471-9920", "crm/ivan", "account 5582-3310"),
    ("hr/jonas", "12 Birch Lane", "hr/klara", "44 Cedar Road"),
    ("crm/lena", "lena.b@example.org", "crm/milan", "milan.k@example.org"),
    ("hr/nadia", "NI QQ123456C", "hr/oskar", "NI ZZ654321B"),
    ("crm/petra", "card ending 4417", "crm/quentin", "card ending 8823"),
    ("hr/radek", "7 Willow Court", "hr/sofia", "31 Maple Way"),
    ("crm/tomas", "tomas@example.net", "crm/ulrika", "ulrika@example.net"),
    ("hr/viktor", "+420 601 234 567", "hr/wanda", "+420 602 765 432"),
    ("crm/xenia", "policy 88-2201", "crm/yusuf", "policy 77-4410"),
    ("hr/zuzana", "19 Aspen Close", "hr/adam", "5 Rowan Street"),
    ("crm/bea", "bea.m@example.com", "crm/cyril", "cyril.n@example.com"),
    ("hr/dita", "passport X8823194", "hr/emil", "passport Y1140228"),
    ("crm/filip", "invoice INV-9081", "crm/gita", "invoice INV-4417"),
    ("hr/honza", "22 Poplar Drive", "hr/iva", "8 Hazel Grove"),
    ("crm/jakub", "jakub@example.io", "crm/kata", "kata@example.io"),
    ("hr/lukas", "+43 660 1234567", "hr/marta", "+43 660 7654321"),
    ("crm/nela", "ref 5590-2213", "crm/otto", "ref 3320-8871"),
]

#: The correction applied before erasure in the under-erasure cell. The subject's CURRENT value is what a
#: DSAR must reach, and it is the copy a naive "erase the record we matched" implementation misses.
NEW_VALUES = ["11 Chestnut Row", "dana.new@example.com", "+44 7700 900999", "account 6693-1102",
              "88 Sycamore Street", "lena.new@example.org", "NI AA112233D", "card ending 9902",
              "3 Juniper Lane", "tomas.new@example.net", "+420 603 111 222", "policy 99-3312",
              "40 Larch Avenue", "bea.new@example.com", "passport Z5540117", "invoice INV-7723",
              "16 Elder Street", "jakub.new@example.io", "+43 660 9998887", "ref 7710-4432"]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # max(0.0, ...) because at k=0 the arithmetic yields -0.0, and a published interval whose lower bound
    # reads "-0.0" invites the reader to distrust every other number on the page.
    return (round(max(0.0, (c - m) / d), 3), round(min(1.0, (c + m) / d), 3))


# ── the adapter interface ────────────────────────────────────────────────────────────────────────
class ErasureAdapter:
    """Four methods, all of which MUST go through the interface the project documents for its users.

    A system that cannot express `erase_subject` at all is not a failure to be scored -- it is a system
    without the capability, and it is reported as `capability: absent` rather than given a zero. Scoring a
    missing feature as a bad score is how a board flatters whoever happens to have built that feature.
    """

    name = "abstract"
    capability = "present"

    def reset(self): ...
    def write(self, text, subject): ...        # store text attributed to `subject`
    def correct(self, text, subject, key): ... # supersede the earlier value for the same subject/key
    def erase_subject(self, subject): ...      # the system's own right-to-erasure call
    def surface(self):                         # everything the store would serve, as text
        ...


class InspeximusAdapter(ErasureAdapter):
    name = "inspeximus"

    def __init__(self, version=None):
        self.version = version
        if version:
            root = _wheel(version)
            sys.path.insert(0, root)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"insp_{version.replace('.', '_')}",
                os.path.join(root, "inspeximus", "__init__.py"),
                submodule_search_locations=[os.path.join(root, "inspeximus")])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            self._cls = mod.Inspeximus
        else:
            from inspeximus import Inspeximus
            self._cls = Inspeximus

    def reset(self):
        self.m = self._cls(path=os.path.join(tempfile.mkdtemp(), "s.json"), receipts=True)

    def write(self, text, subject):
        self.m.remember(text, source={"doc": subject})

    def correct(self, text, subject, key):
        # Through the intent router, which is how a correction actually arrives from a user utterance.
        try:
            self.m.route(text, key=key, object=text.split(" is ")[-1], source=subject)
        except TypeError:
            self.m.route(text, key=key, object=text.split(" is ")[-1])

    def erase_subject(self, subject):
        return self.m.forget_subject(subject, request_id="ramr-erasure", basis="art17")

    def surface(self):
        return " ".join((r.get("text") or "") + " " + str(r.get("object") or "")
                        for r in self.m.items if r.get("status") != "erased")


def _wheel(version):
    """Download and extract a PUBLISHED wheel. Versions are measured as shipped, never from a checkout."""
    import subprocess
    import zipfile
    d = os.path.join(tempfile.gettempdir(), "ramr_wheels", version)
    pkg = os.path.join(d, "x", "inspeximus", "__init__.py")
    if not os.path.exists(pkg):
        os.makedirs(d, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", f"inspeximus=={version}",
                        "--no-deps", "--no-cache-dir", "-d", d], capture_output=True, check=True)
        whl = next(f for f in os.listdir(d) if f.endswith(".whl"))
        with zipfile.ZipFile(os.path.join(d, whl)) as z:
            z.extractall(os.path.join(d, "x"))
    return os.path.join(d, "x")


class Mem0Adapter(ErasureAdapter):
    """mem0, in a fully local free configuration: HuggingFace MiniLM embedder + on-disk qdrant, infer=False.

    `infer=False` is deliberate and it is the FAIR choice here, not a shortcut: it stores the text as
    given, so the cell measures mem0's DELETION, not the quality of an LLM extraction pass it would
    otherwise run. Scoring a store's erasure through a lossy extractor would blame deletion for something
    the extractor did.

    SUBJECT vs SCOPE, stated because it is the honest crux of mem0's row rather than a gotcha: mem0 deletes
    by `user_id`, which is a SCOPE. That is exactly a data subject when one scope holds one person, and it
    is not when a scope holds a conversation in which other people are mentioned -- the ordinary case for
    agent memory. This cell writes one subject per scope, which is the arrangement most favourable to mem0.
    """

    name = "mem0"

    def __init__(self, version=None):
        os.environ.setdefault("OPENAI_API_KEY", "sk-none")   # mem0 initialises an LLM it will not call
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        from mem0 import Memory
        self._Memory = Memory
        self._subjects = []
        self._last_subjects = []
        self.m = None

    def _build(self):
        d = tempfile.mkdtemp()
        return self._Memory.from_config({
            "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
            "vector_store": {"provider": "qdrant",
                             "config": {"path": os.path.join(d, "qd"),
                                        "embedding_model_dims": 384, "on_disk": True}},
            "history_db_path": os.path.join(d, "h.db")})

    def reset(self):
        """ONE store for the whole run, cleared between cases -- and the clearing is VERIFIED.

        mem0 opens a migrations qdrant under ~/.mem0 at construction and holds a file lock on it, so a
        second Memory in the same process dies with "already accessed by another instance"; MEM0_DIR does
        not move it because the path resolves at import. Rebuilding per case is therefore not available.

        The risk this creates is real and is the reason for the assertion: state surviving a reset would
        silently turn an over-erasure case into a pass, because the bystander would appear intact for the
        wrong reason. So the surface is READ after clearing and the run aborts if anything is left.
        """
        if getattr(self, "m", None) is None:
            self.m = self._build()
        for s in self._subjects:
            try:
                self.m.delete_all(user_id=s)
            except Exception:
                pass
        self._subjects = []
        leftover = self._read_all(self._last_subjects)
        if leftover.strip():
            raise RuntimeError(f"mem0 reset left state behind: {leftover[:120]!r} -- "
                               "scores from a dirty store would be meaningless")
        self._last_subjects = []

    def write(self, text, subject):
        self.m.add(text, user_id=subject, infer=False)
        for lst in (self._subjects, self._last_subjects):
            if subject not in lst:
                lst.append(subject)

    def correct(self, text, subject, key):
        # mem0 has no correction channel; a later statement is simply another memory. That IS its design,
        # and the revert/echo cells already score what it costs.
        self.write(text, subject)

    def erase_subject(self, subject):
        return self.m.delete_all(user_id=subject)

    def _read_all(self, subjects):
        """Read every scope. A FAILED read must RAISE, never be swallowed.

        The first version wrapped this in `except Exception: continue`, and mem0 rejects
        `get_all(user_id=...)` in favour of `get_all(filters={"user_id": ...})`. Every read therefore
        returned nothing, the bystander appeared to have vanished, and the cell scored mem0 0.00 on
        over-erasure -- a published accusation that a competitor deletes the wrong person's data, caused
        entirely by my own swallowed TypeError.

        That is the same defect this cell exists to measure, committed by the harness against the system
        it was measuring. A read error is not an empty store, and a benchmark that cannot tell them apart
        will always find the result it went looking for.
        """
        out = []
        for s in subjects:
            try:
                res = self.m.get_all(filters={"user_id": s})
            except TypeError:
                res = self.m.get_all(user_id=s)      # older mem0 signature
            hits = (res.get("results") if isinstance(res, dict) else res) or []
            out += [(h.get("memory") or "") for h in hits]
        return " ".join(out)

    def surface(self):
        # Everything the store would serve, ACROSS scopes -- a surface read that only looked inside the
        # erased scope could never see collateral damage in another one, which is the whole over-erasure
        # question.
        return self._read_all(self._subjects)


ADAPTERS = {"inspeximus": InspeximusAdapter, "mem0": Mem0Adapter}


# ── the two cells ────────────────────────────────────────────────────────────────────────────────
def erasure_works_at_all(ad):
    """PRECONDITION for both cells, and the reason the board cannot be topped by doing nothing.

    Without this, a system whose `erase_subject` is a no-op scores a perfect 1.00 on over-erasure: the
    bystander survives precisely because nothing was ever deleted. A guard that cannot fire reports safe.
    So before either cell is scored, the adapter has to demonstrate that erasing a REAL subject removes
    that subject's own record. A system that fails this is reported as `erasure: non-functional` and gets
    no score at all -- not a zero, and certainly not a one.
    """
    subject, value, _, _ = CASES[0]
    ad.reset()
    ad.write(f"{subject} detail is {value}", subject)
    try:
        ad.erase_subject(subject)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:70]}"
    ok = value not in ad.surface()
    return ok, "" if ok else "erase_subject left the subject's own record in place"


def cell_over_erasure(ad, n):
    """A request naming a subject the store has NEVER SEEN must not delete anybody's records.

    The bystander is the whole point: they never asked for anything, and a store that canonicalises
    subjects lossily (keeping only a host, say) will take them down with a stranger's request.
    """
    verds = []
    for i in range(min(n, len(CASES))):
        subject, value, bystander, b_value = CASES[i]
        ghost = subject.rsplit("/", 1)[0] + "/nobody-" + str(i)
        ad.reset()
        ad.write(f"{bystander} detail is {b_value}", bystander)
        try:
            ad.erase_subject(ghost)
            verds.append("clean" if b_value in ad.surface() else "collateral")
        except Exception:
            # A refusal is a PASS here: refusing an unknown or ambiguous subject is the safe answer, and
            # it is a materially different behaviour from silently deleting the wrong person -- so the
            # surface is still read rather than the refusal being taken on trust.
            verds.append("clean" if b_value in ad.surface() else "collateral")
    return verds


def cell_under_erasure(ad, n):
    """A request naming a REAL subject must reach their CURRENT value, not only the superseded copy."""
    verds = []
    for i in range(min(n, len(CASES))):
        subject, value, _, _ = CASES[i]
        new = NEW_VALUES[i % len(NEW_VALUES)]
        key = subject.replace("/", "::") + "::detail"
        ad.reset()
        ad.write(f"{subject} detail is {value}", subject)
        ad.correct(f"{subject} detail is {new}", subject, key)
        try:
            ad.erase_subject(subject)
        except Exception:
            verds.append("residue")
            continue
        left = ad.surface()
        verds.append("clean" if (new not in left and value not in left) else "residue")
    return verds


def score(verds, good):
    n = sum(1 for v in verds if v != "error")
    ok = sum(1 for v in verds if v == good)
    return {"n": n, "success": ok, "rate": round(ok / n, 3) if n else 0.0,
            "ci95": list(wilson(ok, n)),
            "verdicts": {v: verds.count(v) for v in sorted(set(verds))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default="inspeximus")
    ap.add_argument("--versions", default="", help="comma-separated; measures PUBLISHED wheels")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "erasure.json"))
    a = ap.parse_args()

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": a.n, "judge": "none - deterministic", "results": {}}

    targets = []
    for s in a.systems.split(","):
        s = s.strip()
        if not s:
            continue
        if a.versions and s == "inspeximus":
            targets += [(f"{s} {v}", ADAPTERS[s], v) for v in a.versions.split(",")]
        else:
            targets.append((s, ADAPTERS[s], None))

    for label, cls, version in targets:
        try:
            ad = cls(version) if version else cls()
        except Exception as e:
            out["results"][label] = {"capability": "absent", "why": f"{type(e).__name__}: {str(e)[:90]}"}
            print(f"  {label}: adapter unavailable - {type(e).__name__}")
            continue
        works, why = erasure_works_at_all(ad)
        if not works:
            out["results"][label] = {"erasure": "non-functional", "why": why,
                                     "note": "not scored: a store that erases nothing would otherwise "
                                             "score a perfect 1.00 on over-erasure"}
            print(f"  {label}: erasure NON-FUNCTIONAL - {why}  (not scored)")
            continue
        for cell, fn, good in (("over_erasure", cell_over_erasure, "clean"),
                               ("under_erasure", cell_under_erasure, "clean")):
            print(f"  {label} · {cell} ...", flush=True)
            s = score(fn(ad, a.n), good)
            out["results"][f"{label}:{cell}"] = s
            print(f"    -> {s['success']}/{s['n']} = {s['rate']} CI{s['ci95']} {s['verdicts']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
