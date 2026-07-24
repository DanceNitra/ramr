"""RAMR PREFLIGHT - a parity gate that refuses to let you compare unequal things.

RAMR measures memory reliability. This module measures whether a comparison was even ADMISSIBLE, and it
is the check we ask of ourselves before publishing any RAMR number: if two arms were not handed
comparable context, the ranking between them is a budget result, not a capability result.

WHY IT EXISTS

In a memory-vs-RAG comparison we ran, a memory arm retrieving k=20 sentence-level hits was compared
against a session-level BM25 arm returning whole sessions. Same nominal "top-k", 1323 characters versus
11941 - a measured 9.03x gap. BM25 appeared to win by a mile. Matched (11916 vs 11941, 1.00x), accuracy
went 0.283 -> 0.593 for the memory arms and the ranking flipped. The original finding was a budget result wearing a
granularity costume.

Two more failures from the same run, both producing clean numbers with clean logs:

  * a baseline scored 0.000 twice. Both times it was our harness: inputs truncated before ingestion, and
    `limit=` passed to an API that wanted `top_k=`, which silently swallowed the unknown kwarg. A finding
    about that baseline was already half-written. It was false.
  * the gold evidence was not in the retrieved context at all on 96.5% of probes. You cannot out-rank
    evidence that was never retrieved, so every ranking conclusion above that floor was noise.

THE FOUR GATES

  G0  BUDGET PARITY      are the arms given comparable context? (the one nobody reports)
  G1  RETRIEVAL          is the gold evidence even IN the retrieved context?
  G2  LIVENESS           does each arm pass a probe it cannot fail?
  G3  PARAMETER EFFICACY does the retrieval knob actually change what comes back?

They stay SEPARATE on purpose, because the abort has to NAME the dead layer - folded into one boolean you
get "something died" with the first hour of debugging still ahead. That design point, and G3 itself, came
from u/jacksonxly in a public thread; so did splitting G1 into a per-probe gate plus an aggregate recall
CEILING reported beside accuracy rather than folded into it.

RAMR's own headline arms (FTS5/BM25 keyword, dense vector, inspeximus) retrieve top-1 from an identical
candidate set, so their budgets are equal by construction and G0 passes trivially. That is the point of
running it anyway: the convention only means something if we hold our own numbers to it.

WHAT THIS IS NOT

It does not score a system, rank anything, or care which arm wins, and it never calls an LLM. `liveness`
is a number YOU supply from a positive control YOU run. Passing every gate means a comparison is
INTERPRETABLE, not that it is RIGHT. Zero dependencies, stdlib only.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Callable, Iterable, Sequence

__version__ = "0.5.0"  # ships with RAMR

__all__ = ["Arm", "Probe", "Gate", "Report", "audit", "selftest"]


# ── inputs ────────────────────────────────────────────────────────────────────────────────────────
class Arm:
    """One thing being compared.

    `retrieve(question, k)` must return the context STRING that this arm would put in front of the
    answerer - after any truncation you apply, because the truncation is part of the budget. `k` is
    whatever depth knob the arm has; if it has none, accept the argument and ignore it, and set
    `has_k=False` so G3 reports SKIPPED instead of silently passing.

    `liveness` is this arm's score on a probe it cannot fail (for an answerer arm, a context built from
    the corpus's own ground truth; for a store arm, the smallest input it must ingest and return). It is
    required, not optional: a null is only interpretable if the arm was alive, and the whole reason a
    competitor's 0.000 nearly got published as a finding is that nobody had run one.
    """

    def __init__(self, name: str, retrieve: Callable[[str, int], str],
                 liveness: float | None = None, has_k: bool = True):
        self.name = name
        self.retrieve = retrieve
        self.liveness = liveness
        self.has_k = has_k


class Probe:
    """One question, plus the gold evidence a correct answer would have to be built from.

    `gold_spans` are verbatim quotes from the source material. Probes with no gold spans are EXCLUDED
    from G1's denominator rather than counted as misses - and if that leaves the denominator empty, G1
    fails, because a comparison with nothing to find must never read as a pass.
    """

    def __init__(self, question: str, gold_spans: Sequence[str] = ()):
        self.question = question
        self.gold_spans = [s for s in gold_spans if str(s or "").strip()]


# ── output ────────────────────────────────────────────────────────────────────────────────────────
class Gate:
    """One verdict. `layer` is the point of the whole library: an abort has to name what died."""

    def __init__(self, gate: str, layer: str, status: str, value, requirement, detail: str = ""):
        self.gate, self.layer, self.status = gate, layer, status      # status: PASS | FAIL | SKIPPED
        self.value, self.requirement, self.detail = value, requirement, detail

    @property
    def passed(self) -> bool:
        return self.status != "FAIL"

    def abort_message(self) -> str:
        return (f"DEAD LAYER: {self.layer}  [{self.gate}]  "
                f"measured {self.value}, required {self.requirement}. {self.detail}")

    def as_dict(self) -> dict:
        return {"gate": self.gate, "layer": self.layer, "status": self.status,
                "value": self.value, "requirement": self.requirement, "detail": self.detail}

    def __repr__(self) -> str:
        return f"[{self.status:7}] {self.gate:18} ({self.layer}) = {self.value}"


class Report:
    def __init__(self, gates: list[Gate], budgets: dict, ceilings: dict, n_probes: int, unit: str):
        self.gates, self.budgets, self.ceilings = gates, budgets, ceilings
        self.n_probes, self.unit = n_probes, unit

    @property
    def ok(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failures(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed]

    def as_dict(self) -> dict:
        return {"ramr_preflight": __version__, "ok": self.ok, "n_probes": self.n_probes,
                "budget_unit": self.unit, "budgets": self.budgets, "ceilings": self.ceilings,
                "gates": [g.as_dict() for g in self.gates]}

    def render(self) -> str:
        """The block to paste next to your accuracy table. Reporting it is the entire convention.

        The ceiling column is not decoration: accuracy has to be read against how much of the gold
        evidence each arm could even see. An arm that answers better on less evidence is an interesting
        result; an arm that answers better on 9x the context is not a result at all.
        """
        w = max([len(a) for a in self.budgets] + [4])
        lines = [f"RAMR preflight {__version__} - retrieval preflight, {self.n_probes} probes/arm",
                 "",
                 f"  {'arm'.ljust(w)}  {'budget/probe'.rjust(13)}  {'vs smallest'.rjust(11)}  "
                 f"{'evidence ceiling'.rjust(16)}",
                 f"  {'-' * w}  {'-' * 13}  {'-' * 11}  {'-' * 16}"]
        smallest = min(self.budgets.values()) if self.budgets else 0
        for arm, b in sorted(self.budgets.items(), key=lambda kv: -kv[1]):
            ratio = (b / smallest) if smallest else float("inf")
            ceil = self.ceilings.get(arm)
            lines.append(f"  {arm.ljust(w)}  {b:>9.0f} {self.unit}  {ratio:>10.2f}x  "
                         + (f"{ceil:>16.3f}" if ceil is not None else f"{'n/a':>16}"))
        lines += ["", "  gates:"]
        for g in self.gates:
            lines.append(f"    {g}")
        if self.ok:
            lines += ["", "  ALL GATES PASS - the comparison is interpretable."]
        else:
            lines += [""]
            for g in self.failures:
                lines.append("  " + g.abort_message())
            lines += ["", "  DO NOT report accuracy from this run: the arms were not comparing the "
                          "same thing."]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.render()


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def gate_budget_parity(budgets: dict, tolerance: float, unit: str) -> Gate:
    """G0 - the one nobody reports. Every arm must be handed a comparable amount of context.

    Compares the largest mean budget against the smallest. A 9x gap is not a subtle bias: it is usually
    the entire result. `tolerance=0.25` means the biggest arm may spend at most 25% more than the
    smallest before the comparison is refused.
    """
    if len(budgets) < 2:
        return Gate("G0_budget_parity", "experiment design", "SKIPPED", "one arm",
                    f"<= {1 + tolerance:.2f}x", "Parity needs at least two arms to compare.")
    hi, lo = max(budgets.values()), min(budgets.values())
    ratio = (hi / lo) if lo else float("inf")
    worst = max(budgets, key=lambda a: budgets[a])
    best = min(budgets, key=lambda a: budgets[a])
    ok = ratio <= 1 + tolerance
    return Gate("G0_budget_parity", "experiment design", "PASS" if ok else "FAIL",
                f"{ratio:.2f}x ({worst} {hi:.0f} vs {best} {lo:.0f} {unit})", f"<= {1 + tolerance:.2f}x",
                "Arms were handed different amounts of context, so any accuracy difference is "
                "confounded with budget. Match the budget and re-run - the ranking can flip.")


def gate_retrieval(arm: str, pairs: Iterable[tuple], min_ceiling: float) -> tuple[Gate, float | None]:
    """G1 - is the gold evidence even in the retrieved context?

    Returns the gate AND the ceiling, because the ceiling is a number you report beside accuracy rather
    than a pass/fail you throw away. LLM-free.
    """
    hit = scored = 0
    for ctx, spans in pairs:
        if not spans:
            continue                                  # nothing to find here; not evidence of failure
        scored += 1
        c = _norm(ctx)
        if any(_norm(s) and _norm(s) in c for s in spans):
            hit += 1
    if not scored:
        return (Gate("G1_retrieval", f"retrieval ({arm})", "FAIL", "0/0 probes carried gold spans",
                     f">= {min_ceiling:.2f}",
                     "No probe carried gold evidence, so nothing was actually checked. An empty "
                     "denominator must never be reported as a pass."), None)
    ceiling = round(hit / scored, 4)
    ok = ceiling >= min_ceiling
    return (Gate("G1_retrieval", f"retrieval ({arm})", "PASS" if ok else "FAIL",
                 f"{ceiling} ({hit}/{scored})", f">= {min_ceiling:.2f}",
                 "This is also the recall CEILING for this arm. Report it beside accuracy, never "
                 "folded into it: you cannot out-rank evidence that was never retrieved."), ceiling)


def gate_liveness(arm: str, score: float | None, min_score: float) -> Gate:
    """G2 - the positive control, on an input the arm cannot fail.

    A missing score is a FAIL, not a skip. The failure mode this exists for is publishing a competitor's
    0.000 as a finding when the zero was your own truncation bug, and that only gets caught by someone
    having bothered to run the control.
    """
    if score is None:
        return Gate("G2_liveness", f"store/answerer ({arm})", "FAIL", "not supplied",
                    f">= {min_score:.2f}",
                    "No positive control was run for this arm, so a zero from it cannot be "
                    "distinguished from a broken harness. Supply Arm(liveness=...).")
    ok = float(score) >= min_score
    return Gate("G2_liveness", f"store/answerer ({arm})", "PASS" if ok else "FAIL",
                round(float(score), 4), f">= {min_score:.2f}",
                "Below threshold this arm is a DEAD HARNESS, not a clean loss. Do not report its "
                "number as a result.")


def gate_param_efficacy(arm: str, probe_fn: Callable[[int], int], low: int, high: int,
                        has_k: bool = True) -> Gate:
    """G3 - does the knob actually change what comes back?

    An API that accepts unknown kwargs without raising is hostile in a benchmark: `limit=` where the
    callee wanted `top_k=` produces a clean 0.000 with clean logs. Asserting the parameter CHANGED
    BEHAVIOUR is the cheap, general defence, and it does not care what the parameter is called.
    Fails closed: an exception is a failure, not a skip.
    """
    if not has_k:
        return Gate("G3_param_efficacy", f"harness/API binding ({arm})", "SKIPPED", "no depth knob",
                    "retrieved(high) > retrieved(low)",
                    "Arm declared has_k=False. Its budget is fixed by construction.")
    try:
        lo, hi = int(probe_fn(low)), int(probe_fn(high))
    except Exception as e:
        return Gate("G3_param_efficacy", f"harness/API binding ({arm})", "FAIL", f"error: {e}",
                    "retrieved(high) > retrieved(low)",
                    "The retrieval call raised while probing the depth parameter.")
    ok = hi > lo
    return Gate("G3_param_efficacy", f"harness/API binding ({arm})", "PASS" if ok else "FAIL",
                f"k={low} -> {lo}, k={high} -> {hi}", "retrieved(high) > retrieved(low)",
                "The depth parameter does not change what comes back, so it is being silently "
                "swallowed and every number from this arm is meaningless.")


# ── the entry point ───────────────────────────────────────────────────────────────────────────────
def audit(arms: Sequence[Arm], probes: Sequence[Probe], *, k: int = 10,
          budget_tolerance: float = 0.25, min_ceiling: float = 0.30, min_liveness: float = 0.50,
          measure: Callable[[str], float] | None = None, unit: str = "chars",
          k_probe: tuple[int, int] = (1, 50)) -> Report:
    """Run all four gates over every arm and return one Report.

    `measure` converts a context string to its budget; the default is len() in characters. Pass a
    tokenizer (`lambda s: len(enc.encode(s))`) and `unit="tokens"` if that is what your paper reports -
    the point is not which unit, it is that the number appears at all.

    Nothing here calls an LLM. Run it before you spend anything on an answerer.
    """
    measure = measure or (lambda s: float(len(s or "")))
    budgets: dict = {}
    ceilings: dict = {}
    gates: list[Gate] = []

    per_arm_pairs: dict = {}
    for arm in arms:
        pairs, spend = [], []
        for p in probes:
            ctx = arm.retrieve(p.question, k) or ""
            pairs.append((ctx, p.gold_spans))
            spend.append(measure(ctx))
        per_arm_pairs[arm.name] = pairs
        budgets[arm.name] = (sum(spend) / len(spend)) if spend else 0.0

    gates.append(gate_budget_parity(budgets, budget_tolerance, unit))
    for arm in arms:
        g, ceiling = gate_retrieval(arm.name, per_arm_pairs[arm.name], min_ceiling)
        gates.append(g)
        ceilings[arm.name] = ceiling
    for arm in arms:
        gates.append(gate_liveness(arm.name, arm.liveness, min_liveness))
    for arm in arms:
        lo, hi = k_probe
        gates.append(gate_param_efficacy(
            arm.name,
            lambda n, a=arm: len([x for x in (a.retrieve(probes[0].question, n) or "").split("\n") if x.strip()]),
            lo, hi, has_k=arm.has_k))
    return Report(gates, budgets, ceilings, len(probes), unit)


# ── self-test: every gate must be able to fail ────────────────────────────────────────────────────
def selftest() -> None:
    """A gate that cannot fail is a demonstration, not a test. Each one is run against the real failure
    it exists for, including the swallowed-kwarg bug that produced our clean-looking 0.000."""
    assert gate_budget_parity({"a": 1300.0, "b": 11900.0}, 0.25, "chars").status == "FAIL"
    assert gate_budget_parity({"a": 11900.0, "b": 11900.0}, 0.25, "chars").status == "PASS"
    assert gate_budget_parity({"a": 1.0}, 0.25, "chars").status == "SKIPPED"

    absent = [("unrelated text", ["the gold quote"])] * 10
    present = [("... the gold quote ...", ["the gold quote"])] * 10
    assert gate_retrieval("x", absent, 0.30)[0].status == "FAIL"
    assert gate_retrieval("x", present, 0.30)[0].status == "PASS"
    assert gate_retrieval("x", [("ctx", [])] * 5, 0.30)[0].status == "FAIL", \
        "an empty denominator must not read as a pass"

    assert gate_liveness("x", None, 0.5).status == "FAIL", "a missing control is a failure, not a skip"
    assert gate_liveness("x", 0.10, 0.85).status == "FAIL"
    assert gate_liveness("x", 0.87, 0.85).status == "PASS"

    class HostileAPI:                       # accepts anything, silently ignores what it does not know
        def search(self, q, top_k=5, **swallowed):
            return list(range(top_k))

    api = HostileAPI()
    wrong = gate_param_efficacy("x", lambda n: len(api.search("q", limit=n)), 1, 50)
    right = gate_param_efficacy("x", lambda n: len(api.search("q", top_k=n)), 1, 50)
    assert wrong.status == "FAIL", "G3 must catch a silently swallowed parameter name"
    assert right.status == "PASS"
    assert gate_param_efficacy("x", lambda n: 1 / 0, 1, 50).status == "FAIL", "must fail closed"
    assert gate_param_efficacy("x", lambda n: 0, 1, 50, has_k=False).status == "SKIPPED"

    # and the whole thing end to end, on the confound this library was built from
    wide = Arm("session_rag", lambda q, k: "\n".join(["the gold quote"] + ["filler"] * 400),
               liveness=0.9, has_k=False)
    narrow = Arm("memory", lambda q, k: "\n".join(["the gold quote"][:k] or ["x"]), liveness=0.9)
    rep = audit([wide, narrow], [Probe("q?", ["the gold quote"])] * 5)
    assert not rep.ok and any(g.gate == "G0_budget_parity" for g in rep.failures)
    assert "DO NOT report accuracy" in rep.render()
    print("selftest OK - every gate can fail, and fails naming its own layer")


def demo(path: str = "data/ramr_chains_v0.1.0.jsonl", out: str = "preflight_result.json") -> dict:
    """Reproduce the budget-disparity shape on RAMR's OWN frozen dataset, so every number the README cites
    is verifiable from this repo rather than from a corpus we did not ship.

    Two arms answer the same chains from the same facts. `narrow` returns the gold facts only; `wide`
    returns the gold facts PLUS the whole distractor pool - the sentence-level vs session-level shape that
    produced a 9x gap in the run this module came from. Nothing here is a claim about which retrieval
    strategy is better; it is a claim about which comparisons are admissible.
    """
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    chains = []
    with open(os.path.join(base, path), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                chains.append(json.loads(ln))
    # Route by chain ID, not by question text. The frozen dataset has 300 chains but only 89 distinct
    # questions, so keying on the question silently collapsed 81 collisions and handed probes another
    # chain's facts - which showed up as an evidence ceiling of 0.31 where it should have been ~1.0.
    # Exactly the class of harness bug this module exists to catch, found by its own G1. Kept as a note
    # because a demo that quietly had the bug it warns about would be worth nothing.
    by_q = {f'[{c["id"]}] {c["question"]}': c for c in chains}
    probes = [Probe(f'[{c["id"]}] {c["question"]}', c["gold_facts"]) for c in chains]

    def narrow(q, k):
        return "\n".join(f"- {f}" for f in by_q[q]["gold_facts"][:k])

    def wide(q, k):
        c = by_q[q]
        return "\n".join(f"- {f}" for f in (c["gold_facts"] + c["distractor_pool"]))

    unequal = audit([Arm("narrow_facts", narrow, liveness=0.90),
                     Arm("wide_pool", wide, liveness=0.90, has_k=False)],
                    probes, k=3, budget_tolerance=0.25, min_ceiling=0.30)
    # matched: give the narrow arm the same pool, so only the retrieval strategy differs
    matched = audit([Arm("narrow_facts", wide, liveness=0.90, has_k=False),
                     Arm("wide_pool", wide, liveness=0.90, has_k=False)],
                    probes, k=3, budget_tolerance=0.25, min_ceiling=0.30)

    result = {"ramr_preflight": __version__, "dataset": path, "n_probes": len(probes),
              "unequal": unequal.as_dict(), "matched": matched.as_dict()}
    with open(os.path.join(base, out), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(unequal.render())
    print()
    print(matched.render())
    print(f"\nwrote {out}")
    return result


def _main(argv: list[str]) -> int:
    if "--selftest" in argv:
        selftest()
        return 0
    if "--demo" in argv:
        demo()
        return 0
    print(__doc__.strip().split("\n\n")[0])
    print("\nThis is a library: bring your own arms and probes.\n")
    print("    from ramr_preflight import Arm, Probe, audit\n"
          "    report = audit([Arm('mine', retrieve_mine, liveness=0.87),\n"
          "                    Arm('theirs', retrieve_theirs, liveness=0.91)], probes)\n"
          "    print(report.render())      # paste this next to your accuracy table\n"
          "    assert report.ok            # or refuse to report the comparison\n")
    print("Run `python ramr_preflight.py --selftest` to verify every gate can still fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
