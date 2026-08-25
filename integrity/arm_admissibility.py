#!/usr/bin/env python3
"""Two gates that decide whether a measurement arm is allowed to report a number at all.

WHY THIS EXISTS, and it is the second time in a week a benchmark of ours reported a number it had
not measured.

An eval that asks a model to emit something it was given produces a number whether or not the model
attended to anything. Two distinct ways that goes wrong were established on anthropics/claude-code
#82056 in August 2026, by other people, and this file is the reusable form of their rules rather
than a claim of ours:

  GATE 1 -- RECONSTRUCTION.  @pjt222 applied `floor(cap / units-per-line)` to every published table
  in that thread and reproduced 13 of 14 size-bound cells. A model that read the index and never
  attended to WHERE IT WAS CUT computes the same answer by arithmetic over a width it can see. So a
  measurement that equals its own reconstruction-predicted value has separated nothing, however many
  trials agree. His rule, verbatim: "An arm is informative only where reading and reconstructing
  predict different numbers. State the reconstruction-predicted value beside every measurement; if
  it equals the measurement, the arm has told you nothing you did not already assume."

  GATE 2 -- ADMISSIBILITY.  @JhouCode's Round 0 asked a model what it could see and got NONE, while
  behavioural probes in the same session proved the content had arrived. Self-report inverted rather
  than degraded. Measured again on our side the same week: 2 of 5 trials of one arm returned nothing
  at all, positive control included, on a fixture that had answered 15 minutes earlier. A trial in
  which the positive control stayed silent measured nothing, and averaging it in reports a number
  the run did not observe.

  AND GATE 2 IS FORTY YEARS OLD, WHICH IS THE POINT.  It is not our rule and it is not new: US
  clinical-laboratory regulation requires it of every reportable result. 42 CFR 493.1256(f),
  verbatim from eCFR: "Results of control materials must meet the laboratory's and, as applicable,
  the manufacturer's test system criteria for acceptability BEFORE REPORTING patient test results."
  Same paragraph, (d)(3)(v), requires an inhibition control so that a NEGATIVE is only reportable
  if the control fired -- exactly the case that bit us. What is worth saying is not that the rule
  is new but that no ML evaluation framework enforces it: refusal gets tracked and never gates.
  So this file imports a clinical rule into a field that lacks it, and is framed that way on
  purpose. Presenting it as ours would be the overclaim this repository exists to avoid.

  The phenomenon behind gate 2 is textbook too, and in two fields. Anthropic measured Claude 2.1
  long-context recall going 27% to 98% on one added priming sentence, the whole 71 points being
  reluctance rather than retrieval. Nasr et al. (arXiv:2311.17035) formalise the same split as
  EXTRACTABLE versus DISCOVERABLE memorisation and warn against concluding "that the alignment
  procedure has correctly prevented the model from emitting training data". Cite them; do not
  re-derive them.

  AND THE ASYMMETRY THAT FOLLOWS FROM BOTH.  Presence of an unforgeable planted value is evidence of
  receipt; absence is not evidence of anything, because every failure mode above produces silence.
  So a one-sided arm yields a LOWER bound only. Reporting a two-sided bracket from absences is the
  error this gate refuses, and it is the one we made.

WHAT THIS IS NOT. It cannot tell you your fixture is well designed, only that a given arm did not
discriminate. It has no opinion about your subject. It is a refusal, and a refusal is the useful
half: the failures above all LOOK like results.

    from integrity.arm_admissibility import Arm, Trial
    arm = Arm(name="200-units/line", reconstruction_predicts=125)
    arm.add(Trial(control_fired=True,  present={124: True, 125: True, 126: False}))
    ...
    v = arm.verdict()          # v.informative, v.admissible_trials, v.lower_bound, v.refusals

stdlib only. Run this file directly for the self-test.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trial:
    """One session. `present` maps a needle id to whether its exact planted value came back.

    `control_fired` is the POSITIVE control for this trial specifically, not for the run. A run-level
    control cannot tell an admissible trial from an inadmissible one, which is the distinction the
    whole gate turns on.
    """
    control_fired: bool
    present: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Arm:
    name: str
    # What a model that read nothing about the cut would predict for this arm, computed from
    # documented constants and the fixture's own geometry. None means "no reconstruction exists",
    # which is a claim the caller must be able to defend, so it is recorded, not assumed.
    reconstruction_predicts: Any = None
    # Needle id -> the position (unit, byte, index) its evidence pins. Used to turn presence into a
    # bound. Absent ids simply do not contribute.
    positions: dict = field(default_factory=dict)
    trials: list = field(default_factory=list)

    def add(self, t: Trial) -> "Arm":
        self.trials.append(t)
        return self

    def verdict(self) -> "Verdict":
        adm = [i for i, t in enumerate(self.trials, 1) if t.control_fired]
        refusals: list = []
        if not self.trials:
            refusals.append("no trials: nothing was measured")
        if not adm:
            refusals.append(
                "no admissible trial: the positive control fired in none of them, so every "
                "'absent' here is indistinguishable from an instrument that said nothing")

        # PRESENCE ONLY. A needle counts as read if its exact value came back in at least one
        # ADMISSIBLE trial. Absence is deliberately not aggregated into a bound.
        read = sorted({n for i in adm for n, ok in self.trials[i - 1].present.items() if ok})
        never = sorted({n for t in self.trials for n in t.present} - set(read))
        pos = [self.positions[n] for n in read if n in self.positions]
        lower = max(pos) if pos else None

        informative = True
        if self.reconstruction_predicts is not None:
            # The measurement this arm would report, expressed the same way the reconstruction is.
            measured = max(read) if read else None
            if measured is not None and measured == self.reconstruction_predicts:
                informative = False
                refusals.append(
                    f"uninformative: the arm measured {measured} and reconstruction predicts "
                    f"{self.reconstruction_predicts}. Reading and computing give the same answer "
                    f"here, so this cell cannot separate them")

        # The one that bit us: a two-sided claim built out of silences.
        upper_claimable = False

        return Verdict(arm=self.name, informative=informative, admissible_trials=adm,
                       total_trials=len(self.trials), needles_read=read,
                       needles_never_read=never, lower_bound=lower,
                       upper_bound_claimable=upper_claimable, refusals=refusals)


@dataclass
class Verdict:
    arm: str
    informative: bool
    admissible_trials: list
    total_trials: int
    needles_read: list
    needles_never_read: list
    lower_bound: Any
    upper_bound_claimable: bool
    refusals: list

    @property
    def may_report(self) -> bool:
        return self.informative and bool(self.admissible_trials) and not self.refusals

    def __str__(self) -> str:
        head = f"[{'REPORTABLE' if self.may_report else 'REFUSED'}] {self.arm}"
        body = [f"  admissible trials : {self.admissible_trials} of {self.total_trials}",
                f"  needles read      : {self.needles_read}",
                f"  never read        : {self.needles_never_read}  (NOT evidence of absence)",
                f"  lower bound       : {self.lower_bound}",
                f"  upper bound       : not claimable from absences"]
        return "\n".join([head, *body, *(f"  REFUSAL: {r}" for r in self.refusals)])



# gate 3: how wide is that lower bound, really?
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval, because every n in this work is small."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return ((c - m) / d, (c + m) / d)


@dataclass
class FramingGap:
    """How much of an emit-rate is the QUESTION rather than the content.

    Gates 1 and 2 decide whether an arm may report a bound at all. This asks how far that bound
    sits from the truth, and it exists because a lower bound with no width gets read as a point.
    Hand it the same content asked several ways: the spread between the best and worst framing is
    emission suppressed by the question, at fixed content and fixed position.

    THE HONEST DEFAULT IS TO REFUSE. Our own best receipt is 3 trials per framing, where the
    Wilson intervals overlap ([0.438, 1.000] against [0.000, 0.562]) and Fisher gives p = 0.100 --
    the BEST a 3-vs-3 table can do even when every trial falls the right way. So it flags and does
    not correct. Applying a correction from an unseparated gap would be this file doing the exact
    thing it was written to stop."""
    cells: dict                      # framing label -> (hits, trials)

    def verdict(self) -> dict:
        if len(self.cells) < 2:
            return {"separated": False, "apply_correction": False,
                    "reason": "need at least two framings to see a gap"}
        r = {k: (h / n if n else 0.0, wilson(h, n), n) for k, (h, n) in self.cells.items()}
        best = max(r, key=lambda k: r[k][0])
        worst = min(r, key=lambda k: r[k][0])
        b, w = r[best], r[worst]
        sep = b[1][0] > w[1][1]        # best's lower bound clears worst's upper bound
        return {"best": best, "worst": worst,
                "point_gap": round(b[0] - w[0], 4),
                "best_ci": [round(x, 3) for x in b[1]],
                "worst_ci": [round(x, 3) for x in w[1]],
                "n_per_cell": {k: v[2] for k, v in r.items()},
                "separated": sep, "apply_correction": sep,
                "reason": ("the intervals separate, so the emit channel demonstrably under-reports"
                           " by at least this much under the worse framing" if sep else
                           "the intervals OVERLAP at this n: a gap is visible and not established."
                           " Flag it, widen nothing, and say the bound may be conservative by an"
                           " unmeasured amount")}

# ────────────────────────────────────────────────────────────────────── self-test, with teeth
def _selftest() -> int:
    fails = []

    def check(label, cond):
        if not cond:
            fails.append(label)
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")

    # 1. A clean, discriminating arm reports a LOWER bound and nothing else.
    a = Arm("clean", reconstruction_predicts=999, positions={124: 24799, 125: 24999, 126: 25199})
    for _ in range(3):
        a.add(Trial(control_fired=True, present={124: True, 125: True, 126: False}))
    v = a.verdict()
    check("a clean arm is reportable", v.may_report)
    check("its bound is the highest position READ", v.lower_bound == 24999)
    check("it never claims an upper bound", v.upper_bound_claimable is False)
    check("the unread needle is listed, not counted", v.needles_never_read == [126])

    # 2. THE RECONSTRUCTION GATE. Same data, but the arm's measurement equals what arithmetic
    #    predicts. This is our own 147x180 -> 168 cell, and it must be refused.
    b = Arm("reconstructible", reconstruction_predicts=125,
            positions={124: 24799, 125: 24999})
    for _ in range(3):
        b.add(Trial(control_fired=True, present={124: True, 125: True}))
    vb = b.verdict()
    check("an arm that equals its reconstruction is REFUSED", not vb.may_report)
    check("and it says so", any("uninformative" in r for r in vb.refusals))

    # 3. THE ADMISSIBILITY GATE. Every control silent: the run measured nothing, however many
    #    'absences' it collected.
    c = Arm("all silent", positions={125: 24999})
    for _ in range(5):
        c.add(Trial(control_fired=False, present={125: False}))
    vc = c.verdict()
    check("an arm with no live control is REFUSED", not vc.may_report)
    check("and no bound leaks out of it", vc.lower_bound is None)

    # 4. A MIXED run: 2 of 5 controls silent, exactly the shape we measured. The silent trials must
    #    not drag the bound down, because presence in ANY admissible trial is evidence.
    d = Arm("mixed", positions={124: 24799, 125: 24999})
    d.add(Trial(control_fired=True, present={124: True, 125: True}))
    d.add(Trial(control_fired=False, present={124: False, 125: False}))
    d.add(Trial(control_fired=False, present={124: False, 125: False}))
    vd = d.verdict()
    check("silent trials are excluded, not averaged", vd.admissible_trials == [1])
    check("and the bound survives them", vd.lower_bound == 24999)

    # 5. THE CONTROL ON THE GATE ITSELF. Every check above asserts a REFUSAL, which is the shape
    #    that passes when the gate does nothing at all. So assert the gate can also say YES, and
    #    that its two refusals are independent -- a gate wired to refuse everything would pass 2-4
    #    and fail here.
    check("the gate is not simply always-refusing", Arm(
        "positive", positions={1: 10}).add(
        Trial(control_fired=True, present={1: True})).verdict().may_report)

    # 6. GATE 3, fed our own best receipt: a_probe_with_tools_enabled_can_answer_from_disk,
    #    2.1.241, guarded arm with tools asserted zero. "the last CANARY you can SEE" 3/3,
    #    naming the file 1/3, a neutral phrasing 0/3. A total point gap that n=3 cannot
    #    establish, so the gate must REFUSE to correct. That refusal is why it exists.
    real = FramingGap({"you_can_see": (3, 3), "names_the_file": (1, 3),
                       "neutral": (0, 3)}).verdict()
    check("gate 3 sees the full point gap", real["point_gap"] == 1.0)
    check("gate 3 REFUSES to correct at n=3", real["apply_correction"] is False)
    check("gate 3 names the overlap as the reason", "OVERLAP" in real["reason"])
    check("gate 3 DOES correct once the intervals separate",
          FramingGap({"a": (30, 30), "b": (0, 30)}).verdict()["apply_correction"] is True)
    check("CONTROL identical cells show no separation",
          FramingGap({"a": (15, 30), "b": (15, 30)}).verdict()["separated"] is False)

    print(f"\n{'SELF-TEST PASSED' if not fails else 'SELF-TEST FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest())
