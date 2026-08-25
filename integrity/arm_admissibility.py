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

    print(f"\n{'SELF-TEST PASSED' if not fails else 'SELF-TEST FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest())
