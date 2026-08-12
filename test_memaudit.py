"""Controls for the audit battery itself.

A battery of surface rules run against any labelled data will find SOMETHING -- with enough rules,
noise separates. So the tool is only worth its output if it stays silent on data that carries no
shortcut, and speaks on data that carries a planted one. Both directions are asserted here, on
synthetic fixtures built for the purpose, because neither can be demonstrated on the benchmark under
audit: that one is the thing whose leak status is unknown.

    python test_memaudit.py        # exit 0 = the battery is calibrated
"""
import random
import sys

from memaudit import PROBES, Trace, audit, envelope

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def clean_traces(n=300, k=8, seed=3):
    """No shortcut by construction: the target is chosen uniformly AFTER the items are built, and
    every item is drawn from one distribution, so position, length, casing and recurrence carry
    nothing. Any probe that fires above its own null here is finding structure that is not there."""
    rnd = random.Random(seed)
    out = []
    for t in range(n):
        items = []
        for j in range(k):
            body = " ".join(rnd.choice(WORDS) for _ in range(4))
            items.append({"id": "t%d-i%d" % (t, j), "text": "%s." % body})
        tgt = rnd.randrange(k)
        dec = rnd.choice([j for j in range(k) if j != tgt])
        out.append(Trace("t%d" % t, " ".join(rnd.choice(WORDS) for _ in range(3)),
                         items, items[tgt]["id"], items[dec]["id"]))
    return out


def leaky_traces(n=300, k=8, seed=4):
    """One planted cue and nothing else: the target is always the FIRST candidate."""
    rnd = random.Random(seed)
    out = []
    for t in range(n):
        items = []
        for j in range(k):
            body = " ".join(rnd.choice(WORDS) for _ in range(4))
            items.append({"id": "t%d-i%d" % (t, j), "text": "%s." % body})
        out.append(Trace("t%d" % t, " ".join(rnd.choice(WORDS) for _ in range(3)),
                         items, items[0]["id"], items[1]["id"]))
    return out


def main():
    bad = []

    # NEGATIVE CONTROL -- the one that matters. Silence on clean data.
    rows = audit(clean_traces())
    for r in rows:
        if r["real"]:
            bad.append("clean data: %s reported REAL (acc %.3f vs null %.3f) -- the battery is "
                       "manufacturing shortcuts" % (r["probe"], r["accuracy"] or 0, r["null_accuracy"] or 0))
    env = envelope(clean_traces())
    if env["probes_used"]:
        bad.append("clean data: floor used %s -- must be empty" % env["probes_used"])
    print("negative control (no shortcut) : %d/%d probes REAL, floor uses %d probes"
          % (sum(1 for r in rows if r["real"]), len(rows), len(env["probes_used"])))

    # POSITIVE CONTROL -- it must still SEE one when it is there.
    rows2 = audit(leaky_traces())
    first = [r for r in rows2 if r["probe"] == "position:first"][0]
    if not first["real"]:
        bad.append("planted position leak NOT detected (acc %.3f vs null %.3f) -- the battery is "
                   "blind to the cue it exists to find" % (first["accuracy"] or 0, first["null_accuracy"] or 0))
    print("positive control (target first): position:first acc %.1f%% vs null %.1f%% -> %s"
          % (100 * (first["accuracy"] or 0), 100 * (first["null_accuracy"] or 0),
             "REAL" if first["real"] else "MISSED"))

    # Every probe must be able to abstain rather than guess; a rule that always fires cannot have
    # coverage as evidence.
    for name, fn in PROBES:
        # Two probes legitimately always fire: `position:first` names a candidate unconditionally, and
        # `position:pair-order` is a forced binary choice between the pair. For those, coverage
        # carries no information and only accuracy-against-the-null does -- which is why REAL is
        # decided by lift and never by coverage. Everything else must be able to say "not here".
        ALWAYS_FIRE = ("position:first", "position:pair-order")
        if name not in ALWAYS_FIRE and all(fn(t) is not None for t in clean_traces(n=40)):
            bad.append("%s never abstains -- coverage is meaningless for it" % name)

    print()
    if bad:
        print("FAIL:")
        for b in bad:
            print("  * %s" % b)
        return 1
    print("OK -- silent on clean data, sees a planted cue, and every probe can abstain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
