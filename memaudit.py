"""memaudit -- what can be solved on your benchmark WITHOUT understanding it.

A partial-input baseline battery for retrieval-trace benchmarks. You hand it blind traces (a query and
a candidate set) plus the labels, and it reports how much of the task falls to rules that read surface
form only: position, casing, length, token recurrence, query overlap, and stray label fields. The
headline number is the SHORTCUT FLOOR -- the score a system can reach while knowing nothing.

WHY A FLOOR AND NOT A PASS/FAIL. A probe that passes proves nothing: "failures of partial-input
baselines do not mean the dataset is free of artifacts" (Feng, Wallace & Boyd-Graber, ACL 2019). The
number worth publishing is therefore not "clean/dirty" but the floor itself, printed beside every score
the benchmark reports, so a reader can see how much of a result the benchmark could have produced by
itself. Gururangan et al. (NAACL 2018) reached 67.0 on SNLI from the hypothesis alone against a 34.3
majority baseline; nobody could act on that until it was a number.

THE NULL IS THE POINT. A battery of rules run against any labelled data will find something -- with
enough rules, noise is separable. So every probe is also run against PERMUTED labels, and its score
there is its own chance level, measured rather than assumed. A probe that scores as well on shuffled
labels as on real ones has found nothing, and this tool says so instead of reporting it. Without that
control a shortcut battery is a machine for manufacturing alarming numbers.

WHAT IT DOES NOT DO. It cannot certify a benchmark clean; nothing can. It measures the floor reachable
by THESE rules, which is a lower bound on the true floor and never an upper one.

    python memaudit.py --adapter demo                        # see it work, with no data of yours
    python memaudit.py --adapter ramr
    python memaudit.py --adapter locomo --blind locomo10.json
    python memaudit.py --blind traces.jsonl --labels labels.jsonl --target planted --decoy replaces
"""
import argparse
import io
import json
import os
import random
import sys

__version__ = "0.1.0"


# ── the generic shape everything is adapted into ─────────────────────────────────────────────────────
class Trace(object):
    """One query, one candidate set, and which candidate the task is to pick.

    `target` is the item a scorer is supposed to identify (for RAMR: the planted stale record).
    `decoy` is the item it is most easily confused with, when the benchmark defines one (for RAMR: the
    gold record the plant replaces). Probes that need a PAIR use decoy; the rest need only target.
    """

    __slots__ = ("tid", "query", "items", "target", "decoy")

    def __init__(self, tid, query, items, target, decoy=None):
        self.tid = tid
        self.query = query or ""
        self.items = items                  # [{"id":..., "text":..., **extra}]
        self.target = target
        self.decoy = decoy

    def by_id(self, rid):
        for i in self.items:
            if i.get("id") == rid:
                return i
        return None


# ── probes ───────────────────────────────────────────────────────────────────────────────────────────
#
# Each returns a GUESS (an item id) or None for "I do not fire here". Coverage and accuracy are
# reported separately and never averaged into one number: a rule that fires on 3% of traces and is
# always right is a different object from one that fires everywhere and is usually right, and
# collapsing them is how a benchmark gets called broken on the strength of a rare cue.

def probe_position_first(t):
    """The target is simply the earliest candidate. The cheapest cue there is."""
    return t.items[0].get("id") if t.items else None


def probe_pair_order(t):
    """Of the target/decoy PAIR, the earlier one is the target. Needs a defined pair."""
    if not t.decoy:
        return None
    ranks = {i.get("id"): n for n, i in enumerate(t.items)}
    if t.target not in ranks or t.decoy not in ranks:
        return None
    return t.target if ranks[t.target] < ranks[t.decoy] else t.decoy


def probe_stray_field(t):
    """A field beyond id/text that separates the target. The answer key shipped inside the question.

    Deliberately blind to WHICH field: it looks for any key whose value differs between the target and
    everything else, because a benchmark that leaks its label rarely leaks it under an obvious name.
    """
    tgt = t.by_id(t.target)
    if not tgt:
        return None
    others = [i for i in t.items if i.get("id") != t.target]
    if not others:
        return None
    for k, v in tgt.items():
        if k in ("id", "text"):
            continue
        if all(o.get(k) != v for o in others):
            return t.target
    return None


def _last_token(s):
    w = (s or "").rstrip(".").split()
    return w[-1] if w else None


def probe_value_echo(t):
    """Of the pair, the value that recurs ELSEWHERE in the candidate set is the current one.

    Needs no labels and no semantics: it reads whether the final token of each half appears in the
    rest of the retrieved text. Whether recurrence marks current or stale is a property of the
    generator, so both directions are scored and the better one is reported by name.
    """
    if not t.decoy:
        return None
    a, b = t.by_id(t.target), t.by_id(t.decoy)
    if not a or not b:
        return None
    av, bv = _last_token(a.get("text")), _last_token(b.get("text"))
    if not av or not bv or av == bv:
        return None
    rest = " ".join(i.get("text") or "" for i in t.items
                    if i.get("id") not in (t.target, t.decoy)).split()
    ae, be = av in rest, bv in rest
    if ae == be:
        return None
    return t.decoy if ae else t.target      # the echoing half is the OTHER one


def probe_casing(t):
    """Of the pair, exactly one value is capitalised. Borrowed values often come from another slot."""
    if not t.decoy:
        return None
    a, b = t.by_id(t.target), t.by_id(t.decoy)
    if not a or not b:
        return None
    av, bv = _last_token(a.get("text")), _last_token(b.get("text"))
    if not av or not bv or av[:1].isupper() == bv[:1].isupper():
        return None
    return t.target if av[:1].isupper() else t.decoy


def probe_length(t):
    """Of the pair, the longer text. Generators that append or substitute leave a length signature."""
    if not t.decoy:
        return None
    a, b = t.by_id(t.target), t.by_id(t.decoy)
    if not a or not b:
        return None
    la, lb = len(a.get("text") or ""), len(b.get("text") or "")
    if la == lb:
        return None
    return t.target if la > lb else t.decoy


def _shape(rid):
    """A coarse signature of an id: length plus which character classes appear."""
    s = str(rid or "")
    return (len(s), any(c.isdigit() for c in s), any(c.isalpha() for c in s),
            any(not c.isalnum() for c in s))


def probe_id_outlier(t):
    """A record whose ID does not look like the others, and what it points at.

    WRITTEN AFTER DOING IT. Balancing the echo cue meant adding a corroborating fact for the stale
    value, and the added record was given the id "sib-<chain_id>" while every other id was 10 hex
    characters. A scorer reading nothing but the id format could find that record, and its SUBJECT is
    the stale value -- a cue introduced while removing one. Every other probe here reported "at
    chance" on that corpus, because none of them looked at the ids.

    THE SIZE OF IT, corrected 2026-09-04. This docstring used to say "205 of 300 traces solved
    outright". That figure has no receipt and does not reproduce. Measured over 20 rebuilds under
    the old id scheme over 100 rebuilds, the cue lands on 204 of 300 traces every time and this rule
    fires on 136 to 148 of them, right on 95.2% to 100.0%, against a permuted-label null at or below
    1.0%. It never reaches 205. The range is a range because the generator mints uuid4 record ids, so
    no rebuild matches another, and the bounds are an OBSERVED envelope over 100 runs rather than a
    property of the corpus: 5 runs gave 138-143, 20 gave 139-146. See ramr_id_cue_reconstruction.py.

    So: find the minority id shape, take that record's first token, and return the candidate whose
    final token matches it. That is the exploit, generalised.
    """
    shapes = {}
    for i in t.items:
        shapes.setdefault(_shape(i.get("id")), []).append(i)
    if len(shapes) < 2:
        return None
    minority = min(shapes.values(), key=len)
    if len(minority) != 1:
        return None
    subject = (minority[0].get("text") or "").split()
    if not subject:
        return None
    head = subject[0].rstrip(".")
    for i in t.items:
        if i is minority[0]:
            continue
        if _last_token(i.get("text")) == head:
            return i.get("id")
    return None


def probe_query_overlap(t):
    """The candidate sharing the most tokens with the query. The classic lexical shortcut."""
    q = set((t.query or "").lower().rstrip("?").split())
    if not q:
        return None
    best, score = None, -1
    for i in t.items:
        s = len(q & set((i.get("text") or "").lower().rstrip(".").split()))
        if s > score:
            best, score = i.get("id"), s
        elif s == score:
            best = None                     # a tie is not a cue
    return best


#: (name, fn, binary). BINARY means the probe chooses between exactly two candidates -- the pair -- so
#: its rule can be INVERTED and remain a rule. This is not hypothetical: the v0.3 re-cut drew the stale
#: value to match the current one's length and turned "stale is longer" (73%) into "stale is SHORTER"
#: (83%). Scored one-directionally that reads as 16.7% and therefore "at chance", and the re-cut would
#: have been declared a success while the cue was stronger than before, pointing the other way. For a
#: binary probe the exploitable quantity is |accuracy - 50%|, and the direction is a label, not a result.
PROBES_SPEC = [
    ("position:first", probe_position_first, False),
    ("position:pair-order", probe_pair_order, True),
    ("leak:stray-field", probe_stray_field, False),
    ("leak:id-outlier", probe_id_outlier, False),
    ("content:value-echo", probe_value_echo, True),
    ("content:casing", probe_casing, True),
    ("content:length", probe_length, True),
    ("content:query-overlap", probe_query_overlap, False),
]
PROBES = [(n, f) for n, f, _ in PROBES_SPEC]
BINARY = {n: b for n, _, b in PROBES_SPEC}


# ── scoring, with the null ───────────────────────────────────────────────────────────────────────────
def run_probe(traces, fn):
    fired = right = 0
    for t in traces:
        g = fn(t)
        if g is None:
            continue
        fired += 1
        right += (g == t.target)
    return fired, right


def permuted(traces, rnd):
    """The null: keep every trace, move the LABEL to another candidate.

    Permuting within the candidate set rather than across traces keeps set size, text and order intact,
    so anything a probe still scores is a property of the rules and not of the data.
    """
    out = []
    for t in traces:
        ids = [i.get("id") for i in t.items]
        if len(ids) < 2:
            out.append(t)
            continue
        choices = [x for x in ids if x != t.target]
        new_target = rnd.choice(choices)
        new_decoy = None
        if t.decoy:
            rest = [x for x in ids if x != new_target]
            new_decoy = rnd.choice(rest) if rest else None
        out.append(Trace(t.tid, t.query, t.items, new_target, new_decoy))
    return out


def audit(traces, null_rounds=5, seed=7):
    rnd = random.Random(seed)
    n = len(traces)
    rows = []
    for name, fn in PROBES:
        fired, right = run_probe(traces, fn)
        acc = (right / fired) if fired else None

        nf = nr = 0
        for _ in range(null_rounds):
            p = permuted(traces, rnd)
            f, r = run_probe(p, fn)
            nf += f
            nr += r
        null_acc = (nr / nf) if nf else None

        # A binary rule read backwards is still a rule -- see the note on PROBES_SPEC. Take whichever
        # direction is exploitable and carry the direction as a label.
        inverted = False
        if BINARY.get(name) and acc is not None and null_acc is not None and acc < null_acc:
            acc, null_acc, inverted = 1.0 - acc, 1.0 - null_acc, True

        lift = None if (acc is None or null_acc is None) else acc - null_acc
        rows.append({"probe": name + (" (inverted)" if inverted else ""),
                     "fired": fired, "coverage": fired / n if n else 0.0,
                     "accuracy": acc, "null_accuracy": null_acc, "lift": lift,
                     "inverted": inverted,
                     "real": bool(lift is not None and lift > 0.10)})
    return rows


def envelope(traces):
    """The floor: apply the REAL probes in order of accuracy, first one that fires wins.

    This is what a system knowing nothing can reach, and it is the number that belongs beside every
    score the benchmark publishes.
    """
    rows = [r for r in audit(traces) if r["real"] and r["accuracy"] is not None]
    rows.sort(key=lambda r: -r["accuracy"])
    lut = dict(PROBES)
    order = [(lut[r["probe"].replace(" (inverted)", "")], r["inverted"]) for r in rows]
    fired = right = 0
    for t in traces:
        for fn, inv in order:
            g = fn(t)
            if g is not None:
                fired += 1
                # An inverted rule picks the OTHER member of the pair.
                if inv:
                    other = t.decoy if g == t.target else t.target
                    g = other
                right += (g == t.target)
                break
    return {"probes_used": [r["probe"] for r in rows], "fired": fired,
            "coverage": fired / len(traces) if traces else 0.0,
            "accuracy": (right / fired) if fired else None}


# ── adapters ─────────────────────────────────────────────────────────────────────────────────────────
def demo_traces(n=300, k=8, seed=4, leak=True):
    """A synthetic fixture, so the tool can demonstrate itself with no data of yours.

    With leak=True the target is always the first candidate and nothing else distinguishes it, so
    `position:first` should read ~100% against a ~0% null. With leak=False the target is drawn
    uniformly after the items are built and every probe should fall to its own chance level -- that
    second case is the one worth trusting, because a battery that cannot stay silent is worthless.
    """
    rnd = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
             "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]
    out = []
    for t in range(n):
        items = [{"id": "t%d-i%d" % (t, j),
                  "text": "%s." % " ".join(rnd.choice(words) for _ in range(4))} for j in range(k)]
        tgt = 0 if leak else rnd.randrange(k)
        dec = rnd.choice([j for j in range(k) if j != tgt])
        out.append(Trace("t%d" % t, " ".join(rnd.choice(words) for _ in range(3)),
                         items, items[tgt]["id"], items[dec]["id"]))
    return out


def _load_jsonl(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def adapt_ramr(blind_path, labels_path):
    labels = {r["chain_id"]: r["ground_truth"] for r in _load_jsonl(labels_path)}
    out = []
    for r in _load_jsonl(blind_path):
        gt = labels.get(r["chain_id"]) or {}
        p = gt.get("planted") or {}
        if not p.get("record_id"):
            continue
        items = [{"id": i["record_id"], "text": i.get("text", "")} for i in r["retrieved"]]
        out.append(Trace(r["chain_id"], r.get("query"), items, p["record_id"], p.get("replaces")))
    return out


def adapt_locomo(path, _unused=None, single_evidence_only=True):
    """LoCoMo (snap-research/locomo) -- the category's most-quoted benchmark.

    The retrieval question it poses: given a question and every dialogue turn of a long conversation,
    which turn holds the evidence? That is a partial-input setting whether or not it was built as one,
    so the audit asks how often the evidence turn can be picked WITHOUT reading the question.

    Only single-evidence questions are used by default (1,559 of them). A multi-evidence item has no
    single target, and scoring it as if it did would credit a probe for finding any one of several
    right answers -- inflating exactly the number this tool exists to keep honest.

    The dataset is NOT redistributed here; pass the path to your own copy.
    """
    data = json.load(io.open(path, encoding="utf-8"))
    out = []
    for sample in data:
        conv = sample.get("conversation") or {}
        items = []
        for k in sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                        key=lambda s: int(s.split("_")[1])):
            for turn in conv[k] or []:
                if turn.get("dia_id"):
                    items.append({"id": turn["dia_id"], "text": turn.get("text") or "",
                                  "speaker": turn.get("speaker")})
        if not items:
            continue
        present = {i["id"] for i in items}
        for n, qa in enumerate(sample.get("qa") or []):
            ev = qa.get("evidence") or []
            if single_evidence_only and len(ev) != 1:
                continue
            if not ev or ev[0] not in present:
                continue
            out.append(Trace("%s-q%d" % (sample.get("sample_id"), n),
                             qa.get("question"), items, ev[0], None))
    return out


def adapt_generic(blind_path, labels_path, target_key, decoy_key):
    labels = {}
    for r in _load_jsonl(labels_path):
        tid = r.get("id") or r.get("chain_id") or r.get("trace_id")
        labels[tid] = r
    out = []
    for r in _load_jsonl(blind_path):
        tid = r.get("id") or r.get("chain_id") or r.get("trace_id")
        lab = labels.get(tid) or {}
        flat = dict(lab)
        flat.update(lab.get("ground_truth") or {})
        tgt = flat.get(target_key)
        if isinstance(tgt, dict):
            tgt = tgt.get("record_id") or tgt.get("id")
        dec = flat.get(decoy_key)
        if isinstance(dec, dict):
            dec = dec.get("record_id") or dec.get("id")
        if not tgt:
            continue
        raw = r.get("retrieved") or r.get("items") or r.get("candidates") or []
        items = []
        for i in raw:
            d = dict(i)
            d["id"] = d.pop("record_id", None) or d.get("id")
            items.append(d)
        out.append(Trace(tid, r.get("query") or r.get("question"), items, tgt, dec))
    return out


def pct(x):
    return "  n/a " if x is None else "%5.1f%%" % (100 * x)


def main(argv=None):
    ap = argparse.ArgumentParser(description="partial-input baseline battery for retrieval benchmarks")
    ap.add_argument("--adapter", choices=["ramr", "locomo", "generic", "demo"], default="generic")
    ap.add_argument("--blind", default="ramr_traces_v0.2_blind.jsonl")
    ap.add_argument("--labels", default="ramr_traces_v0.2_labels.jsonl")
    ap.add_argument("--target", default="planted", help="generic adapter: label key for the answer")
    ap.add_argument("--decoy", default="replaces", help="generic adapter: label key for its pair")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)

    needed = [] if a.adapter == "demo" else ([a.blind] if a.adapter == "locomo" else [a.blind, a.labels])
    for p in needed:
        if not os.path.exists(p):
            print("MISSING: %s" % p)
            return 2

    if a.adapter == "demo":
        traces = demo_traces()
    elif a.adapter == "ramr":
        traces = adapt_ramr(a.blind, a.labels)
    elif a.adapter == "locomo":
        traces = adapt_locomo(a.blind)
    else:
        traces = adapt_generic(a.blind, a.labels, a.target, a.decoy)
    if not traces:
        print("no traces loaded -- an empty audit is a refusal, not a pass")
        return 2

    print("memaudit %s -- %d traces, %d candidates median\n"
          % (__version__, len(traces),
             sorted(len(t.items) for t in traces)[len(traces) // 2]))
    rows = audit(traces)
    hdr = "%-24s %8s %9s %9s %8s  %s" % ("probe", "coverage", "accuracy", "null", "lift", "verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-24s %8s %9s %9s %8s  %s"
              % (r["probe"], pct(r["coverage"]), pct(r["accuracy"]), pct(r["null_accuracy"]),
                 pct(r["lift"]), "REAL" if r["real"] else "at chance"))

    env = envelope(traces)
    print("\nSHORTCUT FLOOR: %s accuracy over %s of traces, using %s"
          % (pct(env["accuracy"]), pct(env["coverage"]),
             ", ".join(env["probes_used"]) or "nothing"))
    print("  Publish this beside every score. It is a LOWER bound on what this benchmark can be")
    print("  solved by without understanding it -- these rules, not all possible rules.")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"version": __version__, "traces": len(traces),
                        "probes": rows, "floor": env}, indent=2) + "\n")
        print("wrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
