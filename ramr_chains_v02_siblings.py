"""v0.2.0 of the chain corpus: give every chain a same-type entity that is the SUBJECT of a retrievable fact.

WHY. memaudit's surviving cue on the v0.3 traces is value-echo: 47.7% coverage at 100.0% accuracy. It is
a property of RETRIEVAL, not of the plant. The current value is the subject of the next gold fact, so it
is always retrieved; a stale value borrowed from a distractor is the subject of a fact that lexical
retrieval rarely returns, so it echoes nothing. No choice of donor fixes that -- the donor has to be the
subject of something the retriever actually brings back.

WHICH IS FIXABLE, and the measurement said so before this file existed. The echo cue fires on 143 traces:
82 planted on "P works at C" and 61 on "C is headquartered in CTRY". ZERO on "The currency of CTRY is the
CUR" -- a currency is never a subject here, so neither value echoes and the pair is already balanced. The
slot I had assumed was the structural blocker contributes nothing, and the whole firing set is reachable.

WHAT THIS ADDS. Two sibling facts per chain, built from the chain's OWN distractor vocabulary:

    {c2} is headquartered in {ctry}.        -> c2 is a company AND a subject, and it shares the chain's
                                               country, so lexical retrieval brings it back
    The currency of {ctry2} is the {cur2}.  -> ctry2 is a country AND a subject, and it shares the
                                               question's strongest token, "currency"

They go at the FRONT of the distractor pool, because the export stores only the first 12.

A NEW IDENTIFIER, NOT AN EDIT. v0.1.0 is left exactly as it is and stays pinned. Everything else about
each chain -- id, question, gold facts, answer, drop_index -- is copied unchanged, so every existing
metric asks the same questions of the same answers; only the distractor pool grows. Silently rewriting a
published corpus is the failure this repository already committed once today.

WHETHER IT WORKS IS NOT ASSERTED HERE. Re-cut the traces against this corpus and run memaudit. If the
sibling is not retrieved, the cue does not move, and the number will say so.

    python ramr_chains_v02_siblings.py
"""
import io
import json
import os
import sys

SRC = os.path.join("data", "ramr_chains_v0.1.0.jsonl")
OUT = os.path.join("data", "ramr_chains_v0.2.0.jsonl")


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def _entities(chain):
    """(person, company, country, currency) from the three gold facts."""
    f = chain["gold_facts"]
    if len(f) < 3:
        return None
    company = f[0].rstrip(".").split()[-1]
    country = f[1].rstrip(".").split()[-1]
    currency = f[2].rstrip(".").split()[-1]
    return company, country, currency


def _pool_entities(chain):
    """Companies and (country, currency) pairs available in this chain's own distractor vocabulary."""
    companies, countries, currencies = [], [], []
    for d in chain.get("distractor_pool") or []:
        w = d.rstrip(".").split()
        if len(w) >= 5 and w[1] == "is" and w[2] == "headquartered":
            companies.append(w[0])
            countries.append(w[-1])
        elif len(w) >= 7 and w[0].lower() == "the" and w[1].lower() == "currency":
            countries.append(w[3])
            currencies.append(w[-1])
    return companies, countries, currencies


def _closest(cands, like):
    """Same capitalisation, nearest length, deterministic tie-break -- so the sibling donor does not
    reintroduce the casing and length cues the re-cut just removed."""
    ok = [x for x in cands if x != like and x[:1].isupper() == like[:1].isupper()]
    if not ok:
        return None
    return sorted(ok, key=lambda s: (abs(len(s) - len(like)), s))[0]


def main():
    if not os.path.exists(SRC):
        print("MISSING: %s" % SRC)
        return 2
    chains = load(SRC)
    added = skipped = 0
    out = []
    for ch in chains:
        ents = _entities(ch)
        if not ents:
            out.append(ch)
            skipped += 1
            continue
        company, country, currency = ents
        companies, countries, currencies = _pool_entities(ch)

        sibs = []
        c2 = _closest(companies, company)
        if c2:
            sibs.append("%s is headquartered in %s." % (c2, country))
        ctry2 = _closest(countries, country)
        cur2 = _closest(currencies, currency) if currencies else None
        if ctry2 and cur2:
            sibs.append("The currency of %s is the %s." % (ctry2, cur2))

        if not sibs:
            skipped += 1
        else:
            added += 1
        new = dict(ch)
        # Front of the pool: the export stores only the first 12, and a sibling nobody stores is a
        # sibling nobody can retrieve.
        new["distractor_pool"] = sibs + [d for d in (ch.get("distractor_pool") or []) if d not in sibs]
        out.append(new)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for ch in out:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("chains: %d | siblings added: %d | no donor available: %d" % (len(out), added, skipped))
    print("questions, answers, gold facts and drop_index are unchanged from v0.1.0")
    print("wrote %s" % OUT)
    print("\nnext: re-cut traces against this corpus, then audit -- the number decides, not this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
