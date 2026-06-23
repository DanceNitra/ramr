"""RAMR candidate 6th metric -- FORGET-PRECISION: after a fact is UPDATED, does recall return the CURRENT value or
the STALE one? A memory layer that can't forget superseded info silently serves outdated facts. mnemo has a
state-toggle supersession pass (consolidate() marks the older of a contradicting pair `superseded`, and recall
excludes superseded by default). But its toggle detector `_negation_clash` only fires on an explicit polarity flip
(not/no/never/false) -- so we test TWO regimes:
  NEGATION    : F2 explicitly negates F1 ('X is reliable' -> 'X is NOT reliable')  -> SHOULD supersede.
  VALUE-UPDATE: F2 changes a number  ('retry limit is 5' -> 'retry limit is 12')   -> no negation word.
To isolate supersession, the STALE F1 is given HIGHER value than the current F2, so relevance x value would return
the stale fact UNLESS supersession removes it. Arms: NO-CONSOLIDATE (baseline) vs CONSOLIDATE (run the dream pass).
FORGET-PRECISION = fraction of topics where recall's top-1 is the CURRENT fact (F2). Cloud-free, lexical (no
embedder). Pre-registered falsifier: if CONSOLIDATE does not raise forget-precision over baseline in the NEGATION
regime, supersession is decorative. Prediction: NEGATION -> high after consolidate; VALUE-UPDATE -> stays low (a
real gap: silent numeric updates are merged as duplicates, not superseded). ASCII prints."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnemo import Mnemo

M_TOPICS = int(os.getenv("FP_M", "30"))
N_SEEDS = int(os.getenv("FP_SEEDS", "6"))
ENTS = ["payment api","auth service","search index","billing job","cache layer","upload queue","report engine",
        "email worker","session store","rate limiter","image cdn","audit log","webhook relay","config loader",
        "metrics sink","backup task","login flow","export tool","notify bus","schema migrator","token vault",
        "flag svc","data lake","shard router","pdf renderer","geo lookup","fraud check","recommend svc",
        "chat gateway","trace collector"]


def topics(regime, seed):
    # fully-unique per-topic tokens (tok + prop) so a query matches ONLY its own pair -- no cross-topic
    # interference from other topics' high-value stale facts (which share generic words otherwise).
    r = np.random.default_rng(seed); out = []
    L = "abcdefghijklmnopqrstuvwxyz"
    # EVERY content token is unique per topic (entity, attribute, predicate, value) so different topics share NO
    # words -> avoids mnemo's hub-flagging (a word common to the whole corpus marks every memory a "universal
    # matcher" hub, which recall then skips). 'not' is a stopword, so it never contributes to overlap/hubs; it
    # only triggers the negation clash. This isolates the supersession behaviour we are measuring.
    for i in range(M_TOPICS):
        sfx = L[i // 26] + L[i % 26]                       # pure-alpha unique suffix (tokenizer keeps it)
        tok = f"vox{sfx}"; prop = f"qal{sfx}"; verb = f"act{sfx}"
        if regime == "negation":
            f1 = f"{tok} {prop} {verb}"                    # older / stale
            f2 = f"{tok} {prop} not {verb}"                # newer / current (negates f1)
        else:                                              # value-update (NUMERIC: the common 'fact's value changed' case;
            v1 = int(r.integers(2, 50)); v2 = v1 + int(r.integers(50, 99))   # digits aren't tokens -> no cross-topic/hub issue
            f1 = f"{tok} {prop} {verb} {v1}"               # older / stale value
            f2 = f"{tok} {prop} {verb} {v2}"               # newer / current value
        out.append({"f1": f1, "f2": f2, "q": f"{tok} {prop}"})
    return out


def run(regime, consolidate, seed):
    tps = topics(regime, seed)
    store = Mnemo(path=None, embed=None); store.semantic_threshold = 10 ** 9
    cur_ids = []
    for tp in tps:
        store.remember(tp["f1"], tags=["f"], value=3.0, mtype="semantic")   # STALE gets HIGHER value
        mid2 = store.remember(tp["f2"], tags=["f"], value=1.0, mtype="semantic")  # CURRENT, lower value
        cur_ids.append(mid2)
    if consolidate:
        store.consolidate(dup_threshold=0.5)               # the dream pass: toggle-supersede contradicting pairs
    hits = 0
    for tp, cur in zip(tps, cur_ids):
        res = store.recall(tp["q"], k=3, mode="lexical")   # default excludes superseded
        if res and res[0]["id"] == cur:
            hits += 1
    return hits / len(tps)


if __name__ == "__main__":
    print(f"compile OK - FORGET-PRECISION (M={M_TOPICS}, seeds={N_SEEDS}; stale fact given HIGHER value)", flush=True)
    out = {}
    for regime in ("negation", "value_update"):
        base = np.array([run(regime, False, s) for s in range(N_SEEDS)])
        cons = np.array([run(regime, True, s) for s in range(N_SEEDS)])
        d = cons - base
        rng = np.random.default_rng(7); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[regime] = {"baseline": float(base.mean()), "consolidate": float(cons.mean()),
                       "lift": float(d.mean()), "ci": [float(lo), float(hi)]}
        print(f"\n  {regime.upper()}: forget-precision  baseline {base.mean():.2f} -> consolidate {cons.mean():.2f}  "
              f"(lift {d.mean():+.2f}, CI [{lo:+.2f},{hi:+.2f}])", flush=True)
    neg = out["negation"]; val = out["value_update"]
    print(f"\n  === VERDICT (pre-registered) ===", flush=True)
    neg_works = neg["consolidate"] >= 0.8 and neg["ci"][0] > 0
    val_gap = val["consolidate"] < 0.5
    if neg_works and val_gap:
        print(f"  SUPERSESSION WORKS FOR CONTRADICTIONS, BUT HAS A VALUE-UPDATE GAP. After the dream pass, recall "
              f"returns the CURRENT fact {neg['consolidate']:.0%} of the time when the update is an explicit "
              f"negation (vs {neg['baseline']:.0%} without it -- the stale higher-value fact otherwise wins). But a "
              f"silent NUMERIC update is merged as a near-duplicate, NOT superseded, so recall still serves the "
              f"stale value {1-val['consolidate']:.0%} of the time ({val['consolidate']:.0%} current). FORGET-"
              f"PRECISION is a real 6th metric: a memory layer's ability to forget is only as good as its "
              f"contradiction detector -- here, polarity flips are caught, value-updates leak. Actionable: detect "
              f"same-entity/same-attribute value changes as toggles, not duplicates.", flush=True)
    elif neg_works:
        print(f"  Supersession lifts forget-precision for negations ({neg['baseline']:.2f}->{neg['consolidate']:.2f}); "
              f"value-update regime at {val['consolidate']:.2f} (gap not as sharp as predicted). Report as measured.", flush=True)
    else:
        print(f"  Supersession did NOT reliably raise forget-precision even for negations "
              f"({neg['baseline']:.2f}->{neg['consolidate']:.2f}, CI [{neg['ci'][0]:+.2f},{neg['ci'][1]:+.2f}]). "
              f"Honest negative -- the toggle pass is weaker than expected on this task.", flush=True)
    json.dump({"M": M_TOPICS, "seeds": N_SEEDS, "results": out},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "forget_precision_result.json"), "w"))
    print("DONE", flush=True)
