"""RAMR ABSTENTION / false-recall metric (validates mnemo's new relevance-floor). A memory layer that confabulates
a weak false match when the queried fact is NOT in the store is worse than one that says "not in memory". Test:
store M templated facts; probe with (a) IN-STORE queries (the stored facts) and (b) OUT-OF-STORE queries (same
attribute, an entity that was never stored -> partially overlaps stored facts via the shared attribute word, so a
naive retriever returns a WRONG fact). Sweep recall(min_relevance=floor) and measure:
  in-store recall  = fraction of in-store probes whose top-1 is the correct fact
  abstention-prec  = fraction of out-of-store probes where recall returns [] (correctly abstains)
A good floor maximizes both. FALSIFIER (pre-registered): if no floor gives abstention-prec >= 0.8 while keeping
in-store recall >= 0.9, the relevance-floor does not buy clean abstention on this task. Cloud-free, lexical. ASCII."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnemo import Mnemo

M = int(os.getenv("ABS_M", "40"))
N_SEEDS = int(os.getenv("ABS_SEEDS", "5"))
FLOORS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
ATTRS = ["retrylimit", "timeoutsec", "batchsize", "cachettl", "concurrency"]
L = "abcdefghijklmnopqrstuvwxyz"


def build(seed):
    r = np.random.default_rng(seed)
    store = Mnemo(path=None, embed=None); store.semantic_threshold = 10 ** 9
    instore, outstore = [], []
    for i in range(M):
        sfx = L[i // 26] + L[i % 26]
        ent = f"ent{sfx}"; attr = ATTRS[i % len(ATTRS)]; val = f"val{sfx}"
        mid = store.remember(f"{ent} {attr} {val}", tags=["cfg"], value=1.0, mtype="semantic")
        instore.append({"q": f"{ent} {attr}", "id": mid})
        # an entity that was NEVER stored, same attribute -> partially overlaps via the shared attr word
        outstore.append({"q": f"newent{sfx}z {attr}"})
    return store, instore, outstore


def run(seed):
    store, instore, outstore = build(seed)
    rows = {}
    for fl in FLOORS:
        inhit = sum(1 for p in instore if (lambda res: res and res[0]["id"] == p["id"])
                    (store.recall(p["q"], k=3, mode="lexical", min_relevance=fl))) / len(instore)
        abst = sum(1 for p in outstore if not store.recall(p["q"], k=3, mode="lexical", min_relevance=fl)) / len(outstore)
        rows[fl] = (inhit, abst)
    return rows


if __name__ == "__main__":
    print(f"compile OK - RAMR ABSTENTION metric (M={M}, seeds={N_SEEDS}, floors={FLOORS})", flush=True)
    agg = {fl: ([], []) for fl in FLOORS}
    for s in range(N_SEEDS):
        rows = run(s)
        for fl in FLOORS:
            agg[fl][0].append(rows[fl][0]); agg[fl][1].append(rows[fl][1])
    print(f"\n  min_relevance | in-store recall | abstention-precision", flush=True)
    res = {}
    for fl in FLOORS:
        ir = float(np.mean(agg[fl][0])); ap = float(np.mean(agg[fl][1])); res[fl] = (ir, ap)
        print(f"    {fl:.1f}         |   {ir:.2f}          |   {ap:.2f}", flush=True)
    good = [fl for fl in FLOORS if res[fl][1] >= 0.8 and res[fl][0] >= 0.9]
    print(f"\n  === VERDICT (pre-registered) ===", flush=True)
    if good:
        best = max(good, key=lambda fl: res[fl][0] + res[fl][1])
        print(f"  RELEVANCE-FLOOR BUYS CLEAN ABSTENTION: at min_relevance={best:.1f}, recall correctly abstains on "
              f"{res[best][1]:.0%} of out-of-store probes while keeping {res[best][0]:.0%} in-store recall. Without "
              f"a floor (0.0) it confabulates a wrong fact on {1-res[0.0][1]:.0%} of out-of-store probes. The "
              f"feature works: mnemo can say 'not in memory' instead of returning noise -- a real reliability win "
              f"and the substrate for verify-before-citing.", flush=True)
    else:
        print(f"  no floor cleanly separated (abstention>=0.8 & in-recall>=0.9). Report honestly; tune the metric "
              f"or the sim measure before shipping a default floor.", flush=True)
    json.dump({"M": M, "seeds": N_SEEDS, "by_floor": {str(fl): res[fl] for fl in FLOORS}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abstention_result.json"), "w"))
    print("DONE", flush=True)
