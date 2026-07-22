"""RAMR CROSS-SCOPE LEAKAGE metric (validates inspeximus's new recall(scope=) isolation). A shared memory store (many
agents/tenants in one Inspeximus -- exactly Agora's 8-agent dungeon) must not bleed one scope's memories into another's
recall. Setup: M topics; for each, store an A-scoped fact and a B-scoped fact that share the SAME entity+attribute
tokens (same schema, different tenant) but different values. Query as tenant B:
  no-scope   : recall(q)            -> leakage if the returned top-1 is an A-scoped fact
  scoped     : recall(q, scope='B') -> leakage MUST be ~0; also measure in-scope recall (returns B's own fact)
FALSIFIER (pre-registered): if scoped leakage is not ~0 while no-scope leakage is substantial, the isolation
filter doesn't work. Cloud-free, lexical. ASCII."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus

M = int(os.getenv("SCOPE_M", "40"))
N_SEEDS = int(os.getenv("SCOPE_SEEDS", "5"))
ATTRS = ["retrylimit", "timeoutsec", "batchsize", "cachettl", "concurrency"]
L = "abcdefghijklmnopqrstuvwxyz"


def build(seed):
    store = Inspeximus(path=None, embed=None); store.semantic_threshold = 10 ** 9
    A_ids, B_ids, probes = set(), set(), []
    for i in range(M):
        sfx = L[i // 26] + L[i % 26]; attr = ATTRS[i % len(ATTRS)]
        aid = store.remember(f"ent{sfx} {attr} valA{sfx}", tags=["t"], value=1.0, meta={"scope": "A"})
        bid = store.remember(f"ent{sfx} {attr} valB{sfx}", tags=["t"], value=1.0, meta={"scope": "B"})
        A_ids.add(aid); B_ids.add(bid)
        probes.append({"q": f"ent{sfx} {attr}", "bid": bid})
    return store, A_ids, B_ids, probes


def run(seed):
    store, A_ids, B_ids, probes = build(seed)
    leak_no = leak_sc = inrec_sc = 0
    for p in probes:
        rn = store.recall(p["q"], k=3, mode="lexical")                 # no scope
        if rn and rn[0]["id"] in A_ids: leak_no += 1
        rs = store.recall(p["q"], k=3, mode="lexical", scope="B")       # scoped to B
        if any(x["id"] in A_ids for x in rs): leak_sc += 1
        if rs and rs[0]["id"] == p["bid"]: inrec_sc += 1
    n = len(probes)
    return leak_no / n, leak_sc / n, inrec_sc / n


if __name__ == "__main__":
    print(f"compile OK - RAMR CROSS-SCOPE LEAKAGE metric (M={M}, seeds={N_SEEDS}; A+B share schema in one store)", flush=True)
    ln, ls, ir = np.mean([run(s) for s in range(N_SEEDS)], axis=0)
    print(f"\n  no-scope leakage  (top-1 is an A-fact when querying as B): {ln:.2f}", flush=True)
    print(f"  scoped leakage    (any A-fact returned with scope='B')    : {ls:.2f}", flush=True)
    print(f"  scoped in-recall  (top-1 is B's own fact)                 : {ir:.2f}", flush=True)
    print(f"\n  === VERDICT (pre-registered) ===", flush=True)
    if ls <= 0.02 and ln >= 0.2 and ir >= 0.9:
        print(f"  SCOPE ISOLATION WORKS: with recall(scope='B') cross-scope leakage is {ls:.0%} (A never bleeds in) "
              f"and in-scope recall stays {ir:.0%}, while WITHOUT a scope the shared store leaks an A-tenant fact "
              f"into B's recall {ln:.0%} of the time. A real multi-tenant / multi-agent isolation guarantee for the "
              f"shared store -- security-relevant for the 8-agent dungeon. Clean win.", flush=True)
    else:
        print(f"  inspect: scoped leakage {ls:.2f}, no-scope leakage {ln:.2f}, scoped in-recall {ir:.2f} "
              f"(want scoped~0, no-scope high, in-recall high).", flush=True)
    json.dump({"M": M, "seeds": N_SEEDS, "leak_noscope": float(ln), "leak_scoped": float(ls), "inrecall_scoped": float(ir)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scope_leakage_result.json"), "w"))
    print("DONE", flush=True)
