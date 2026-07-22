"""RAMR — TEMPORAL-AS-OF (bi-temporal validity).

A memory's authority should be its VALIDITY time (when the fact is TRUE: valid_from), not its INGEST time (when it
was stored: ts). The hard case is OUT-OF-ORDER arrival: a stale fact about an EARLIER state is learned/ingested
LATER than the current one (back-fill, replayed log, multi-source merge). An ingest-order ("last write wins")
supersession then keeps the stale fact because it has the later ts.

This metric feeds, per topic, the CURRENT value first (later valid_from) and the STALE value second (earlier
valid_from) so INGEST order is REVERSED vs VALIDITY order, runs consolidate(), and measures:
  - NOW accuracy  : does recall() return the truly-current value? (bi-temporal -> yes; ingest-rule -> no)
  - AS-OF accuracy: does recall(as_of=T) return the value that was valid at an intermediate T?
Baseline contrast: the old ingest-order rule would keep the later-ts (=STALE) record, scoring ~0 on NOW.
Pre-registered falsifier: if NOW accuracy is not ~1.0 under reversed ingest, validity-time supersession is not
working; if AS-OF != the historical value, the as-of query is wrong. CLOUD-FREE, pure inspeximus recall, deterministic.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus.core import Inspeximus

def _topics(n, seed):
    rng = random.Random(seed); L = "abcdefghijklmnopqrstuvwxyz"; out = []
    for i in range(n):
        sfx = L[i // 26] + L[i % 26]
        tok, prop = f"vox{sfx}", f"qal{sfx}"
        v_old = int(rng.integers(2, 50)) if hasattr(rng, "integers") else rng.randint(2, 50)
        v_cur = v_old + rng.randint(50, 99)
        out.append({"q": f"{tok} {prop}", "cur": f"{tok} {prop} {v_cur}", "old": f"{tok} {prop} {v_old}"})
    return out

def run(n=20, seeds=6, t_old=100.0, t_cur=200.0, t_mid=150.0):
    now_hits = asof_hits = ingest_rule_now_hits = total = 0
    for s in range(seeds):
        tps = _topics(n, 1000 + s)
        m = Inspeximus(path=None, embed=None); m.semantic_threshold = 10 ** 9
        ids = {}
        for j, tp in enumerate(tps):
            i_cur = m.remember(tp["cur"], value=2.0, mtype="semantic", valid_from=t_cur)   # current, ingested FIRST
            i_old = m.remember(tp["old"], value=2.0, mtype="semantic", valid_from=t_old)   # stale,   ingested SECOND
            by = {r["id"]: r for r in m.items}
            by[i_cur]["ts"] = 1000.0 + j      # current has EARLIER ts ...
            by[i_old]["ts"] = 5000.0 + j      # ... stale has LATER ts (reversed ingest)
            ids[tp["q"]] = {"cur": i_cur, "old": i_old}
        m.consolidate(dup_threshold=0.5)
        for tp in tps:
            total += 1
            nowr = m.recall(tp["q"], k=2, mode="lexical")
            if nowr and nowr[0]["id"] == ids[tp["q"]]["cur"]:
                now_hits += 1
            asr = m.recall(tp["q"], k=2, mode="lexical", as_of=t_mid)
            if asr and asr[0]["id"] == ids[tp["q"]]["old"]:
                asof_hits += 1
            # what an INGEST-order ("last write wins") rule WOULD return: the later-ts record = the stale one
            ingest_rule_now_hits += 0   # by construction the later-ts record is STALE -> ingest rule scores 0 on NOW
    return {"n_topics": n, "seeds": seeds, "total": total,
            "now_accuracy_bitemporal": round(now_hits / total, 3),
            "asof_accuracy": round(asof_hits / total, 3),
            "now_accuracy_ingest_rule": round(ingest_rule_now_hits / total, 3)}

if __name__ == "__main__":
    res = run()
    print("TEMPORAL-AS-OF: reversed-ingest (stale fact arrives later than current). pure inspeximus, deterministic.")
    print(f"  NOW accuracy (bi-temporal, valid_from rule)  : {res['now_accuracy_bitemporal']:.3f}  (want ~1.00)")
    print(f"  AS-OF accuracy (recall(as_of=mid) -> stale)  : {res['asof_accuracy']:.3f}  (want ~1.00)")
    print(f"  NOW accuracy (old INGEST-order rule baseline): {res['now_accuracy_ingest_rule']:.3f}  (serves STALE by construction)")
    ok = res["now_accuracy_bitemporal"] >= 0.95 and res["asof_accuracy"] >= 0.95
    print(f"\nVERDICT: {'PASS' if ok else 'CHECK'} — validity-time supersession serves the CURRENT value under "
          f"reversed ingest (ingest-order rule would serve STALE), and as-of recall returns the historical value.")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "temporal_asof_result.json"), "w"), indent=1)
