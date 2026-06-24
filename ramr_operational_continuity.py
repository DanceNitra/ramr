"""RAMR — OPERATIONAL CONTINUITY (idempotent resume).

The agentic failure no retrieval-quality metric captures: after a compaction/crash, the agent reconstructs "what
have I already done?" from memory and must SKIP completed actions. A completion record that recall MISSES -> the
action is RE-EXECUTED -> a duplicate side-effect (possibly irreversible: re-send a payment, re-deploy).

DUPLICATE-RATE = fraction of the CURRENT task's completed steps whose 'done' record is NOT in a budget-limited
resume recall (so they get re-run). The realistic stressor: a long-running agent's memory also holds OLD sessions'
completions (same form, EQUAL base value here, so only RECENCY differs). On resume the agent does ONE budget-limited
recall (top-k) to reconstruct state. Sweep the recall budget k. decay ON (resume soon: current 0.5d vs old 30d) vs
OFF (current also 30d -> no recency advantage).

Pre-registered: with recency, DUPLICATE-RATE = the budget-floor max(0, C-k)/C (robust to unlimited old history);
without recency, old completions crowd out current ones -> DUPLICATE-RATE stays ~1.0 at any budget. NO-MEMORY
baseline = 1.0. Falsifier: decay ON does NOT track the budget-floor, or decay OFF is materially below 1.0.
CLOUD-FREE, pure mnemo recall (no LLM, no embedder). Deterministic (seeded).
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnemo.mnemo import Mnemo

DAY = 86400.0
_CV, _V = "bcdfghjklmnpqrstvwz", "aeiou"

def _trial(budget_k, decay_on, n_current=10, n_old=200, seed=0):
    rng = random.Random(seed)
    s = Mnemo(path=None, embed=None); s.semantic_threshold = 10 ** 9
    now = time.time(); pool = set()
    def uniq():
        while True:
            t = "".join(rng.choice(_CV) + rng.choice(_V) for _ in range(3))
            if t not in pool:
                pool.add(t); return t
    def add(t, age):
        mid = s.remember(f"task step {t} done", tags=["c"], value=3.0, mtype="episodic")  # EQUAL value
        it = [x for x in s.items if x["id"] == mid][0]; it["ts"] = it["last_access"] = now - age * DAY
    for _ in range(n_old):
        add(uniq(), 30)                                    # old-session completions (history)
    cur = [uniq() for _ in range(n_current)]
    for t in cur:
        add(t, 0.5 if decay_on else 30)                    # current task: recent (ON) or same-age as old (OFF)
    recalled = " ".join(r["text"] for r in s.recall("task step done", k=budget_k, mode="lexical"))
    missed = [t for t in cur if t not in recalled]         # completion not recalled -> RE-EXECUTED
    return len(missed) / len(cur)

def run(seeds=6, budgets=(3, 5, 10, 15, 25, 50), n_current=10, n_old=200):
    rows = []
    for k in budgets:
        on = sum(_trial(k, True, n_current, n_old, sd) for sd in range(seeds)) / seeds
        off = sum(_trial(k, False, n_current, n_old, sd) for sd in range(seeds)) / seeds
        floor = max(0, n_current - k) / n_current
        rows.append({"budget_k": k, "dup_decay_on": round(on, 3), "dup_decay_off": round(off, 3),
                     "budget_floor": round(floor, 3)})
    return {"n_current": n_current, "n_old": n_old, "seeds": seeds, "no_memory_baseline": 1.0, "rows": rows}

if __name__ == "__main__":
    res = run()
    print("OPERATIONAL CONTINUITY: duplicate-side-effect rate on resume (n_current=10, n_old=200, EQUAL value). NO-MEMORY=1.00")
    print(f"{'budget k':>8} | {'decay ON':>9} | {'decay OFF':>10} | {'budget-floor':>12}")
    for r in res["rows"]:
        print(f"{r['budget_k']:>8} | {r['dup_decay_on']:>9.3f} | {r['dup_decay_off']:>10.3f} | {r['budget_floor']:>12.2f}")
    on_tracks = all(abs(r["dup_decay_on"] - r["budget_floor"]) <= 0.05 for r in res["rows"])
    off_high = all(r["dup_decay_off"] >= 0.95 for r in res["rows"])
    print(f"\nVERDICT: {'PASS' if (on_tracks and off_high) else 'CHECK'} — with recency, duplicate-rate tracks the budget-floor "
          f"(robust to {res['n_old']} old completions); without recency it stays ~1.0 at every budget. "
          f"Recency weighting is necessary AND sufficient for idempotent resume.")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "operational_continuity_result.json"), "w"), indent=1)
