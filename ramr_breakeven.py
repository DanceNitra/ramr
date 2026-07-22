"""RAMR break-even sweep + the test that decides which inspeximus cal_mode ships. Our signal-reliability law says the
outcome-credit (was-it-right) channel only beats relevance once the credit signal's reliability p exceeds the
no-signal floor 1/(1+D). The legacy cal (mode 'full', [0.5,1.5]) BACKFIRES below that floor (a wrong credit
durably suppresses a correct memory). We test two safer modes against it:
  full  : cal in [0.5,1.5] (legacy)            -> expected: lift>0 above break-even, NEGATIVE (backfire) below.
  boost : cal in [1.0,1.5] (promote-only)      -> expected: no backfire (lift>=~0), gain preserved above.
  gated : disable cal when pooled signal <= 1/(1+D) -> expected: ~0 below break-even, gain preserved above.
Setup: lexical (no embedder), M topics each = 1 correct + D distractors sharing the query tokens (equal sim ->
ranking decided by value x cal). Each session: recall top-1; credit it with the TRUE outcome w.p. p, else flipped.
Lift = final-session accuracy(mode) - accuracy(no-credit baseline). FALSIFIER for shipping a mode: it must (a) keep
lift >= ~0 across ALL p (no backfire) and (b) preserve a positive lift where p > 1/(1+D). Cloud-free. ASCII."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus

M = int(os.getenv("BE_M", "24"))
T = int(os.getenv("BE_T", "8"))
N_SEEDS = int(os.getenv("BE_SEEDS", "3"))
P_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
D_LEVELS = [1, 3]
L = "abcdefghijklmnopqrstuvwxyz"


def topics(seed, D):
    r = np.random.default_rng(seed); out = []
    for i in range(M):
        sfx = L[i // 26] + L[i % 26]
        tok = f"tok{sfx}a tok{sfx}b"                       # shared per-topic query tokens
        cands = [{"text": f"{tok} cwrd{sfx}", "ok": True}]
        for j in range(D):
            cands.append({"text": f"{tok} dwrd{sfx}{L[j]}", "ok": False})   # equal sim to query
        r.shuffle(cands)
        out.append({"q": tok, "cands": cands})
    return out


def run(mode, p, D, seed):
    rr = np.random.default_rng(seed * 1000 + int(p * 100) + D)
    tps = topics(seed, D)
    store = Inspeximus(path=None, embed=None); store.semantic_threshold = 10 ** 9
    store.cal_mode = "full" if mode == "none" else mode    # baseline uses 'full' but never credits
    correct = []
    for tp in tps:
        cs = set()
        for c in tp["cands"]:
            mid = store.remember(c["text"], tags=["x"], value=1.0, mtype="semantic")
            if c["ok"]: cs.add(mid)
        correct.append(cs)
    acc = []
    for _ in range(T):
        hits = 0
        for tp, cs in zip(tps, correct):
            res = store.recall(tp["q"], k=1 + D, mode="lexical")
            if not res: continue
            top = res[0]; ok = top["id"] in cs; hits += int(ok)
            if mode != "none":
                credited = ok if rr.random() < p else (not ok)   # signal of reliability p
                store.credit([top["id"]], outcome=credited)
        acc.append(hits / len(tps))
    return acc[-1]


if __name__ == "__main__":
    print(f"compile OK - RAMR break-even sweep (M={M}, T={T}, seeds={N_SEEDS}, modes=full/boost/gated)", flush=True)
    res = {}
    for D in D_LEVELS:
        base = np.mean([run("none", 1.0, D, s) for s in range(N_SEEDS)])
        floor = 1.0 / (1.0 + D)
        print(f"\n  D={D} (break-even floor 1/(1+D)={floor:.2f}); no-credit baseline acc={base:.2f}", flush=True)
        print(f"  p     | full lift | boost lift | gated lift   (lift = mode_acc - baseline)", flush=True)
        for p in P_LEVELS:
            row = {}
            for mode in ("full", "boost", "gated"):
                a = np.mean([run(mode, p, D, s) for s in range(N_SEEDS)])
                row[mode] = float(a - base)
            res[f"D{D}_p{p}"] = row
            flag = "  <- above floor" if p > floor else "  (below floor)"
            print(f"  {p:.1f}   |  {row['full']:+.2f}    |  {row['boost']:+.2f}     |  {row['gated']:+.2f}{flag}", flush=True)
    # decision: which mode never backfires AND preserves gain above the floor?
    def worst(mode):  return min(v[mode] for v in res.values())
    def gain_above(mode): return np.mean([res[f"D{D}_p{p}"][mode] for D in D_LEVELS for p in P_LEVELS if p > 1/(1+D)])
    print(f"\n  === VERDICT (which cal_mode to ship) ===", flush=True)
    for mode in ("full", "boost", "gated"):
        print(f"  {mode:6s}: worst lift (backfire risk) {worst(mode):+.2f} | mean gain above floor {gain_above(mode):+.2f}", flush=True)
    cands = [m for m in ("boost", "gated") if worst(m) >= -0.05 and gain_above(m) >= 0.10]
    if cands:
        best = max(cands, key=gain_above)
        print(f"  SHIP cal_mode='{best}': no backfire (worst {worst(best):+.2f}) and preserves gain above the floor "
              f"({gain_above(best):+.2f}), while legacy 'full' backfires to {worst('full'):+.2f}. The signal-"
              f"reliability law, enforced in the substrate.", flush=True)
    else:
        print(f"  KEEP 'full': neither safe mode preserved enough gain (boost {gain_above('boost'):+.2f}, gated "
              f"{gain_above('gated'):+.2f}). Report honestly; do not change the default.", flush=True)
    json.dump({"M": M, "T": T, "seeds": N_SEEDS, "res": res,
               "worst": {m: worst(m) for m in ("full", "boost", "gated")},
               "gain_above": {m: gain_above(m) for m in ("full", "boost", "gated")}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ramr_breakeven_result.json"), "w"))
    print("DONE", flush=True)
