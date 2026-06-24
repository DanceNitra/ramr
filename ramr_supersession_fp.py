"""RAMR — SUPERSESSION FALSE-POSITIVE (the dual of FORGET-PRECISION).

FORGET-PRECISION asks: after a fact is UPDATED, does recall return the CURRENT value? But a memory layer can score
1.0 there and still be broken in the OTHER direction: wrongly DELETING coexisting facts while trying to forget.
This metric measures that failure. ENUMERATED facts ("step 1 takes 5 min", "step 2 takes 8 min", ...) share a
number-stripped skeleton and differ only in numbers, so a naive value-update detector treats each pair as one fact
being updated and supersedes all but one.

SUPERSESSION-FALSE-POSITIVE = fraction of a coexisting enumerated set wrongly superseded by consolidate().
Pre-registered falsifier: if a store of N distinct enumerated facts keeps all N active after consolidate(), the
detector is safe (FP=0). Any toggle is a false-positive (a coexisting record silently removed from recall).

Control: TRUE value-updates ("config timeout is 5" -> "...is 12") MUST still supersede the stale one, or the fix
has broken FORGET-PRECISION (run ramr_forget_precision.py as the paired regression gate).

CLOUD-FREE: pure in-memory mnemo, lexical only (no embedder, no LLM). Reproducible (fixed inputs).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnemo.mnemo import Mnemo, _value_clash

def active(m):
    return [r["text"] for r in m.items if r["status"] == "active"]

def enumerated_fp(n=6):
    m = Mnemo(path=None, embed=None)
    for i in range(1, n + 1):
        m.remember(f"step {i} takes {3 + 2 * i} min", tags=["proc"], value=2)
    before = len(active(m))
    m.consolidate(keep=50)
    survived = active(m)
    fp = (before - len(survived)) / before
    return {"n": before, "survived": len(survived), "false_positive_rate": fp}

def true_update_intact():
    m = Mnemo(path=None, embed=None)
    m.remember("config alpha timeout is 5", tags=["cfg"], value=2)
    m.remember("config alpha timeout is 12", tags=["cfg"], value=2)
    m.consolidate(keep=50)
    a = active(m)
    return {"active": a, "current_kept": any("12" in t for t in a),
            "stale_dropped": not any(t.endswith("is 5") for t in a)}

if __name__ == "__main__":
    units = [("step 1 takes 5 min", "step 2 takes 8 min", False),       # enumerated -> must NOT clash
             ("config alpha timeout is 5", "config alpha timeout is 12", True),   # update -> must clash
             ("battery alpha at 5 of 10 cells", "battery alpha at 7 of 10 cells", True),
             ("step 1 takes 5 min", "step 5 takes 13 min", False)]      # index==another value -> must NOT clash
    unit_ok = all(_value_clash(a, b) == want for a, b, want in units)
    enum = enumerated_fp(6)
    upd = true_update_intact()
    res = {"unit_detector_ok": unit_ok, "enumerated": enum, "true_update": upd}
    print(json.dumps(res, indent=2))
    verdict = unit_ok and enum["false_positive_rate"] == 0.0 and upd["current_kept"] and upd["stale_dropped"]
    print("\nVERDICT:", "PASS — 0 false-positive supersessions, true updates intact." if verdict
          else "FAIL — supersession over-fires or true-update broke.")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "supersession_fp_result.json"), "w"), indent=1)
