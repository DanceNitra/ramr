"""RAMR reliability-layer reference for the RAMR<->LS interop (anthropics/claude-code#34556).

RAMR owns retrieval measurement; LS owns the deterministic continuation verdict. This module is the small retrieval
harness safal207/LS pins against: it emits the `ramr-ls-evidence-v0.1` envelope from a mnemo store as a THIN
PROJECTION of fields the engine already carries —
    valid_from / invalidated_at  <- bi-temporal validity
    provenance                   <- source-span origin
    reliability_signal           <- per-record Beta(good,bad)
    budget / recency_weight      <- recall budget + recency
— and scores `recovered_side_effect` on the canonical fixture (fixtures/ramr_ls/duplicate_successful_outcome.json).

Boundary invariant (frozen with the fixture): *a retrieval miss is a reliability failure, NOT execution permission*
— so LS must REJECT a duplicate completed side effect in BOTH the recovered and not-recovered cases; RAMR only
MEASURES whether the completion record was recovered. CLOUD-FREE (lexical recall; no embedder/LLM).
"""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnemo.mnemo import Mnemo

BASE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(BASE, "fixtures", "ramr_ls", "duplicate_successful_outcome.json")

def _digest(s): return "sha256:" + hashlib.sha256(s.encode()).hexdigest()

def emit_envelope(store, query_context, query, budget=8, as_of=None):
    """RAMR projection: recall completion records and map each to a recovered_evidence entry (v0.1)."""
    hits = store.recall(query, k=budget, mode="lexical", as_of=as_of)
    raw = {r["id"]: r for r in store.items}
    ev, rel = [], []
    for h in hits:
        r = raw[h["id"]]
        if (r.get("meta") or {}).get("evidence_type") != "completion_record":
            continue
        src = r.get("source") or {}
        rel.append(h.get("reliability", 0.5))
        ev.append({
            "evidence_id": r["id"], "evidence_type": "completion_record",
            "payload_digest": _digest(r["text"]),
            "scope": (r.get("meta") or {}).get("scope", {}),
            "bindings": (r.get("meta") or {}).get("bindings", {}),
            "valid_from": r.get("valid_from"), "invalidated_at": r.get("invalidated_at"),
            "retrieved_at": None,
            "provenance": {"record_source": src.get("doc"), "record_digest": _digest(str(src))},
        })
    return {"envelope_version": "ramr-ls-evidence-v0.1",
            "query_context": query_context,
            "recovered_evidence": ev,
            "retrieval": {"budget": budget, "recency_weight": 0.25,
                          "reliability_signal": round(max(rel), 2) if rel else 0.0}}

def recovered_side_effect(envelope, target_side_effect_key, query_context):
    """RAMR's scored quantity: was a completion_record for the target side effect, with matching bindings, recovered?"""
    for e in envelope["recovered_evidence"]:
        b = e["bindings"]
        if (b.get("side_effect_key") == target_side_effect_key
                and b.get("continuation_id") == query_context.get("continuation_id")
                and b.get("intent_digest") == query_context.get("intent_digest")):
            return True
    return False

def _store_for_case(fx, recovered):
    """Build a mnemo store that reproduces a fixture case: the completion record is recoverable (recovered) or not."""
    s = Mnemo(path=None, embed=None); s.semantic_threshold = 10 ** 9
    led = fx["authoritative_state"]["completion_ledger"][0]
    if recovered:
        mid = s.remember(
            f"completion record side effect {led['side_effect_key']} continuation {led['continuation_id']} done",
            value=3.0, mtype="procedural", source={"doc": "agent-checkpoint", "span": [0, 64]},
            meta={"evidence_type": "completion_record", "scope": {"workspace_id": led.get("workspace_id", "ws-123")},
                  "bindings": {k: led[k] for k in ("continuation_id", "intent_digest", "target_state_digest",
                                                   "approval_id", "side_effect_key")}})
        s.credit([mid], "good")
    else:
        for i in range(4):   # only unrelated records -> the needed completion is NOT recoverable
            s.remember(f"completion record side effect noise{i} continuation other done", value=3.0,
                       mtype="procedural", meta={"evidence_type": "completion_record",
                       "bindings": {"continuation_id": f"other{i}", "side_effect_key": f"noise{i}"}})
    return s

if __name__ == "__main__":
    txt = open(FIXTURE).read()
    fx = json.loads(txt)
    led = fx["authoritative_state"]["completion_ledger"][0]
    qc = fx["query_context"]
    query = f"completion record side effect {led['side_effect_key']} continuation {led['continuation_id']}"
    print("RAMR<->LS interop — canonical fixture:", fx["fixture_id"], "| envelope:", fx["envelope_version"])
    print("content digest:", _digest(txt))
    ok = True
    for case in fx["cases"]:
        recovered = case["case"] == "completion_recovered"
        env = emit_envelope(_store_for_case(fx, recovered), qc, query, budget=8)
        got = recovered_side_effect(env, led["side_effect_key"], qc)
        exp = case["expected"]["ramr_recovered_side_effect"]
        ok &= (got == exp)
        print(f"  {case['case']:>26}: RAMR recovered_side_effect={got} (expected {exp}) "
              f"reliability_signal={env['retrieval']['reliability_signal']}  | LS verdict (per fixture): {case['expected']['ls_verdict']}")
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} — RAMR emits ramr-ls-evidence-v0.1 from native mnemo fields and "
          f"scores recovered_side_effect to match the fixture. Boundary invariant: {fx['_meta']['boundary_invariant']}")
