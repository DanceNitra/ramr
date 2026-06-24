"""Conformance runner for the canonical RAMR<->LS evidence fixtures (fixtures/ramr_ls/, envelope v0.1).

For each fixture it scores the RAMR-side measured quantity per case (purely from the frozen recovered_evidence +
query_context/authoritative_state — no model needed) and checks it against `expected`. LS consumes the SAME frozen
files and checks the deterministic continuation verdict on its side. RAMR's job is reliability MEASUREMENT; LS's is
the verdict — and the four fixtures span the four verdicts (REJECT / REJECT / ABSTAIN / REVALIDATE).

`ramr_ls_evidence.py` shows RAMR *produces* the envelope from a live memory store (bi-temporal valid_from/
invalidated_at, provenance, Beta reliability, recall budget); this runner is the frozen-fixture conformance check.
CLOUD-FREE, deterministic, zero deps.
"""
import os, json, hashlib

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "fixtures", "ramr_ls")

def _evid(case): return case.get("recovered_evidence", [])

def score_duplicate(fx, case):
    led = fx["authoritative_state"]["completion_ledger"][0]; qc = fx["query_context"]
    key = led["side_effect_key"]
    return any(e.get("evidence_type") == "completion_record"
               and e["bindings"].get("side_effect_key") == key
               and e["bindings"].get("continuation_id") == qc["continuation_id"]
               and e["bindings"].get("intent_digest") == qc["intent_digest"]
               for e in _evid(case))

def score_superseded(fx, case):
    # RAMR recovered the CURRENT (revoked) approval state iff a recovered approval_record carries invalidated_at
    return any(e.get("evidence_type") == "approval_record" and e.get("invalidated_at") is not None for e in _evid(case))

def score_incomplete(fx, case):
    req = set(fx["authoritative_state"]["required_chain"])
    got = {e["bindings"].get("chain_role") for e in _evid(case)}
    return req.issubset(got)

def score_drift(fx, case):
    cur = fx["query_context"]["target_state_digest"]; ev = _evid(case)
    return bool(ev) and all(e["bindings"].get("target_state_digest") == cur for e in ev)

SCORERS = {
    "duplicate_successful_outcome": ("ramr_recovered_side_effect", score_duplicate),
    "superseded_approval": ("ramr_recovered_current_approval", score_superseded),
    "incomplete_dependency_chain": ("ramr_full_chain_recovered", score_incomplete),
    "target_state_drift": ("ramr_target_current", score_drift),
}

if __name__ == "__main__":
    print("RAMR<->LS fixture conformance (RAMR-side measured quantity vs expected; LS owns the verdict):\n")
    allok = True
    for fid, (key, scorer) in SCORERS.items():
        path = os.path.join(DIR, fid + ".json")
        txt = open(path).read(); fx = json.loads(txt)
        dg = "sha256:" + hashlib.sha256(txt.encode()).hexdigest()
        print(f"{fid}  [{dg[:19]}…]")
        for c in fx["cases"]:
            got = scorer(fx, c); exp = c["expected"][key]; ok = got == exp; allok &= ok
            print(f"   {c['case']:>24}: RAMR {key}={str(got):>5} (exp {str(exp):>5}) {'OK' if ok else 'MISMATCH'}"
                  f"  | LS verdict: {c['expected']['ls_verdict']}")
    print(f"\nVERDICTS COVERED: {sorted({c['expected']['ls_verdict'] for fid,(k,s) in SCORERS.items() for c in json.load(open(os.path.join(DIR,fid+'.json')))['cases']})}")
    print(f"CONFORMANCE: {'PASS — all RAMR-side measurements match the frozen fixtures' if allok else 'FAIL'}")
