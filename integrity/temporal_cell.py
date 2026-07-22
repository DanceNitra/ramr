#!/usr/bin/env python3
"""TEMPORAL / bitemporal integrity cell (cross-system, deterministic, no LLM judge).

Ground truth uses unique tokens, so scoring is EXACT -- no model in the loop. Four cells:
  A) reversed-ingest now-accuracy -- ingest the newest-VALID fact FIRST and the oldest-valid LAST; a store that
     resolves by ingest-order (or pure similarity) returns the stale last-written value, a valid-time store returns
     the current one.
  B/C) as_of(valid-time) point queries -- "what was the value valid at time T?" Requires a point-in-time channel.
  D) transaction-time back-fill -- "what did the store BELIEVE at write-time tw, about valid-time T?" Requires a
     second clock (bitemporal). A later correction must not leak into the earlier belief.

HONEST SCOPE: this is a PARITY-with-leaders cell, not an inspeximus-uniqueness claim. Bi-temporal modelling is the
documented design of the graph-memory leaders (Zep, Graphiti) too; this cell shows inspeximus MATCHES that design and
that a plain vector store (mem0's default) lacks a valid-time channel. Zep/Graphiti are not run here (they need a
live Neo4j + an LLM extraction pipeline); their row is marked accordingly, not scored, so we never claim a win we
did not measure. Run it yourself; add your system.

Usage:  python temporal_cell.py            # writes results/temporal.json
"""
import tempfile, os, time, json, sys

T1, T2, T3 = 1000.0, 2000.0, 3000.0
VALS = {T1: "ROLE-ENGINEER", T2: "ROLE-MANAGER", T3: "ROLE-DIRECTOR"}  # current (latest valid_from) = DIRECTOR


def run_inspeximus():
    from inspeximus import Inspeximus
    v = lambda r: r.get("text") if r else None
    d = tempfile.mkdtemp(); m = Inspeximus(path=os.path.join(d, "s.json"))
    for vt in (T3, T2, T1):                       # reversed valid-time ingest order
        m.remember(VALS[vt], key="role", source={"doc": "hr"}, valid_from=vt)
    m._save(force=True)
    A = v(m.as_of("role", when=9999.0)) == VALS[T3]
    B = v(m.as_of("role", when=1500.0)) == VALS[T1]
    C = v(m.as_of("role", when=2500.0)) == VALS[T2]
    # D: forward-order store, transaction-time back-fill
    d2 = tempfile.mkdtemp(); m2 = Inspeximus(path=os.path.join(d2, "s.json"))
    m2.remember(VALS[T1], key="role", source={"doc": "hr"}, valid_from=T1)
    tw1 = max(r["ts"] for r in m2.items if r.get("key") == "role"); time.sleep(0.01)
    m2.remember(VALS[T2], key="role", source={"doc": "hr"}, valid_from=T2)
    tw2 = max(r["ts"] for r in m2.items if r.get("key") == "role")
    D = (v(m2.as_of("role", when=2500.0, as_recorded=tw1)) == VALS[T1] and
         v(m2.as_of("role", when=2500.0, as_recorded=tw2)) == VALS[T2])
    return {"now_accuracy_reversed_ingest": bool(A), "as_of_valid_time": bool(B and C),
            "transaction_time_backfill": bool(D), "score": f"{sum([A,B,C,D])}/4"}


def run_mem0():
    os.environ.setdefault("OPENAI_API_KEY", "sk-none")
    from mem0 import Memory
    dm = tempfile.mkdtemp()
    mm = Memory.from_config({"embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(dm, "qd"),
            "embedding_model_dims": 384, "on_disk": True}},
        "history_db_path": os.path.join(dm, "h.db")})
    for vt in (T3, T2, T1):
        mm.add(f"my role is {VALS[vt]}", user_id="u", infer=False)
    res = mm.search("what is my current role", filters={"user_id": "u"}, limit=3)
    hits = (res.get("results") if isinstance(res, dict) else res) or []
    top = (hits[0].get("memory") if hits else "") or ""
    return {"now_accuracy_reversed_ingest": VALS[T3] in top,
            "as_of_valid_time": "N/A (no valid-time API)",
            "transaction_time_backfill": "N/A (no valid-time API)",
            "score": "no valid-time channel", "top1": top}


ADAPTERS = {"inspeximus": run_inspeximus, "mem0": run_mem0}
# documented bitemporal, not run here (need live Neo4j + LLM pipeline) -- parity, not measured
NOT_RUN = {"zep": "documented bi-temporal (valid + transaction time, point-in-time) -- expected PARITY; not run",
           "graphiti": "documented bi-temporal (valid_at/invalid_at + created_at) -- expected PARITY; not run"}


def main(argv):
    want = [a.lower() for a in argv] or list(ADAPTERS)
    out = {"cell": "temporal", "deterministic": True, "judge": "none (unique-token ground truth)",
           "honest_scope": "PARITY-with-leaders cell, not an inspeximus-uniqueness claim; graph-memory leaders "
                           "(Zep, Graphiti) are also bi-temporal and are not run here.",
           "systems": {}, "not_run": NOT_RUN}
    for name in want:
        fn = ADAPTERS.get(name)
        if not fn:
            out["systems"][name] = {"error": "no adapter"}; continue
        try:
            out["systems"][name] = fn()
        except ImportError:
            out["systems"][name] = {"error": "not installed"}
        except Exception as ex:
            out["systems"][name] = {"error": repr(ex)[:120]}
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    path = os.path.join(here, "results", "temporal.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["systems"], indent=2))
    print("\nnot run (documented bi-temporal; parity):")
    for k, val in NOT_RUN.items():
        print(f"  {k}: {val}")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
