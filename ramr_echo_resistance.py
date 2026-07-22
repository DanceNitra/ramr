"""RAMR metric -- ECHO-RESISTANCE: after a fact is corrected, does a RE-STATEMENT of the old value resurrect it?

FORGET-PRECISION (ramr_forget_precision.py) tests whether a correction sticks: assert F1, correct to F2, does
recall return F2? This metric tests the ADVERSARIAL/BENIGN sequel that vendors don't report: after the correction,
the OLD value is stated AGAIN -- a benign restatement (a user repeating a preference they forgot they changed) or
an attacker re-injecting the stale value. On a store whose supersession is last-writer-wins by ingest/validity
recency, the echo is the NEWEST assertion of the old value, so it WINS and the corrected fact is resurrected. This
is the "echo attack"; STALE (arXiv:2605.06527) and LongMemEval measure a single correction but run no re-injection,
so the post-correction restatement rate is unmeasured.

We test two regimes of echo, both keeping the OLD VALUE (a value-preserving restatement, verbatim or reworded):
  VERBATIM  : the exact original old-value sentence is re-stated.
  REWORDED  : the old value is re-asserted in different words (same value token).
Arms (inspeximus keyed supersession, object = the value token):
  NO-GUARD  : echo_guard off -- keyed supersession is validity-recency, so the later echo supersedes the
              correction and the STALE value becomes current again.
  GUARD     : echo_guard on  -- a superseded-object ledger refuses to let an already-retired value be
              resurrected by a mere restatement (value-preserving); a genuine reversal needs reaffirm=True.
To isolate supersession from ranking, the STALE F1 is given HIGHER value than the current F2, so relevance x value
returns the stale fact UNLESS supersession + the guard hold. Cloud-free, lexical (no embedder). ASCII prints.

ECHO-RESISTANCE = fraction of topics where recall's top-1 is the CURRENT value (F2) AFTER the echo arrives.
Pre-registered falsifier: if GUARD does not raise echo-resistance over NO-GUARD, the guard is decorative.
Prediction: NO-GUARD -> ~0 (the echo resurrects the stale value); GUARD -> ~1 (the restatement is blocked),
and the guard must NOT hurt FORGET-PRECISION (the correction itself still sticks when no echo arrives).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus

M_TOPICS = int(os.getenv("ER_M", "30"))
ENTS = ["payment api", "auth service", "search index", "billing job", "cache layer", "upload queue",
        "report engine", "email worker", "session store", "rate limiter", "image cdn", "audit log",
        "webhook relay", "config loader", "metrics sink", "backup task", "login flow", "export tool",
        "notify bus", "schema migrator", "token vault", "flag svc", "data lake", "shard router",
        "pdf renderer", "geo lookup", "fraud check", "recommend svc", "chat gateway", "trace collector"]
OLD = ["frankfurt", "oregon", "dublin", "tokyo", "virginia", "sydney", "london", "mumbai", "sao paulo",
       "toronto", "paris", "seoul", "milan", "osaka", "cairo", "lima", "accra", "oslo", "riga", "sofia",
       "hanoi", "quito", "amman", "tunis", "davao", "cebu", "kigali", "napoli", "bern", "utah"]
NEW = ["ohio", "belgium", "norway", "kenya", "chile", "ghana", "peru", "egypt", "latvia", "vietnam",
       "jordan", "rwanda", "bern", "cebu", "utah", "oslo", "riga", "sofia", "hanoi", "quito", "amman",
       "tunis", "davao", "napoli", "seoul", "milan", "osaka", "cairo", "lima", "accra"]


def build(store, ent, old_v, new_v):
    key = f"{ent}::region"
    # F1 (old) gets HIGHER value than F2 (new): relevance x value would return the stale one unless supersession holds
    store.remember(f"{ent} region is {old_v}", key=key, object=old_v, value=5.0)
    store.remember(f"correction: {ent} region is {new_v}", key=key, object=new_v, value=1.0)


def echo(store, ent, old_v, kind):
    key = f"{ent}::region"
    text = (f"{ent} region is {old_v}" if kind == "verbatim"
            else f"just to confirm, the region for {ent} stays {old_v}")
    store.remember(text, key=key, object=old_v, value=5.0)


def top1_is_current(store, ent, new_v):
    hits = store.recall(f"{ent} region", k=1, mode="lexical")
    return bool(hits) and new_v in hits[0]["text"].lower()


def run(guard, echo_kind):
    def fresh():
        import tempfile
        fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(p)
        m = Inspeximus(path=p); m.echo_guard = guard; return m
    persist_ok, echo_ok = 0, 0
    for i in range(M_TOPICS):
        ent, ov, nv = ENTS[i % len(ENTS)], OLD[i % len(OLD)], NEW[i % len(NEW)]
        if ov == nv:
            continue
        m = fresh(); build(m, ent, ov, nv)
        if top1_is_current(m, ent, nv):
            persist_ok += 1
        echo(m, ent, ov, echo_kind)
        if top1_is_current(m, ent, nv):
            echo_ok += 1
    n = sum(1 for i in range(M_TOPICS) if OLD[i % len(OLD)] != NEW[i % len(NEW)])
    return persist_ok / n, echo_ok / n, n


def main():
    print("=== RAMR ECHO-RESISTANCE ===")
    print(f"(m={M_TOPICS} topics; F1 old value given HIGHER value than F2 correction; lexical, cloud-free)\n")
    out = {}
    for kind in ("verbatim", "reworded"):
        print(f"-- echo kind: {kind} --")
        pn, en, n = run(False, kind)
        pg, eg, _ = run(True, kind)
        print(f"  NO-GUARD  forget-precision(no echo)={pn:.2f}   echo-resistance={en:.2f}")
        print(f"  GUARD     forget-precision(no echo)={pg:.2f}   echo-resistance={eg:.2f}")
        out[kind] = {"n": n,
                     "no_guard": {"forget_precision": round(pn, 3), "echo_resistance": round(en, 3)},
                     "guard": {"forget_precision": round(pg, 3), "echo_resistance": round(eg, 3)}}
    # verdict
    v = out["reworded"]
    lift = v["guard"]["echo_resistance"] - v["no_guard"]["echo_resistance"]
    keeps_fp = v["guard"]["forget_precision"] >= v["no_guard"]["forget_precision"] - 1e-9
    verdict = ("PASS -- echo_guard raises echo-resistance without hurting forget-precision"
               if lift > 0.2 and keeps_fp else
               "FAIL -- guard did not lift echo-resistance (decorative) or hurt forget-precision")
    print(f"\nVERDICT: {verdict}")
    print(f"  reworded-echo resistance lift (guard - no-guard) = {lift:+.2f}; forget-precision preserved = {keeps_fp}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "echo_resistance_result.json"), "w"), indent=2)
    print("-> echo_resistance_result.json")


if __name__ == "__main__":
    main()
