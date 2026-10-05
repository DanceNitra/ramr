"""Is the gain from "random lessons of the same repository" repository-specific, or generic?

Control suggested by u/nitish-kmr on r/Rag (2026-10-01): random lessons from a DIFFERENT repository, everything else
held. In the 2026-09-28 run (ramr_taste_samerepo_opus.py), random same-repository lessons scored 0.750 against 0.594
with no lessons. If cross-repository random lessons still buy most of that gain, the effect is generic. If it
collapses, the gain is repository-specific information.

  none               no lessons                                        (cached from the 2026-09-28 run)
  memory_sem         top-3 by nomic-embed-text similarity, leave-task-out (cached)
  random_repo_match  random other lesson, SAME repository, same surface signature (cached)
  random_cross_repo  for each recalled lesson, a random lesson from a DIFFERENT repository with the same
                     surface-prior signature (longer option, isolate/stub cue). Same count and form, same cues.

Prompts of the first three arms are identical to the earlier run, so the shared call cache supplies their answers
and only the new arm is called (160 forks, two option orders, 320 calls). Decider: Claude Opus 5.5 through
`claude -p`, effort medium, as before.

    python ramr_taste_crossrepo_opus.py --n 160 --build-only
    python ramr_taste_crossrepo_opus.py --n 160            # resumable, cached by prompt hash
"""
import argparse, concurrent.futures as cf, hashlib, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_taste_memory as T
import ramr_taste_poison_opus as O
import ramr_taste_samerepo_opus as S

HERE = T.HERE
MODEL = S.MODEL
EFFORT = S.EFFORT
CACHE = S.CACHE
OUT = os.path.join(HERE, "taste_crossrepo_opus55_medium_result.json")
ARMS = ("none", "memory_sem", "random_repo_match", "random_cross_repo")
LOCK = os.path.join(os.path.expanduser("~"), ".claude", "compute", "taste_crossrepo_opus.lock")


def build_jobs(pool, items):
    O.wait_for_gpu()
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    open(LOCK, "w").write("taste cross-repo control, nomic embeddings only (a few minutes)\n")
    try:
        ms, own = T.build_memory_sem(pool)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)
    rnd_same = random.Random(T.SEED + 7)      # the same stream as the earlier run, so its arm is rebuilt byte for byte
    rnd_cross = random.Random(T.SEED + 11)
    stats = {"recalled": 0, "same_repo": 0, "cross_matched_exact": 0, "cross_any_signature": 0, "cross_unmatched": 0}
    lessons = {c: {} for c in ARMS}
    for q in items:
        got = T.recalled(ms, own, q, items=True)
        lessons["none"][q["id"]] = []
        lessons["memory_sem"][q["id"]] = [T.lesson(p) for p in got]
        same_ctrl, used = [], {p["id"] for p in got}
        for r in got:
            base = [p for p in pool if p["task_id"] != q["task_id"] and p["id"] not in used
                    and T.repo_of(p) == T.repo_of(r)]
            exact = [p for p in base if T.lesson_sig(p) == T.lesson_sig(r)]
            pick = rnd_same.choice(exact) if exact else (rnd_same.choice(base) if base else None)
            if pick is not None:
                used.add(pick["id"])
                same_ctrl.append(T.lesson(pick))
        lessons["random_repo_match"][q["id"]] = same_ctrl
        cross_ctrl, used2 = [], {p["id"] for p in got}
        for r in got:
            stats["recalled"] += 1
            stats["same_repo"] += T.repo_of(r) == T.repo_of(q)
            other = [p for p in pool if p["task_id"] != q["task_id"] and p["id"] not in used2
                     and T.repo_of(p) != T.repo_of(q)]
            exact = [p for p in other if T.lesson_sig(p) == T.lesson_sig(r)]
            pick = rnd_cross.choice(exact) if exact else (rnd_cross.choice(other) if other else None)
            stats["cross_matched_exact" if exact else ("cross_any_signature" if other else "cross_unmatched")] += 1
            if pick is not None:
                used2.add(pick["id"])
                cross_ctrl.append(T.lesson(pick))
        lessons["random_cross_repo"][q["id"]] = cross_ctrl
    leaked = 0
    by_lesson = {}
    for p in pool:
        by_lesson.setdefault(T.lesson(p), set()).add(T.repo_of(p))
    for q in items:
        for ls in lessons["random_cross_repo"][q["id"]]:
            if T.repo_of(q) in by_lesson.get(ls, set()) and len(by_lesson[ls]) == 1:
                leaked += 1
    stats["cross_lessons_from_the_forks_own_repo"] = leaked
    assert leaked == 0, f"{leaked} cross-repository control lessons come from the fork's own repository"
    jobs = []
    for q in items:
        for c in ARMS:
            for order in ("published", "reversed"):
                user, correct = T.prompt(q, order, lessons[c][q["id"]])
                jobs.append({"id": q["id"], "cond": c, "order": order, "user": user, "correct": correct,
                             "psha": hashlib.sha256(user.encode()).hexdigest()[:16]})
    return jobs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--stop-after-min", type=float, default=0)
    a = ap.parse_args()
    pool = T.load_questions("engineering", 0)
    items = pool[:a.n]
    jobs, stats = build_jobs(pool, items)
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            if not str(r.get("tail", "")).startswith("ERROR"):
                done[r["psha"]] = r
    todo = list({j["psha"]: j for j in jobs if j["psha"] not in done}.values())
    cached_old = sum(1 for j in jobs if j["cond"] != "random_cross_repo" and j["psha"] in done)
    old_total = sum(1 for j in jobs if j["cond"] != "random_cross_repo")
    print(f"[{T.el()}] control build: {stats}", flush=True)
    print(f"[{T.el()}] {len(jobs)} jobs, {len(todo)} to call, effort {EFFORT}; earlier arms found in the cache: "
          f"{cached_old} of {old_total}", flush=True)
    if a.build_only:
        return
    assert cached_old == old_total, "the earlier arms did not rebuild identically, so the comparison would not hold"
    stop_at = time.time() + a.stop_after_min * 60 if a.stop_after_min else None

    def run(j):
        if stop_at and time.time() > stop_at:
            return None
        text, tin, tout, usd = S.call(j["user"])
        return {"psha": j["psha"], "model": MODEL, "effort": EFFORT, "answer": T.parse(text), "tin": tin,
                "tout": tout, "usd_equiv": usd, "tail": text[-300:]}

    with cf.ThreadPoolExecutor(a.workers) as px, open(CACHE, "a", encoding="utf-8") as f:
        for n, r in enumerate(px.map(run, todo), 1):
            if r is None:
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["psha"]] = r
            if n % 10 == 0 or n == len(todo):
                usd = sum(x.get("usd_equiv", 0) for x in done.values())
                print(f"[{T.el()}] {n}/{len(todo)} calls, {usd:.2f} USD-equivalent in cache", flush=True)
    missing = {j["id"] for j in jobs if j["psha"] not in done}
    ids = [q["id"] for q in items if q["id"] not in missing]
    recs = [{"id": j["id"], "cond": j["cond"], "order": j["order"],
             "ok": int(done[j["psha"]].get("answer") == j["correct"])} for j in jobs if j["id"] not in missing]
    per = {c: T.both_correct(recs, c, ids) for c in ARMS}
    new_used = [done[j["psha"]] for j in jobs if j["cond"] == "random_cross_repo" and j["psha"] in done]
    out = {"metric": "taste_cross_repo_random_control", "model": MODEL, "effort": EFFORT, "n_forks": len(ids),
           "control_build": stats, "both_correct_rate": {c: round(sum(v) / len(v), 4) for c, v in per.items()},
           "cross_repo_minus_none": T.boot_diff(per["random_cross_repo"], per["none"]),
           "same_repo_minus_cross_repo": T.boot_diff(per["random_repo_match"], per["random_cross_repo"]),
           "memory_minus_cross_repo": T.boot_diff(per["memory_sem"], per["random_cross_repo"]),
           "repo_random_minus_none": T.boot_diff(per["random_repo_match"], per["none"]),
           "errors": sum(1 for r in new_used if str(r.get("tail", "")).startswith("ERROR")),
           "unparsed": sum(1 for r in new_used if r.get("answer") not in ("A", "B")),
           "usd_equivalent_new_arm": round(sum(r.get("usd_equiv", 0) for r in new_used), 2),
           "raw": {"ids": ids, **per}, "seed": T.SEED}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
