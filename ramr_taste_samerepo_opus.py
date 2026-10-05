"""Does inspeximus RANKING help, or does any lesson from the same project help as much?

The pre-draft skeptic of 2026-09-28 found that 86 % of the lessons inspeximus recalls come from the fork's own
repository, while the v3c random-lesson control drew from every repository. So memory_sem - control mixed two
effects: "a lesson from this project" and "the right lesson from this project". This run separates them.

  none              no lessons
  memory_sem        top-3 by nomic-embed-text similarity (inspeximus), leave-task-out
  random_repo_match for each recalled lesson, a random other lesson from the SAME repository with the SAME
                    surface-prior signature (longer option, isolate/stub cue). Same repo mix, same cues, same
                    count and form as memory_sem; only the ranking is taken away.

memory_sem - random_repo_match is the value of ranking. Decider: Claude Opus 5.5 through `claude -p` on the
subscription, effort medium (owner, 2026-09-28), all three arms re-run at that effort so they compare.

    python ramr_taste_samerepo_opus.py --n 160 --build-only
    python ramr_taste_samerepo_opus.py --n 160            # resumable, cached by prompt hash
"""
import argparse, concurrent.futures as cf, hashlib, json, os, random, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_taste_memory as T
import ramr_taste_poison_opus as O

HERE = T.HERE
MODEL = "claude-opus-5-5"
EFFORT = "medium"
CACHE = os.path.join(HERE, "taste_memory_calls_engineering_opus55_medium.jsonl")
OUT = os.path.join(HERE, "taste_samerepo_opus55_medium_result.json")
SYSTEM = O.SYSTEM
ARMS = ("none", "memory_sem", "random_repo_match")
LOCK = os.path.join(os.path.expanduser("~"), ".claude", "compute", "taste_samerepo_opus.lock")


def build_jobs(pool, items):
    O.wait_for_gpu()
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    open(LOCK, "w").write("taste same-repo control, nomic embeddings only (a few minutes)\n")
    try:
        ms, own = T.build_memory_sem(pool)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)
    rnd = random.Random(T.SEED + 7)
    stats = {"recalled": 0, "same_repo": 0, "matched_exact": 0, "matched_repo_only": 0, "unmatched": 0}
    lessons = {c: {} for c in ARMS}
    for q in items:
        got = T.recalled(ms, own, q, items=True)
        lessons["none"][q["id"]] = []
        lessons["memory_sem"][q["id"]] = [T.lesson(p) for p in got]
        ctrl, used = [], {p["id"] for p in got}
        for r in got:
            stats["recalled"] += 1
            stats["same_repo"] += T.repo_of(r) == T.repo_of(q)
            base = [p for p in pool if p["task_id"] != q["task_id"] and p["id"] not in used
                    and T.repo_of(p) == T.repo_of(r)]
            exact = [p for p in base if T.lesson_sig(p) == T.lesson_sig(r)]
            pick = rnd.choice(exact) if exact else (rnd.choice(base) if base else None)
            key = "matched_exact" if exact else ("matched_repo_only" if base else "unmatched")
            stats[key] += 1
            if pick is not None:
                used.add(pick["id"])
                ctrl.append(T.lesson(pick))
        lessons["random_repo_match"][q["id"]] = ctrl
    same = sum(a == b for q in items for a, b in zip(lessons["memory_sem"][q["id"]], lessons["random_repo_match"][q["id"]]))
    assert same == 0, f"{same} control lessons equal the recalled ones"
    jobs = []
    for q in items:
        for c in ARMS:
            for order in ("published", "reversed"):
                user, correct = T.prompt(q, order, lessons[c][q["id"]])
                jobs.append({"id": q["id"], "cond": c, "order": order, "user": user, "correct": correct,
                             "psha": hashlib.sha256(user.encode()).hexdigest()[:16]})
    return jobs, stats


def call(user):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    d = {"result": "no attempt"}
    for attempt in range(3):
        exe = shutil.which("claude")
        if not exe:
            d = {"is_error": True, "result": "claude not on PATH"}
            time.sleep(60)
            continue
        cmd = [exe, "-p", "--model", MODEL, "--effort", EFFORT, "--system-prompt", SYSTEM, "--tools", "",
               "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands", "--output-format", "json"]
        try:
            r = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", env=env, timeout=900)
            d = json.loads(r.stdout)
        except ValueError:
            d = {"is_error": True, "result": (r.stdout or r.stderr)[:300]}
        except (OSError, subprocess.TimeoutExpired) as e:
            d = {"is_error": True, "result": repr(e)[:300]}
        if not d.get("is_error"):
            u = d.get("usage") or {}
            return (d.get("result") or "", u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                    + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0), d.get("total_cost_usd", 0))
        time.sleep(20 * (attempt + 1))
    return "ERROR " + str(d.get("result"))[:200], 0, 0, 0


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
    print(f"[{T.el()}] control build: {stats}", flush=True)
    print(f"[{T.el()}] {len(jobs)} jobs, {len(todo)} to call, effort {EFFORT}", flush=True)
    if a.build_only:
        return
    stop_at = time.time() + a.stop_after_min * 60 if a.stop_after_min else None

    def run(j):
        if stop_at and time.time() > stop_at:
            return None
        text, tin, tout, usd = call(j["user"])
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
    used = [done[p] for p in {j["psha"] for j in jobs} if p in done]
    out = {"metric": "taste_ranking_vs_same_repo_random", "model": MODEL, "effort": EFFORT, "n_forks": len(ids),
           "control_build": stats, "both_correct_rate": {c: round(sum(v) / len(v), 4) for c, v in per.items()},
           "memory_minus_none": T.boot_diff(per["memory_sem"], per["none"]),
           "memory_minus_repo_random": T.boot_diff(per["memory_sem"], per["random_repo_match"]),
           "repo_random_minus_none": T.boot_diff(per["random_repo_match"], per["none"]),
           "heuristic_rate": round(sum(T.heuristic_pick(q) == q["answer_index"] for q in items if q["id"] in ids)
                                   / len(ids), 4),
           "errors": sum(1 for r in used if str(r.get("tail", "")).startswith("ERROR")),
           "unparsed": sum(1 for r in used if r.get("answer") not in ("A", "B")),
           "usd_equivalent": round(sum(r.get("usd_equiv", 0) for r in used), 2),
           "raw": {"ids": ids, **per}, "seed": T.SEED}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
