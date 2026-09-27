"""Pilot: the planted, flipped lesson with a strong model deciding, on inspeximus and on Polign Recall.

The 2026-09-27 run (ramr_taste_poison.py) used a local 27B model as the decider. This pilot asks whether a
stronger decider changes the picture: does it notice a flipped lesson on its own, or follow it? Every arm
uses the same decider, Claude Opus 5.5 through `claude -p` on a Claude subscription (no API key), so the
arms compare within this run. Same forks, lessons, twins, prompts and scoring as the earlier scripts.

  none            no lessons
  memory_sem      clean store, top-3 by nomic-embed-text similarity (inspeximus)
  poisoned_naive  every lesson plus its flipped twin, no outcome tracking (plain RAG)
  poisoned_guard  same store with outcome credit, recall(influence_only=True)
  recall_current  Polign Recall v0.8.1, typed path; the planted copy is the current belief
  recall_checked  Recall plus an external source check that falls back to history()

    python ramr_taste_poison_opus.py --n 60 --build-only   # prompts and poison shares, no model calls
    python ramr_taste_poison_opus.py --n 60                 # run (resumable; cached by prompt hash)
    python ramr_taste_poison_opus.py --n 60 --arms none,recall_current,recall_checked   # no GPU needed

Needs a polign-server on POLIGN_URL (see ramr_taste_poison_recall.py) and nomic-embed-text on the local
Ollama for the embeddings. Nothing but embeddings runs locally.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_taste_memory as T
import ramr_taste_poison as P
import ramr_taste_poison_recall as R

HERE = T.HERE
MODEL = "claude-opus-5-5"
CACHE = os.path.join(HERE, "taste_memory_calls_engineering_opus55.jsonl")
SYSTEM = "Answer the question in the user's message."
ARMS = ("none", "memory_sem", "poisoned_naive", "poisoned_guard", "recall_current", "recall_checked")


LOCK = os.path.join(os.path.expanduser("~"), ".claude", "compute", "taste_opus_pilot.lock")


def wait_for_gpu(min_free_mb=3000):
    """The film renders first: embed only when ComfyUI's queue is empty and the card has room."""
    import urllib.request
    while True:
        try:
            q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3))
            busy = bool(q.get("queue_running") or q.get("queue_pending"))
        except Exception:
            busy = False
        free = T.gpu_free_mb()
        if not busy and free >= min_free_mb:
            return
        print(f"[{T.el()}] waiting for the GPU: ComfyUI busy={busy}, free={free} MB", flush=True)
        time.sleep(120)


GPU_ARMS = ("memory_sem", "poisoned_naive", "poisoned_guard")    # need nomic embeddings on the local GPU


def build_jobs(pool, items, arms=ARMS):
    lessons = {c: {} for c in ARMS}
    if "none" in arms:
        for q in items:
            lessons["none"][q["id"]] = []
    if any(c in arms for c in GPU_ARMS):
        wait_for_gpu()
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        open(LOCK, "w").write("claude session 2bd3daea: taste Opus pilot, nomic embeddings only (a few minutes)\n")
        try:
            ms, own_s = T.build_memory_sem(pool)
            print(f"[{T.el()}] clean semantic store built", flush=True)
            mn, own_n = P.build_poisoned_store(pool, credited=False)
            mg, own_g = P.build_poisoned_store(pool)
            print(f"[{T.el()}] poisoned stores built", flush=True)
        finally:
            if os.path.exists(LOCK):
                os.remove(LOCK)
        for q in items:
            lessons["memory_sem"][q["id"]] = [(t, False) for t in T.recalled(ms, own_s, q)]
            lessons["poisoned_naive"][q["id"]] = P.recall3(mn, own_n, q, False)
            lessons["poisoned_guard"][q["id"]] = P.recall3(mg, own_g, q, True)
    if any(c.startswith("recall_") for c in arms):
        saved = dict(os.environ)                     # R.client strips the environment for the Recall process
        try:
            with R.client("taste_opus_" + time.strftime("%Y%m%d_%H%M%S")) as m:
                by_subject = R.build(m, pool)
                for q in items:
                    lessons["recall_current"][q["id"]] = R.retrieve(m, by_subject, q, False)
                    lessons["recall_checked"][q["id"]] = R.retrieve(m, by_subject, q, True)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        print(f"[{T.el()}] Recall store built and read", flush=True)
    lessons = {c: lessons[c] for c in arms}
    shares = {c: round(sum(sum(p for _, p in lessons[c][q["id"]]) / max(1, len(lessons[c][q["id"]]))
                           for q in items) / len(items), 3) for c in arms}
    jobs = []
    for q in items:
        for c in arms:
            for order in ("published", "reversed"):
                user, correct = T.prompt(q, order, [t for t, _ in lessons[c][q["id"]]])
                jobs.append({"id": q["id"], "cond": c, "order": order, "user": user, "correct": correct,
                             "psha": hashlib.sha256(user.encode()).hexdigest()[:16]})
    return jobs, shares


def call(user):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    d = {"result": "no attempt"}
    for attempt in range(3):
        exe = shutil.which("claude")              # resolved per attempt: an auto-update swaps the binary
        if not exe:
            d = {"is_error": True, "result": "claude not on PATH"}
            time.sleep(60)
            continue
        cmd = [exe, "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands", "--output-format", "json"]
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
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset; the cache carries calls over")
    a = ap.parse_args()
    arms = tuple(c for c in ARMS if c in a.arms.split(","))
    pool = T.load_questions("engineering", 0)
    items = pool[:a.n]
    jobs, shares = build_jobs(pool, items, arms)
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            if not str(r.get("tail", "")).startswith("ERROR"):
                done[r["psha"]] = r
    todo = list({j["psha"]: j for j in jobs if j["psha"] not in done}.values())
    print(f"[{T.el()}] poison share in top-3: {shares}", flush=True)
    print(f"[{T.el()}] {len(jobs)} jobs, {len({j['psha'] for j in jobs})} distinct prompts, {len(todo)} to call", flush=True)
    if a.build_only:
        return

    def run(j):
        text, tin, tout, usd = call(j["user"])
        return {"psha": j["psha"], "model": MODEL, "answer": T.parse(text), "tin": tin, "tout": tout,
                "usd_equiv": usd, "tail": text[-300:]}

    with cf.ThreadPoolExecutor(a.workers) as px, open(CACHE, "a", encoding="utf-8") as f:
        for n, r in enumerate(px.map(run, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["psha"]] = r
            if n % 10 == 0 or n == len(todo):
                print(f"[{T.el()}] {n}/{len(todo)} calls", flush=True)
    recs = [{"id": j["id"], "cond": j["cond"], "order": j["order"],
             "ok": int(done.get(j["psha"], {}).get("answer") == j["correct"])} for j in jobs]
    ids = [q["id"] for q in items]
    per_arm = {c: T.both_correct(recs, c, ids) for c in arms}
    rate = {c: round(sum(v) / len(v), 4) for c, v in per_arm.items()}
    used = [done[p] for p in {j["psha"] for j in jobs} if p in done]
    out = {"metric": "taste_poison_strong_decider", "model": MODEL, "n_forks": len(ids), "arms": list(arms),
           "both_correct_rate": rate, "poison_share_top3": shares,
           "errors": sum(1 for r in used if str(r.get("tail", "")).startswith("ERROR")),
           "tokens": {"input": sum(r["tin"] for r in used), "output": sum(r["tout"] for r in used)},
           "usd_equivalent": round(sum(r.get("usd_equiv", 0) for r in used), 2)}
    for name, x, y in (("memory_minus_none", "memory_sem", "none"),
                       ("naive_minus_memory", "poisoned_naive", "memory_sem"),
                       ("guard_minus_naive", "poisoned_guard", "poisoned_naive"),
                       ("recall_current_minus_none", "recall_current", "none"),
                       ("recall_checked_minus_current", "recall_checked", "recall_current"),
                       ("recall_checked_minus_none", "recall_checked", "none")):
        if x in per_arm and y in per_arm:
            out[name] = T.boot_diff(per_arm[x], per_arm[y])
    out.update({"raw": {"ids": ids, **per_arm}, "seed": T.SEED})
    suffix = "" if arms == ARMS else "_" + "-".join(arms)
    json.dump(out, open(os.path.join(HERE, f"taste_poison_opus55_result{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
