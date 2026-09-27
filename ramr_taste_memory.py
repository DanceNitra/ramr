"""RAMR metric: TASTE LIFT FROM MEMORY -- does recalled experience make an agent choose the better
direction at a long-horizon decision fork?

Taste-Bench (Pan et al., arXiv 2609.25804; data CC-BY-4.0 at huggingface.co/datasets/wenbopan/taste-bench,
code MIT at github.com/wbopan/tastebench) measures "taste": at a fork mined from real engineering and
research trajectories, which of two next steps leads to the better outcome. Its authors report the best
frontier model at 59.7 %. It measures a model alone. RAMR asks the question a memory product has to answer:
does giving the agent recalled lessons from OTHER tasks move that number, and is the move caused by the
memory or merely by extra text in the prompt?

Three conditions, same model, same questions, Taste-Bench's paired-order protocol (published order and
its exact reverse, correct letter recomputed, errors and unparseable replies count as wrong):
  none     -- the fork alone (the Taste-Bench baseline).
  memory   -- plus the top-K lessons recalled by inspeximus (lexical, no embedder) from a store of
              lessons written from OTHER tasks' forks: "choosing X over Y led to the better outcome".
              Leave-task-out: no lesson from the same task_id can be recalled, so no answer leaks.
  control  -- plus K lessons drawn at random from other tasks. Same length and form as memory, so
              memory - control isolates relevance from "any extra text".

Headline: both_correct_rate per condition, and the paired differences memory-none and memory-control with
a paired bootstrap 95 % CI. Raw per-item 0/1 arrays are persisted so verify_numbers.py can recompute.

Honest scope: lessons are written from the dataset's own labels, so this measures the ceiling of
"perfectly written lessons from other tasks", not what an agent's self-written memory achieves. The
dataset is gated on Hugging Face: an account must accept its terms before the parquet downloads.

    python ramr_taste_memory.py --fixture                 # plumbing test, no network, no model
    python ramr_taste_memory.py --estimate --n 40          # token estimate before any model call
    python ramr_taste_memory.py --n 40                     # local qwen3.8:27b, thinking off
"""
import argparse, concurrent.futures as cf, json, os, random, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "tastebench")
HF = "https://huggingface.co/datasets/wenbopan/taste-bench/resolve/main/data/{domain}/test-00000-of-00001.parquet"
OLLAMA = "http://127.0.0.1:11434/api/chat"
SEED = 20260925
K = 3
INVERTED_ON_FIRST = 100
CONFIG = "v3c"  # v3: red-team fixes (answer stored, prior-matched control, cross-repo arms, inverted lessons, all forks)
MAX_PREFIX_CHARS = 12000  # v3c (owner chose variant C): shorter for all arms alike; lowers the no-memory baseline only.
_OLD_PREFIX_NOTE = 20000  # head kept + tail nearest the fork (upstream does the same on overflow; our limit is lower to fit 8k ctx)
HEAD_CHARS = 3000
T0 = time.time()


def el():
    return f"{time.time() - T0:.0f}s"


# ------------------------------------------------------------------ data
def fixture_questions():
    """Schema-valid synthetic forks for the plumbing test. The better step is obvious by construction."""
    good = ["Run the existing test suite before changing anything", "Check the data split for leakage first",
            "Reproduce the bug with a minimal failing test", "Profile before optimizing",
            "Read the error message and the stack trace", "Pin the dependency version that worked"]
    bad = ["Rewrite the module from scratch", "Train a larger model immediately", "Delete the failing test",
           "Guess which loop is slow and unroll it", "Restart the machine", "Upgrade every dependency at once"]
    qs = []
    for n in range(12):
        g, b = good[n % 6], bad[n % 6]
        flip = n % 2 == 1
        choices = [b, g] if flip else [g, b]
        qs.append({"id": f"fx{n}", "domain": "engineering", "method": "parallel", "cell": "parallel_engineering",
                   "task_id": f"task{n}", "query": f"Fixture task {n}: make the build pass.",
                   "prefix_text": f"Step 1: opened the repository. Step 2: saw a failure in module {n}.",
                   "choices": choices, "answer": "B" if flip else "A", "answer_index": 1 if flip else 0})
    return qs


def load_questions(domain, n):
    import pandas as pd
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{domain}.parquet")
    if not os.path.exists(path):
        tok_path = os.path.expanduser("~/.cache/huggingface/token")
        tok = open(tok_path).read().strip() if os.path.exists(tok_path) else ""
        req = urllib.request.Request(HF.format(domain=domain), headers={"Authorization": f"Bearer {tok}"} if tok else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                f.write(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"Taste-Bench download refused (HTTP {e.code}). The dataset is gated: accept its terms at "
                     f"https://huggingface.co/datasets/wenbopan/taste-bench with the account whose token is in "
                     f"{tok_path}, then run again.")
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    for r in rows:
        r["choices"] = list(r["choices"])
    rows.sort(key=lambda r: r["id"])
    random.Random(SEED).shuffle(rows)
    return rows[:n] if n else rows


# ------------------------------------------------------------------ memory
def lesson(q):
    good = q["choices"][q["answer_index"]]
    bad = q["choices"][1 - q["answer_index"]]
    return (f"Lesson from task: {q['query'][:300]}\nAt a decision fork, choosing \"{good[:400]}\" over "
            f"\"{bad[:400]}\" led to the better outcome.")


def build_memory(pool):
    m = Inspeximus()
    owner = {}
    for q in pool:
        owner[m.remember(lesson(q))] = q
    return m, owner


def recalled(m, owner, q, k=K, cross_repo=False, items=False):
    hits = m.recall(recall_query(q), k=k * 12, observe=False)
    out = []
    for h in hits:
        p = owner.get(h["id"])
        if p is None or p["task_id"] == q["task_id"]:
            continue
        if cross_repo and repo_of(p) == repo_of(q):
            continue
        out.append(p)
        if len(out) == k:
            break
    assert all(p["task_id"] != q["task_id"] for p in out), "leave-task-out violated"
    return out if items else [lesson(p) for p in out]


EMBED_URL = "http://127.0.0.1:11434/api/embed"


def _nomic(prefix):
    def f(text):
        body = json.dumps({"model": "nomic-embed-text", "input": prefix + text[:6000]}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(EMBED_URL, body, {"Content-Type": "application/json"}),
                                             timeout=120))
        return r["embeddings"][0]
    return f


def build_memory_sem(pool):
    """Same lessons, same store, but recall ranks by nomic-embed-text similarity (search_document /
    search_query prefixes), so relevance can be judged on meaning, not shared words."""
    m = Inspeximus(embed=_nomic("search_document: "), embed_query=_nomic("search_query: "))
    owner = {}
    for q in pool:
        owner[m.remember(lesson(q))] = q
    return m, owner


def recall_query(q):
    """The fork as a query: the task, the tail of the trajectory nearest the decision, and both options."""
    return q["query"][:600] + "\n" + q["prefix_text"][-1500:] + "\n" + "\n".join(c[:400] for c in q["choices"])


ISOLATE_WORDS = ("isolat", "stub", "custom", "probe", "download", "minimal", "standalone", "separate")


def repo_of(q):
    m = re.match(r"instance_(.+?__.+?)-[0-9a-f]{20,}", q["task_id"])
    return m.group(1) if m else q["task_id"].split("-")[0]


def prior_sig(good, bad):
    """The surface prior the red-team found: longer option, and the 'isolated/stub/probe' option.
    A lesson's signature is which way it points on both axes."""
    iso = lambda t: any(w in t.lower() for w in ISOLATE_WORDS)
    return (len(good) > len(bad), iso(good) and not iso(bad), iso(bad) and not iso(good))


def heuristic_pick(q):
    """Zero-model baseline: prefer the isolated/stub option, else the longer one (red-team: ~0.63-0.67)."""
    c = q["choices"]
    iso = [any(w in x.lower() for w in ISOLATE_WORDS) for x in c]
    if iso[0] != iso[1]:
        return 0 if iso[0] else 1
    return 0 if len(c[0]) >= len(c[1]) else 1


def lesson_sig(p):
    return prior_sig(p["choices"][p["answer_index"]], p["choices"][1 - p["answer_index"]])


def matched_control(pool, q, recalled_pool_items, rnd):
    """Random other-task lessons with the SAME prior signature as the recalled ones: whatever the
    recalled set teaches through surface cues, this set teaches too. A lead over it is relevance."""
    out = []
    for r in recalled_pool_items:
        sig = lesson_sig(r)
        cands = [p for p in pool if p["task_id"] != q["task_id"] and p is not r and lesson_sig(p) == sig]
        if cands:
            out.append(lesson(rnd.choice(cands)))
    return out


def inverted(text):
    """Swap the direction of a lesson: tests whether the model reads lesson content at all."""
    m = re.search(r'choosing "(.*)" over "(.*)" led to the better outcome', text, re.S)
    if not m:
        return text
    return text[:m.start()] + f'choosing "{m.group(2)}" over "{m.group(1)}" led to the better outcome' + text[m.end():]


FILLER_SENTENCE = ("This note is a calibration placeholder and carries no information about the task or the options. ")


def filler(lessons):
    """Neutral text of the same length as the recalled lessons, under the same header: isolates
    'any extra text in the prompt' from 'lesson content'."""
    n = sum(len(x) for x in lessons) or 600
    per = max(1, len(lessons) or K)
    unit = (FILLER_SENTENCE * (n // len(FILLER_SENTENCE) + 1))[: n // per]
    return [unit for _ in range(per)]


def random_lessons(pool, q, rnd, k=K):
    others = [p for p in pool if p["task_id"] != q["task_id"]]
    return [lesson(p) for p in rnd.sample(others, min(k, len(others)))]


# ------------------------------------------------------------------ prompt + model
def fit_prefix(text):
    if len(text) <= MAX_PREFIX_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n\n[... middle of the trajectory omitted ...]\n\n" + text[-(MAX_PREFIX_CHARS - HEAD_CHARS):]


def prompt(q, order, lessons):
    choices = q["choices"] if order == "published" else list(reversed(q["choices"]))
    mem = ""
    if lessons:
        mem = "\n\nNotes from past experience on other tasks (may or may not apply):\n" + \
              "\n".join(f"- {x}" for x in lessons)
    user = (f"TASK GIVEN TO THE AGENT:\n{q['query']}\n\nTRAJECTORY UP TO THE DECISION POINT:\n"
            f"{fit_prefix(q['prefix_text'])}{mem}\n\nThe agent must choose the next step. Which option leads "
            f"to the better outcome?\nA) {choices[0]}\nB) {choices[1]}\n\nEnd your reply with exactly one line: "
            f"Final answer: A or Final answer: B")
    correct = q["answer"] if order == "published" else ("B" if q["answer"] == "A" else "A")
    return user, correct


def parse(text):
    t = text or ""
    m = re.findall(r"final answer\s*:?\s*\**\s*([AB])\b", t, re.I)
    if m:
        return m[-1].upper()
    # Fallback for an unambiguous verdict sentence only (pilot 1: some replies ended "Option A is the better choice.")
    m = re.findall(r"\boption\s*\**([AB])\**\s+is\s+(?:the\s+)?(?:better|correct|best|right)\b", t, re.I)
    if m and len({x.upper() for x in m}) == 1:
        return m[-1].upper()
    return None


LOCAL_CTX = 12288  # pilot 1: prompts reach ~7.3k tokens, and 8192 cut 42 of 240 replies before the final line


def gpu_free_mb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def call(model, user):
    # Thinking OFF: the Taste-Bench paper reports that a larger reasoning budget does not improve accuracy,
    # and measured 2026-09-25 GLM spent 4,471 output tokens a call thinking, 27 % hitting the cap with no answer.
    body = {"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": 2500,
                        **({} if model.endswith("cloud") else {"num_ctx": LOCAL_CTX})}}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)


# ------------------------------------------------------------------ scoring
def both_correct(recs, cond, ids):
    by = {(r["id"], r["order"]): r["ok"] for r in recs if r["cond"] == cond}
    return [int(by.get((i, "published"), 0) and by.get((i, "reversed"), 0)) for i in ids]


def boot_diff(a, b, n=5000):
    rnd = random.Random(SEED)
    d = [x - y for x, y in zip(a, b)]
    stats = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return round(sum(d) / len(d), 4), round(stats[int(0.025 * n)], 4), round(stats[int(0.975 * n)], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--domain", default="engineering")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="qwen3.8:27b",
                    help="local by default to save Ollama cloud credit (owner 2026-09-26); a :cloud tag must be explicit")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", default="", help="comma list to run only these arms (e.g. none,control_matched,memory_sem)")
    a = ap.parse_args()

    pool = fixture_questions() if a.fixture else load_questions(a.domain, 0)
    rnd = random.Random(SEED)
    items = pool if a.fixture else pool[:a.n]
    m, owner = build_memory(pool)
    ms, owner_s = (None, None) if a.fixture else build_memory_sem(pool)
    print(f"[{el()}] memory built: lexical + semantic (nomic)", flush=True)
    jobs = []
    for q in items:
        if a.fixture:
            mem = recalled(m, owner, q)
            ctx = {"none": [], "memory": mem, "control": random_lessons(pool, q, rnd), "filler": filler(mem)}
        else:
            sem_items = recalled(ms, owner_s, q, items=True)
            sem = [lesson(p) for p in sem_items]
            ctx = {"none": [], "filler": filler(sem),
                   "control_matched": matched_control(pool, q, sem_items, rnd),
                   "memory_sem": sem,
                   "memory_sem_xrepo": recalled(ms, owner_s, q, cross_repo=True),
                   "inverted": [inverted(x) for x in sem]}
            if items.index(q) >= INVERTED_ON_FIRST:
                ctx.pop("inverted")  # variant C: the content-reading check runs on the first 100 forks only
            q["_same_repo_share"] = sum(repo_of(p) == repo_of(q) for p in sem_items) / max(1, len(sem_items))
        if a.arms:
            ctx = {c: v for c, v in ctx.items() if c in a.arms.split(",")}
        for cond, lessons in ctx.items():
            for order in ("published", "reversed"):
                user, correct = prompt(q, order, lessons)
                jobs.append({"id": q["id"], "cond": cond, "order": order, "user": user, "correct": correct,
                             "n_lessons": len(lessons)})
    chars = sum(len(j["user"]) for j in jobs)
    print(f"[{el()}] {len(items)} forks x {len(jobs) // max(1, 2 * len(items))} conditions x 2 orders = {len(jobs)} calls; "
          f"~{chars // 4:,} input tokens (chars/4), plus the model's own output", flush=True)
    if a.estimate:
        return

    if a.fixture:
        # Plumbing: an oracle that picks the option a lesson names as the better one, else always "A".
        # Expected: memory 1.0 (a lesson from ANOTHER task names the right option), none 0.0 (always A is
        # right in only one of the two orders), control in between. Anything else means the harness is broken.
        for j in jobs:
            opts = dict(re.findall(r"^([AB])\) (.*)$", j["user"], re.M))
            notes = j["user"].split("Notes from past experience")[1] if "Notes from past experience" in j["user"] else ""
            good = re.findall(r'choosing "(.*?)" over', notes)
            said = next((L for L, t in opts.items() if any(t[:40] == g[:40] for g in good)), "A")
            j.update(ok=int(said == j["correct"]), tin=0, tout=0)
    else:
        out_path = os.path.join(HERE, f"taste_memory_calls_{a.domain}_{CONFIG}.jsonl")
        done = {}
        if os.path.exists(out_path):
            for line in open(out_path, encoding="utf-8"):
                r = json.loads(line)
                if r.get("model") == a.model:
                    done[(r["id"], r["cond"], r["order"])] = r
        todo = [j for j in jobs if (j["id"], j["cond"], j["order"]) not in done]
        resident = False
        try:  # the model already sitting fully on the GPU needs no more room
            ps = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5)).get("models", [])
            resident = any(x.get("name") == a.model and x.get("size_vram", 0) >= x.get("size", 1) for x in ps)
        except Exception:
            pass
        if todo and not a.model.endswith("cloud") and not resident and gpu_free_mb() < 17500:
            sys.exit(f"GPU has {gpu_free_mb()} MB free; the local model needs ~17.5 GB. Waiting is the rule: "
                     f"run again when ComfyUI is idle. No cloud fallback.")
        print(f"[{el()}] resuming: {len(done)} cached, {len(todo)} to call, {a.workers} workers", flush=True)

        def comfy_busy():
            try:
                qq = json.load(urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3))
                return bool(qq.get("queue_running") or qq.get("queue_pending"))
            except Exception:
                return False

        def run(j):
            waited = 0
            while not a.model.endswith("cloud") and comfy_busy() and waited < 4 * 3600:
                time.sleep(30)  # WATCH / ComfyUI has priority on the GPU; we wait, never go to CPU or cloud
                waited += 30
            try:
                text, tin, tout = call(a.model, j["user"])
            except Exception as e:
                text, tin, tout = f"ERROR {e}", 0, 0
            return {"id": j["id"], "cond": j["cond"], "order": j["order"], "model": a.model,
                    "answer": parse(text), "correct": j["correct"], "ok": int(parse(text) == j["correct"]),
                    "tin": tin, "tout": tout, "tail": text[-300:]}

        with cf.ThreadPoolExecutor(a.workers) as pool_x, open(out_path, "a", encoding="utf-8") as f:
            for n, r in enumerate(pool_x.map(run, todo), 1):
                f.write(json.dumps(r) + "\n"); f.flush()
                done[(r["id"], r["cond"], r["order"])] = r
                if n % 10 == 0 or n == len(todo):
                    print(f"[{el()}] {n}/{len(todo)} calls", flush=True)
        for j in jobs:
            r = done[(j["id"], j["cond"], j["order"])]
            j.update(ok=r["ok"], tin=r["tin"], tout=r["tout"], answer=r["answer"])

    ids = [q["id"] for q in items]
    conds = [c for c in ("none", "filler", "control", "control_matched", "memory", "memory_sem", "memory_sem_xrepo",
                         "inverted") if any(j["cond"] == c for j in jobs)]
    # Each arm is scored on the forks it actually ran: "inverted" runs on the first INVERTED_ON_FIRST forks
    # only, and dividing it by all forks read 0.069 where the true rate was 0.270 (found 2026-09-26).
    # Paired differences use the forks both arms share.
    cond_ids = {c: [i for i in ids if any(j["id"] == i and j["cond"] == c for j in jobs)] for c in conds}
    res_by = {c: dict(zip(cond_ids[c], both_correct(jobs, c, cond_ids[c]))) for c in conds}
    res = {c: [res_by[c][i] for i in cond_ids[c]] for c in conds}
    rate = {c: round(sum(v) / len(v), 4) for c, v in res.items()}
    n_forks_per_arm = {c: len(v) for c, v in res.items()}

    def paired(x, y):
        common = [i for i in cond_ids[x] if i in res_by[y]]
        return boot_diff([res_by[x][i] for i in common], [res_by[y][i] for i in common])
    out = {"metric": "taste_lift_from_memory", "dataset": "wenbopan/taste-bench" if not a.fixture else "fixture",
           "domain": a.domain, "model": a.model if not a.fixture else "fixture-oracle", "n_forks": len(ids),
           "k_lessons": K, "both_correct_rate": rate,
           # the headline pair: the recall arm against no lessons and against prior-matched random lessons
           "n_forks_per_arm": n_forks_per_arm,
           "memory_minus_none": paired(MEM, "none") if (MEM := "memory" if "memory" in res else "memory_sem") else None,
           "memory_minus_control": paired(MEM, "control" if "control" in res else "control_matched"),
           "diffs": {f"{x}_minus_{y}": paired(x, y) for x in conds for y in conds if x != y},
           "position_locked": {c: sum(1 for i in ids if (lambda a_, b_: a_ is not None and a_ == b_)(
               *[next((j.get("answer") for j in jobs if j["id"] == i and j["cond"] == c and j["order"] == o), None)
                 for o in ("published", "reversed")])) for c in conds},
           "tokens": {"input": sum(j["tin"] for j in jobs), "output": sum(j["tout"] for j in jobs)},
           "heuristic_zero_model": round(sum(heuristic_pick(q) == q["answer_index"] for q in items) / len(items), 4),
           "same_repo_share_of_recall": round(sum(q.get("_same_repo_share", 0) for q in items) / len(items), 4),
           "raw": {"ids": ids, "ids_per_arm": cond_ids, **res}, "seed": SEED}
    name = "taste_memory_fixture_result.json" if a.fixture else f"taste_memory_{a.domain}_result.json"
    json.dump(out, open(os.path.join(HERE, name), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    main()
