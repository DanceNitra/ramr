"""RAMR ECHO-RESISTANCE (cross-backend, ANSWER-LEVEL) -- the fair comparison across real memory systems.

ramr_echo_resistance.py scores echo-resistance at RETRIEVAL top-1 (cloud-free, lexical) -- fair for a
supersession store (inspeximus) whose design is to REMOVE the stale value, but UNFAIR to an add-based store
(mem0) whose design keeps both values and reconciles at READ time (top-k handed to an answering LLM). So a
top-1 retrieval metric is a strawman for mem0 (like scoring a bi-temporal graph on a config it never uses).

This variant scores ANSWER-LEVEL echo-resistance, fair to BOTH designs: ingest old -> correct to new
[-> echo old], recall top-k from the backend, hand those memories to a judge LLM, ask for the CURRENT
value, and check it is `new`. Sequence per topic:
  1. assert "ENT region is OLD"; 2. "correction: ENT region is NEW"; (forget-precision: answer == NEW?)
  3. echo: re-state the OLD value (value-preserving); (echo-resistance: answer STILL == NEW?)

Backends: inspeximus (echo_guard off / on) always; mem0 if installed (`pip install mem0ai chromadb`). Add your
own backend with a 3-method adapter (reset / store / recall_texts). Value-preserving echoes only (a
value-obscuring "go back to the old one" is out of scope for any object-level defense).

REQUIRES a cloud/OpenAI-compatible LLM for the judge (and for mem0's extraction): set OPENAI_API_KEY and,
for a non-OpenAI endpoint, OPENAI_BASE_URL + JUDGE_MODEL (e.g. an Ollama/vLLM OpenAI-compatible server).
This is the one RAMR metric that is NOT cloud-free (an answer-level metric needs an answerer). n=ER_N (30).
RUN: OPENAI_API_KEY=... OPENAI_BASE_URL=... JUDGE_MODEL=... python ramr_echo_resistance_backends.py
"""
import os, sys, json, tempfile, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspeximus import Inspeximus

N = int(os.getenv("ER_N", "30"))
BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

ENTS = ["payment api", "auth service", "search index", "billing job", "cache layer", "upload queue",
        "report engine", "email worker", "session store", "rate limiter", "image cdn", "audit log",
        "webhook relay", "config loader", "metrics sink", "backup task", "login flow", "export tool",
        "notify bus", "schema migrator", "token vault", "flag svc", "data lake", "shard router",
        "pdf renderer", "geo lookup", "fraud check", "recommend svc", "chat gateway", "trace collector"]
OLD = ["frankfurt", "oregon", "dublin", "tokyo", "virginia", "sydney", "london", "mumbai", "toronto", "paris",
       "seoul", "milan", "osaka", "cairo", "lima", "accra", "oslo", "riga", "sofia", "hanoi", "quito", "amman",
       "tunis", "davao", "cebu", "kigali", "napoli", "bern", "utah", "denver"]
NEW = ["ohio", "belgium", "norway", "kenya", "chile", "ghana", "peru", "egypt", "vietnam", "jordan", "rwanda",
       "zurich", "cebu", "utah", "oslo", "riga", "sofia", "hanoi", "quito", "amman", "tunis", "davao", "napoli",
       "denver", "seoul", "milan", "osaka", "cairo", "lima", "accra"]


def judge(ent, mems, old, new):
    ctx = "\n".join(f"- {t}" for t in mems)
    q = (f"Memories:\n{ctx}\n\nBased ONLY on these, what is the CURRENT region of {ent}? "
         f"Reply with ONLY the single place name.")
    body = json.dumps({"model": JUDGE, "messages": [{"role": "user", "content": q}],
                       "max_tokens": 8000, "temperature": 0.0}).encode()
    for a in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}), timeout=120)
            ans = json.loads(r.read())["choices"][0]["message"]["content"].strip().lower()
            return "new" if new in ans else ("old" if old in ans else "other")
        except Exception:
            if a == 2:
                return None
            time.sleep(3)


# ---------- backend adapters (reset / store / recall_texts) ----------
class InspeximusBackend:
    def __init__(self, guard): self.guard = guard; self.name = f"inspeximus (echo_guard={'on' if guard else 'off'})"
    def reset(self):
        fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(p)
        self.m = Inspeximus(path=p); self.m.echo_guard = self.guard
    def store(self, text, key, object, value=1.0): self.m.remember(text, key=key, object=object, value=value)
    def recall_texts(self, query): return [r["text"] for r in self.m.recall(query, k=5, mode="lexical")]

class Mem0Backend:
    name = "mem0 (add-based, read-time reconcile)"
    def __init__(self):
        from mem0 import Memory
        self._Memory = Memory
    def reset(self):
        cdir = tempfile.mkdtemp()
        cfg = {"llm": {"provider": "openai", "config": {"model": JUDGE, "openai_base_url": BASE,
                       "api_key": KEY, "temperature": 0.0}},
               "embedder": {"provider": "ollama", "config": {"model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")}},
               "vector_store": {"provider": "chroma", "config": {"path": cdir, "collection_name": "c" + os.urandom(4).hex()}}}
        self.m = self._Memory.from_config(cfg)
    def store(self, text, key, object, value=1.0): self.m.add(text, user_id="t")
    def recall_texts(self, query):
        res = self.m.search(query, filters={"user_id": "t"})
        rs = res.get("results") if isinstance(res, dict) else res
        return [x.get("memory") for x in rs]


def score(be, kind="reworded"):
    persist = echo = 0
    for i in range(N):
        ent, old, new = ENTS[i], OLD[i], NEW[i]
        be.reset()
        be.store(f"{ent} region is {old}", key=f"{ent}::r", object=old, value=5.0)
        be.store(f"correction: {ent} region is {new}", key=f"{ent}::r", object=new, value=1.0)
        if judge(ent, be.recall_texts(f"{ent} region"), old, new) == "new":
            persist += 1
        et = f"{ent} region is {old}" if kind == "verbatim" else f"just to confirm, {ent} stays in {old}"
        be.store(et, key=f"{ent}::r", object=old, value=5.0)
        if judge(ent, be.recall_texts(f"{ent} region"), old, new) == "new":
            echo += 1
    return round(persist / N, 3), round(echo / N, 3)


def main():
    if not KEY:
        print("Set OPENAI_API_KEY (+ OPENAI_BASE_URL / JUDGE_MODEL for a non-OpenAI endpoint)."); return
    backends = [InspeximusBackend(False), InspeximusBackend(True)]
    try:
        backends.append(Mem0Backend())
    except Exception as e:
        print(f"(mem0 not benchmarked: {e}; `pip install mem0ai chromadb` to include it)")
    print(f"=== RAMR ECHO-RESISTANCE (answer-level, judge={JUDGE}, n={N}) ===\n")
    print(f"{'backend':38s} {'forget-prec':>11s} {'echo-resist':>12s}")
    print("-" * 63)
    out = {}
    for be in backends:
        p, e = score(be)
        print(f"{be.name:38s} {p:>11.2f} {e:>12.2f}")
        out[be.name] = {"forget_precision": p, "echo_resistance": e}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "echo_resistance_backends_result.json"), "w"), indent=2)
    print("\n1.0 = an agent reading top-k answers with the CURRENT value; lower = the echo resurrected the stale one.")
    print("-> echo_resistance_backends_result.json")


if __name__ == "__main__":
    main()
