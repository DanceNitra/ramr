"""RAMR v0 -- the CONVERSION metric on a CONTAMINATION-RESISTANT synthetic multi-hop dataset.
Core question the field never measures: does retrieval RECALL convert into correct multi-hop ANSWERS, and how
does partial/irrelevant context change that? We generate composable fact-chains (hidden answer key, regenerable,
so no model can have memorized them), then run a local model under controlled context conditions and measure
ANSWER accuracy:
  closed   : no context (parametric only) -> should be ~0 (entities are random -> uncontaminated)
  gold     : exactly the K gold facts forming the chain (perfect, complete retrieval)
  partial  : gold minus ONE hop (a broken chain -> tests whether incomplete recall still answers)
  noisy    : gold facts + equal-count irrelevant distractor facts (realistic imperfect retrieval)
Metrics: accuracy per condition; CONVERSION = gold accuracy (does complete recall convert?); CHAIN-FRAGILITY =
gold - partial; DISTRACTION = noisy - gold. Cloud-free (local qwen via Ollama). ASCII prints."""
import os, re, json, time, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OLL = "http://localhost:11434/v1/chat/completions"
MODEL = os.getenv("RAMR_MODEL", "qwen3-coder:30b")
N_Q = int(os.getenv("RAMR_N", "40"))
HOPS = int(os.getenv("RAMR_HOPS", "3"))
rng = np.random.default_rng(7)

# synthetic vocab -> random, uncontaminated entities/values
PEOPLE = [f"{a}{b}" for a in ["Vor","Mil","Tan","Que","Rho","Bex","Cal","Dun","Esk","Fir","Gad","Hox"] for b in ["ander","ette","ovic","wyn","ash","oro","ill","une"]]
COMPANIES = [f"{a}{b}" for a in ["Zyn","Orb","Vex","Lum","Kor","Pyx","Nim","Hal","Tre","Wisp"] for b in ["corp","tech","dyne","labs","works","soft","grid","ware"]]
CITIES = [f"{a}{b}" for a in ["Bru","Cas","Dor","Eln","Fel","Gri","Hol","Ino","Jor","Kel","Lun","Mor"] for b in ["ton","vik","gard","mere","port","haven","fell","stad"]]
COUNTRIES = [f"{a}{b}" for a in ["Az","Bo","Ca","Dr","Es","Fa","Gi","Ha","Il","Jo","Ka","Lo"] for b in ["landia","ovia","mark","stan","ria","nesia","gard"]]
CURRENCIES = [f"{a}{b}" for a in ["dra","fen","gul","hak","jin","kor","lim","mun","pol","rix"] for b in ["", "a", "o", "ek"]]


def gen_chain(r):
    """A HOPS-long chain: person -> company -> city -> country -> currency (trim to HOPS+1 nodes)."""
    p = r.choice(PEOPLE); c = r.choice(COMPANIES); city = r.choice(CITIES); ctry = r.choice(COUNTRIES); cur = r.choice(CURRENCIES)
    facts = [f"{p} works at {c}.", f"{c} is headquartered in {city}.", f"{city} is a city in {ctry}.",
             f"The currency of {ctry} is the {cur}."]
    # 3-hop question: currency of the country where the company that the person works at is located
    q = f"What is the currency of the country containing the city where the company that {p} works at is headquartered?"
    if HOPS == 3:
        q = f"What is the currency of the country where the company that {p} works at is headquartered?"
        # collapse city hop into country for a clean 3-hop (person->company->? ) -> use city->country chain
        facts = [f"{p} works at {c}.", f"{c} is headquartered in {ctry}.", f"The currency of {ctry} is the {cur}."]
    return {"q": q, "facts": facts, "answer": cur, "entities": [p, c, ctry, cur]}


def distractors(r, n):
    out = []
    for _ in range(n):
        kind = r.integers(4)
        if kind == 0: out.append(f"{r.choice(PEOPLE)} works at {r.choice(COMPANIES)}.")
        elif kind == 1: out.append(f"{r.choice(COMPANIES)} is headquartered in {r.choice(COUNTRIES)}.")
        elif kind == 2: out.append(f"The currency of {r.choice(COUNTRIES)} is the {r.choice(CURRENCIES)}.")
        else: out.append(f"{r.choice(CITIES)} is a city in {r.choice(COUNTRIES)}.")
    return out


def call(ctx, q):
    sysmsg = ("Answer using ONLY the provided facts; reason step by step in at most two sentences. If the facts "
              "are insufficient, say UNKNOWN. End with exactly:\nANSWER: <short answer>")
    usr = (("Facts:\n" + "\n".join(ctx) + "\n\n") if ctx else "") + "Question: " + q
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 300,
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": usr}]}
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                OLL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=180).read())
            return r["choices"][0]["message"].get("content") or ""
        except Exception:
            time.sleep(0.8)
    return ""


def hit(reply, ans):
    m = re.search(r"ANSWER:\s*(.+)", reply or "", re.I)
    pred = (m.group(1) if m else (reply or "")[-60:]).lower()
    return ans.lower() in re.sub(r"[^a-z0-9 ]", " ", pred)


if __name__ == "__main__":
    print(f"compile OK - RAMR v0 CONVERSION metric ({MODEL}, N={N_Q}, hops={HOPS}, synthetic/uncontaminated)", flush=True)
    qs = [gen_chain(rng) for _ in range(N_Q)]
    conds = ["closed", "gold", "partial", "noisy"]
    def ctx_for(item, cond):
        if cond == "closed": return []
        if cond == "gold": return list(item["facts"])
        if cond == "partial":
            f = list(item["facts"]); f.pop(int(rng.integers(len(f)))); return f   # drop one hop
        if cond == "noisy": return list(item["facts"]) + distractors(rng, len(item["facts"]))
        return []
    def run(item):
        out = {}
        for cond in conds:
            ctx = ctx_for(item, cond)
            if cond == "noisy": rng.shuffle(ctx)
            out[cond] = hit(call(ctx, item["q"]), item["answer"])
        return out
    with ThreadPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(run, qs))
    acc = {c: float(np.mean([r[c] for r in res])) for c in conds}
    print(f"\n  n={len(res)} synthetic {HOPS}-hop questions (random entities -> closed-book should be ~0):", flush=True)
    for c in conds: print(f"    {c:8s}: {acc[c]:.3f}", flush=True)
    print(f"\n  CONVERSION (gold accuracy)       : {acc['gold']:.3f}   (does complete recall convert to answers?)", flush=True)
    print(f"  CHAIN-FRAGILITY (gold - partial) : {acc['gold']-acc['partial']:+.3f}   (cost of one missing hop)", flush=True)
    print(f"  DISTRACTION (noisy - gold)       : {acc['noisy']-acc['gold']:+.3f}   (cost of irrelevant facts in context)", flush=True)
    print(f"  contamination check (closed)     : {acc['closed']:.3f}   (should be ~0 -> uncontaminated)", flush=True)
    json.dump({"acc": acc, "n": len(res), "model": MODEL, "hops": HOPS},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ramr_v0_result.json"), "w"))
    print("DONE", flush=True)
