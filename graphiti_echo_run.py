"""RAMR ECHO-RESISTANCE — real Zep/Graphiti runtime measurement.

Runs the correction -> echo -> recall -> judge protocol end-to-end against a real
Graphiti graph (Neo4j backend, OpenAI extraction/embedding), and disaggregates the
raw resurrection rate into (a) pre-echo extraction misses vs (b) echo-attributable
flips — the fair denominator for "does the echo actually resurrect a corrected fact?".

Setup:
  - Neo4j reachable at bolt://localhost:7687 (user neo4j / password below; override to taste).
  - pip install graphiti-core
  - export OPENAI_API_KEY=sk-...
  - optional: ER_N=30 (number of independent cases)
Result: prints per-case pre/post + a summary, writes graphiti_echo_result.json.
"""
import os, sys, json, asyncio, warnings, datetime as dt, urllib.request, time, random
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(errors="replace", line_buffering=True)
except Exception:
    pass
KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword123")
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

def judge(ent, facts, old, new):
    """Hand the recalled facts to a small instruct model: what is the CURRENT value?"""
    ctx = "\n".join(f"- {t}" for t in facts)
    q = (f"Facts:\n{ctx}\n\nBased ONLY on these, what is the CURRENT region of {ent}? "
         f"Reply with ONLY the place name.")
    body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": q}],
                       "max_tokens": 20, "temperature": 0}).encode()
    for a in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}),
                timeout=60).read())
            ans = r["choices"][0]["message"]["content"].strip().lower()
            return "new" if new in ans else ("old" if old in ans else "other")
        except Exception:
            if a == 2:
                return None
            time.sleep(3)

async def main():
    cfg = LLMConfig(api_key=KEY, model="gpt-4o-mini", small_model="gpt-4o-mini")
    g = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                 llm_client=OpenAIClient(config=cfg),
                 embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(
                     api_key=KEY, embedding_model="text-embedding-3-small", embedding_dim=1536)),
                 cross_encoder=OpenAIRerankerClient(config=cfg))
    await g.build_indices_and_constraints()
    ENTS = ["payment api", "auth service", "search index", "billing job", "cache layer", "upload queue",
            "report engine", "email worker", "session store", "rate limiter", "image cdn", "audit log",
            "webhook relay", "config loader", "metrics sink", "queue broker", "token vault", "render farm",
            "backup agent", "log shipper", "trace collector", "feature flag", "cron runner", "media encoder",
            "spam filter", "geo router", "ledger service", "notify hub", "index builder", "stream gateway"]
    OLD = ["frankfurt", "oregon", "dublin", "tokyo", "virginia", "sydney", "london", "mumbai", "toronto",
           "paris", "seoul", "milan", "osaka", "cairo", "lima", "madrid", "warsaw", "vienna", "athens",
           "lisbon", "helsinki", "oslo", "prague", "zagreb", "riga", "sofia", "tallinn", "malta", "porto", "nice"]
    NEW = ["ohio", "belgium", "norway", "kenya", "chile", "ghana", "peru", "egypt", "vietnam", "jordan",
           "rwanda", "zurich", "cebu", "utah", "denver", "bhutan", "fiji", "samoa", "tonga", "palau",
           "nauru", "tuvalu", "brunei", "qatar", "oman", "macau", "aruba", "belize", "guyana", "kigali"]
    now = dt.datetime(2026, 7, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
    per = []; prepost = []; persist = 0
    N = int(os.getenv("ER_N", "30")); t0 = time.time()
    for i in range(N):
        ent, old, new = ENTS[i], OLD[i], NEW[i]
        gid = f"g{i}_{os.urandom(2).hex()}"
        # 1. state the fact, 2. correct it
        await g.add_episode(name="a", episode_body=f"The {ent} region is {old.title()}.",
                            source_description="chat", reference_time=now, group_id=gid)
        await g.add_episode(name="b", episode_body=f"Correction: the {ent} region is now {new.title()}.",
                            source_description="chat", reference_time=now + dt.timedelta(minutes=1), group_id=gid)
        facts = [e.fact for e in await g.search(f"{ent} region", group_ids=[gid])]
        pre_new = judge(ent, facts, old, new) == "new"       # did the correction register?
        persist += 1 if pre_new else 0
        # 3. echo the OLD value
        await g.add_episode(name="c", episode_body=f"By the way, the {ent} region is {old.title()}.",
                            source_description="chat", reference_time=now + dt.timedelta(minutes=2), group_id=gid)
        facts = [e.fact for e in await g.search(f"{ent} region", group_ids=[gid])]
        held = judge(ent, facts, old, new) == "new"          # did it survive the echo?
        per.append(0 if held else 1)
        prepost.append((1 if pre_new else 0, 1 if held else 0))
        print(f"  {i+1}/{N} pre_new={1 if pre_new else 0} post_new={1 if held else 0} "
              f"resurrected={per[-1]} ({time.time()-t0:.0f}s)", flush=True)
    await g.close()
    n = len(per); res = sum(per) / n
    rng = random.Random(0)
    boot = sorted(sum(per[rng.randrange(n)] for _ in range(n)) / n for _ in range(5000))
    lo, hi = boot[125], boot[4875]
    # echo-attributable: among cases where the correction HELD pre-echo, how many did the echo flip?
    held_pre = [(a, b) for (a, b) in prepost if a == 1]
    echo_flip = sum(1 for (a, b) in held_pre if b == 0)
    echo_attr_res = echo_flip / len(held_pre) if held_pre else None
    out = {"backend": "graphiti-neo4j-native-openai", "n": n, "forget_precision": round(persist / n, 3),
           "echo_resistance": round(1 - res, 3), "resurrection_rate": round(res, 3),
           "resurrection_95CI": [round(lo, 3), round(hi, 3)],
           "correction_held_pre_echo": len(held_pre), "echo_attributable_flips": echo_flip,
           "echo_attributable_resurrection": round(echo_attr_res, 3) if echo_attr_res is not None else None,
           "note_metric": ("resurrection_rate counts ALL post-echo non-new (incl. pre-echo extraction failures); "
                           "echo_attributable = flips among the correction-held-pre-echo subset (the fair denominator)")}
    print(f"ECHO-ATTRIBUTABLE: correction held pre-echo in {len(held_pre)}/{n}; "
          f"echo flipped {echo_flip} of them = {echo_attr_res}")
    json.dump(out, open("graphiti_echo_result.json", "w"), indent=2)
    print(f"\nGRAPHITI NATIVE (neo4j+openai): n={n} forget-prec={persist/n:.2f} "
          f"echo-resistance={1-res:.2f} resurrection={res:.2f} 95CI=[{lo:.2f},{hi:.2f}]")

asyncio.run(main())
