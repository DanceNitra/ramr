"""RAMR COMPRESSION-vs-RAW metric: does a compiled/consolidated memory ever BEAT the raw context, or only lose to
it? FACT-RETENTION showed compaction LOSES facts vs a clean store. But raw context isn't always clean: when it is
large + noisy (the realistic regime, ~tens of chunks for one needed fact), a compiled summary that distills the
answer-relevant part might beat the noisy raw context (which buries the fact among distractors). So we sweep the
context noise K (distractor facts around the 1 target) and measure:
  RAW arm     : answer from all K+1 facts (target + K distractors), shuffled
  COMPILED arm: LLM-summarize the K+1 facts (<=S sentences), then answer from the summary
  COMPRESSION-LIFT = acc(compiled) - acc(raw)
Prediction: at low K, raw wins (compiled needlessly loses the fact); as K grows, compiled may catch up or beat raw
(noise hurts raw more than compaction hurts compiled) -- a crossover. FALSIFIER: if compiled never beats raw at any
K, compaction is strictly a cost (still a useful, honest result). Cloud-free (local qwen). ASCII."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramr_factret as fr
import ramr_v0_conversion as v0
from concurrent.futures import ThreadPoolExecutor

MODEL = os.getenv("CO_MODEL", "qwen3-coder:30b")
N = int(os.getenv("CO_N", "20"))
K_LEVELS = [int(x) for x in os.getenv("CO_K", "5,20,50").split(",")]
rng = np.random.default_rng(91)
ENTS = fr.ENTITIES; ATTRS = fr.ATTRS


def make(seed):
    r = np.random.default_rng(seed); out = []
    for i in range(N):
        e = r.choice(ENTS); a = r.choice(ATTRS); v = int(r.integers(2, 400))
        target = {"text": f"The {a} of the {e} is {v}.", "q": f"What is the {a} of the {e}?", "val": str(v)}
        out.append({"target": target, "e": e, "a": a})
    return out


def distractors(r, k, avoid_e, avoid_a):
    out = []
    for _ in range(k):
        e = r.choice(ENTS); a = r.choice(ATTRS)
        while e == avoid_e and a == avoid_a:
            e = r.choice(ENTS); a = r.choice(ATTRS)
        out.append({"text": f"The {a} of the {e} is {int(r.integers(2,400))}."})
    return out


if __name__ == "__main__":
    print(f"compile OK - COMPRESSION-vs-RAW ({MODEL}, N={N}, K={K_LEVELS})", flush=True)
    qs = make(7)
    rr = np.random.default_rng(3)
    def one(args):
        item, K = args
        ctx = [item["target"]] + distractors(rr, K, item["e"], item["a"])
        texts = [c["text"] for c in ctx]; rng.shuffle(texts)
        raw_hit = v0.hit(fr.probe(MODEL, "\n".join(texts), item["target"]["q"]), item["target"]["val"])
        summary = fr.compile_kb(MODEL, [{"text": t} for t in texts], fr.S_SENT)
        comp_hit = v0.hit(fr.probe(MODEL, summary or "", item["target"]["q"]), item["target"]["val"]) if summary else 0
        return (K, raw_hit, comp_hit)
    jobs = [(it, K) for K in K_LEVELS for it in qs]
    with ThreadPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(one, jobs))
    print(f"\n  K (distractors) | RAW acc | COMPILED acc | COMPRESSION-LIFT (compiled - raw)", flush=True)
    rows = {}
    for K in K_LEVELS:
        raw = np.mean([h for (k, h, c) in res if k == K]); comp = np.mean([c for (k, h, c) in res if k == K])
        rows[K] = (float(raw), float(comp), float(comp - raw))
        print(f"    {K:3d}           |  {raw:.2f}   |   {comp:.2f}      |   {comp-raw:+.2f}", flush=True)
    print(f"\n  === VERDICT ===", flush=True)
    best_lift = max(rows[K][2] for K in K_LEVELS); best_K = max(K_LEVELS, key=lambda K: rows[K][2])
    if best_lift > 0.05:
        print(f"  COMPRESSION CAN BEAT RAW: at K={best_K} distractors, the compiled summary outperforms the raw "
              f"noisy context by {best_lift:+.2f} (compiled {rows[best_K][1]:.2f} vs raw {rows[best_K][0]:.2f}). So "
              f"consolidation is not only acceptable loss -- past a noise threshold it is an ACCURACY GAIN, because "
              f"distilling the answer beats burying it among distractors. The crossover quantifies WHEN to compile.", flush=True)
    else:
        print(f"  COMPRESSION IS A COST (honest negative): compiled never beat raw by >0.05 (best {best_lift:+.2f} at "
              f"K={best_K}). Raw context, even noisy, kept the answer better than a lossy summary on this task -- "
              f"consolidation buys structure/cost, not accuracy. Either way, a clean measured answer.", flush=True)
    json.dump({"model": MODEL, "N": N, "rows": {str(K): rows[K] for K in K_LEVELS}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compression_oracle_result.json"), "w"))
    print("DONE", flush=True)
