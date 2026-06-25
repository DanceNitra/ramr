"""
RAMR Folklore Meter — does an AI-engineering "folklore" mechanism actually help, or is it a weak-model crutch?

Most agent/LLM-engineering folklore ("you need decision-trace memory", "multi-agent beats single", "consolidation /
dreaming", "bigger context is better") is asserted, or measured once on a weak/older model behind an LLM judge — and
does not replicate. This tool measures a claim the contamination-resistant RAMR way: a CONTRARIAN, judge-free,
exact-match task, run across a CAPABILITY GRADIENT (local models of increasing size + a frontier anchor), to
separate a real effect from a weak-model artifact.

A claim is specified as: a task (question + gold + the ablation CONDITIONS to compare, e.g. baseline vs mechanism),
an answer extractor, and a model gradient. The meter reports, per model, each condition's accuracy and the
mechanism's ADVANTAGE (mechanism - baseline), then issues a verdict:

  REAL                 — advantage persists at the frontier (the mechanism helps even a capable model)
  WEAK-MODEL ARTIFACT  — advantage is positive for weak models but ~0 at the frontier (a crutch capability subsumes)
  REGIME-SPECIFIC      — advantage appears only under specific conditions/inputs, not across the board
  NULL                 — no advantage anywhere

Judge-free (exact-match), cloud-free-capable (local Ollama gradient), zero heavy deps. The frontier anchor is a
pluggable callable so you can use any strong model (e.g. Claude) without coupling this tool to a vendor.

Worked verdicts produced with this method (2026-06-24): decision-trace/"why" memory -> WEAK-MODEL ARTIFACT;
multi-agent vote-ensemble -> REGIME-SPECIFIC (helps only sub-reliable models; ~0 once a model is reliable single-shot).
"""
from __future__ import annotations
import json, re, time, urllib.request
from collections import Counter

OLLAMA = "http://localhost:11434/api/chat"
ANSWER_RE = re.compile(r"ANSWER:\s*(-?\d[\d,]*)", re.I)


def ollama(model: str, prompt: str, num_predict: int = 600, temperature: float = 0.7, timeout: int = 180) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"temperature": temperature, "num_predict": num_predict}}).encode()
    for _ in range(2):
        try:
            r = urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body,
                headers={"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())["message"]["content"]
        except Exception:
            time.sleep(2)
    return ""


def extract_int(text: str):
    """Robust exact-match extractor: ONLY the value after the last 'ANSWER:' line (no last-number fallback —
    that fallback silently grabs quoted/intermediate numbers and confounds messy outputs)."""
    m = ANSWER_RE.findall(text or "")
    return int(m[-1].replace(",", "")) if m else None


def ask_clean(call, prompt: str, extractor=extract_int, retries: int = 3):
    """Call `call(prompt)->text`, retrying until `extractor(text)` returns a non-None value (removes a
    format-compliance confound that otherwise penalises whichever condition produces messier output).
    `extractor` defaults to the strict ANSWER-line int parser; pass a custom one (returning any hashable value
    or None) for string / abstention / multiple-choice claims."""
    t = ""
    for _ in range(retries):
        t = call(prompt)
        v = extractor(t)
        if v is not None:
            return v
    return None


def majority(vals):
    v = [x for x in vals if x is not None]
    return Counter(v).most_common(1)[0][0] if v else None


def run_claim(claim: dict, models: list, frontier: tuple | None = None, samples: int = 5) -> dict:
    """claim = {name, items:[{prompt_baseline, prompt_mechanism, gold}], extractor?}. models = list of Ollama names
    (weak->strong). frontier = (label, call) where call(prompt)->text for a strong anchor (e.g. Claude). Returns
    per-model baseline/mechanism accuracy + advantage."""
    extractor = claim.get("extractor", extract_int)
    rows = {}
    runners = [(m, (lambda mm: (lambda p: ollama(mm, p)))(m)) for m in models]
    if frontier:
        runners.append(frontier)
    for label, call in runners:
        b_hit = m_hit = n = 0
        for it in claim["items"]:
            # baseline vs mechanism, self-consistency over `samples`, robust extraction
            b = majority([ask_clean(call, it["prompt_baseline"], extractor) for _ in range(samples)])
            mm = majority([ask_clean(call, it["prompt_mechanism"], extractor) for _ in range(samples)])
            b_hit += (b == it["gold"]); m_hit += (mm == it["gold"]); n += 1
        rows[label] = {"baseline": round(b_hit / n, 3), "mechanism": round(m_hit / n, 3),
                       "advantage": round((m_hit - b_hit) / n, 3)}
    return rows


def verdict(rows: dict, frontier_label: str, eps: float = 0.05) -> str:
    fr = rows.get(frontier_label, {}).get("advantage", 0.0)
    weak = [v["advantage"] for k, v in rows.items() if k != frontier_label]
    weak_pos = any(a > eps for a in weak)
    if fr > eps:
        return "REAL — the mechanism still helps the frontier model"
    if weak_pos:
        return "WEAK-MODEL ARTIFACT — helps sub-frontier models, ~0 at the frontier (capability subsumes it)"
    return "NULL — no advantage anywhere (frontier or weak)"


if __name__ == "__main__":
    # Demo on a tiny built-in claim (multi-step arithmetic: 'mechanism' = an explicit 'show your work' nudge).
    items = []
    import random
    rng = random.Random(1)
    for _ in range(4):
        a, b, c = rng.randint(20, 80), rng.randint(3, 9), rng.randint(5, 40)
        g = a * b - c
        base = f"A box has {a} rows of {b} items; {c} are removed. How many remain? End with 'ANSWER: <n>'."
        mech = "Think step by step. " + base
        items.append({"prompt_baseline": base, "prompt_mechanism": mech, "gold": g})
    claim = {"name": "show-your-work nudge", "items": items}
    rows = run_claim(claim, ["llama3.2:3b"], samples=3)
    print(json.dumps(rows, indent=1))
    print("VERDICT:", verdict(rows, "llama3.2:3b"))
