"""Can anything resolve a contradiction from CONTENT ALONE? The first score v0.5 can carry.

THE QUESTION THE FIXTURE NOW ASKS. Noticing a contradiction is easy: two retrieved records agreeing on
every word but the last. The `contradiction twin` reference row gets 100% detection doing exactly that,
and pays for it with a false positive on every plant, because it cannot say WHICH half is stale. On
v0.2 a string rule could -- "the value that echoes elsewhere is the current one" scored 98.9%. On v0.5
the shortcut floor is 40.0% over 1.7% of traces, so that route is closed and the question is open:
given the text and nothing else, can a model tell the superseded value from the live one?

WHY IT MATTERS FOR A MEMORY PRODUCT, and this is the honest framing rather than a flattering one. A
store with supersession does not answer this question at all -- it RECORDS which write replaced which,
at write time, and reads the answer off provenance. Content-only resolution is what you are left with
when that record was never kept. So the pair of numbers is the argument: whatever a model reaches here,
against 100% for a store that kept the link. If content-only resolution turns out to be easy, the
provenance argument is weaker than we think, and that is worth knowing.

RUNS ON THE OLLAMA CLOUD TIER, not OpenAI: our own measurements are free credits, and the paid key is
reserved for running a competitor in its own recommended config. The key is read from a file and never
appears on a command line.

    python ramr_content_resolver.py --key-file ../agora/server/.env --limit 60
    python ramr_content_resolver.py --key-file ../agora/server/.env          # all of them
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE = "https://ollama.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash:0731-cloud"


def read_key(path, names=("OLLAMA_API_KEY", "AGORA_API_KEY", "OPENAI_API_KEY")):
    """Pull the first matching key out of a .env-style file. Never printed, never echoed."""
    if not path or not os.path.exists(path):
        return os.environ.get("OPENAI_API_KEY", "")
    text = io.open(path, encoding="utf-8", errors="replace").read()
    for n in names:
        m = re.search(r"^%s\s*=\s*(.+)$" % re.escape(n), text, re.M)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY", "")


def pairs_in(items):
    """Contradiction pairs: same text except the final token, different final token."""
    keys = {}
    for i in items:
        w = (i.get("text") or "").rstrip(".").split()
        if len(w) >= 2:
            keys.setdefault(tuple(w[:-1]), []).append(i)
    out = []
    for _, group in keys.items():
        if len(group) == 2 and group[0]["text"] != group[1]["text"]:
            out.append((group[0], group[1]))
    return out


def ask(base, key, model, query, context, a, b, timeout=120):
    prompt = (
        "Below are records retrieved from an agent's memory. Exactly two of them state the same fact "
        "with different values -- one is CURRENT and one is SUPERSEDED (stale).\n\n"
        "Records:\n%s\n\nThe two in conflict:\n  A: %s\n  B: %s\n\n"
        "Question the memory is being used to answer: %s\n\n"
        "Which is the SUPERSEDED one? Reply with exactly one character, A or B, and nothing else."
        % ("\n".join("- " + (i.get("text") or "") for i in context), a["text"], b["text"], query))
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 16000,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            txt = (json.loads(r.read())["choices"][0]["message"]["content"] or "").strip().upper()
            for ch in txt:
                if ch in ("A", "B"):
                    return ch
            return None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 + attempt * 3)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="content-only contradiction resolution on RAMR traces")
    ap.add_argument("--version", default="0.5")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)

    from ramr_score_traces import rebuild
    blind = "ramr_traces_v%s_blind.jsonl" % a.version
    labels = "ramr_traces_v%s_labels.jsonl" % a.version
    for p in (blind, labels):
        if not os.path.exists(p):
            print("MISSING: %s" % p)
            return 2
    key = read_key(a.key_file)
    if not key:
        print("no API key found -- pass --key-file pointing at a .env, or set OPENAI_API_KEY")
        return 2

    traces = rebuild(blind, labels)
    if a.limit:
        traces = traces[:a.limit]
    print("v%s | %d traces | %s | %d workers" % (a.version, len(traces), a.model, a.workers))

    work = []
    for t in traces:
        ps = pairs_in(t["retrieved"])
        if len(ps) != 1:            # no pair, or ambiguous: the resolver does not fire
            continue
        work.append((t, ps[0][0], ps[0][1]))
    print("traces carrying exactly one contradiction pair: %d of %d\n" % (len(work), len(traces)))

    t0 = time.time()
    done = [0]

    def run(job):
        t, x, y = job
        ans = ask(a.base, key, a.model, t.get("query") or "", t["retrieved"], x, y)
        done[0] += 1
        if done[0] % 25 == 0:
            print("  %d/%d  %.0fs elapsed" % (done[0], len(work), time.time() - t0), flush=True)
        return t, x, y, ans

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run, work))

    fired = right = empty = 0
    for t, x, y, ans in results:
        if ans is None:
            empty += 1
            continue
        fired += 1
        chosen = x if ans == "A" else y
        right += (chosen["kind"] == "planted")

    print("\ncontent-only resolution, %s" % a.model)
    print("  answered            : %d of %d  (%d empty/failed)" % (fired, len(work), empty))
    print("  correct             : %d of %d = %.1f%%" % (right, fired, 100.0 * right / fired if fired else 0))
    print("  chance for a binary : 50.0%%")
    print("  coverage of corpus  : %d of %d = %.1f%%" % (len(work), len(traces), 100.0 * len(work) / len(traces)))
    print("  elapsed             : %.0fs" % (time.time() - t0))
    print("\nRead against the shortcut floor for this version, and against a store that kept the")
    print("supersession link at write time -- that pair of numbers is the whole question.")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps({
            "version": a.version, "model": a.model, "traces": len(traces), "pairs": len(work),
            "answered": fired, "correct": right, "empty": empty,
            "accuracy": (right / fired) if fired else None}, indent=2) + "\n")
        print("wrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
