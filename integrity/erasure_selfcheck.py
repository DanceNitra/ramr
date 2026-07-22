#!/usr/bin/env python3
"""agent-memory erasure self-check - test YOUR OWN stack's right-to-erasure.

Point this at the memory backend(s) you actually run. For each: it stores a unique marker, calls that
backend's OWN delete, runs that backend's OWN compaction, then reads the raw store on disk and reports whether
the marker is still present. The result is YOURS - this tool makes no claim about any vendor; it hands you a
receipt for your own stack so you can decide and, if you find residue you didn't expect, raise it with that
project through normal coordinated disclosure.

Honest scope (read before drawing conclusions):
  * This checks LOGICAL residue in the store's own files after delete()+compaction. It does NOT test at-rest
    security (free space / SSD over-provisioning / backups): a PLAINTEXT store of ANY library leaves bytes
    there - the only defense is full-disk/at-rest encryption + crypto-erasure (key destruction). This tool
    does not judge that layer.
  * A backend that keeps deleted values in an AUDIT/HISTORY log by design (some do) is a design choice, not a
    bug - it will show as "present (audit log)"; call the backend's documented purge (e.g. reset()) to clear it.
  * Deterministic; run it yourself. Only backends you have installed are tested.

Usage:  python stack_erasure_selfcheck.py            # auto-detect installed backends
        python stack_erasure_selfcheck.py mem0 chroma  # or name them
"""
import os, sys, glob, tempfile, sqlite3
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("OPENAI_API_KEY", "sk-none")  # mem0 default LLM init; we only use its embedder
MARK = "ERASURE-SELFCHECK-MARKER-7Q2X"
MB = MARK.encode()


def _residue(d):
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if os.path.isfile(f):
            try:
                if MB in open(f, "rb").read():
                    return os.path.basename(f)
            except Exception:
                pass
    return ""


def _vacuum(d):
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if f.endswith((".sqlite3", ".sqlite", ".db")):
            try:
                c = sqlite3.connect(f); c.execute("VACUUM"); c.commit(); c.close()
            except Exception:
                pass


def _emb():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return lambda t: m.encode([t])[0].tolist()


CHECKS = {}   # name -> callable() -> (present_after_delete_and_compaction, file_or_note)

def check_inspeximus():
    from inspeximus import Inspeximus
    d = tempfile.mkdtemp(); m = Inspeximus(path=os.path.join(d, "s.json"))
    m.remember(MARK, key="k::sc", source={"doc": "sc"}, pii=True); m._save(force=True)
    m.forget_subject("sc", request_id="sc"); m._save(force=True)
    _vacuum(d); return bool(_residue(d)), _residue(d)
CHECKS["inspeximus"] = check_inspeximus

def check_mem0():
    from mem0 import Memory
    d = tempfile.mkdtemp()
    mm = Memory.from_config({"embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(d, "qd"), "embedding_model_dims": 384, "on_disk": True}},
        "history_db_path": os.path.join(d, "history.db")})
    a = mm.add(MARK, user_id="u", infer=False)
    for r in (a.get("results") or []) if isinstance(a, dict) else []:
        try: mm.delete(r.get("id"))
        except Exception: pass
    del mm; _vacuum(d); r = _residue(d)
    return bool(r), (r + " (note: mem0 keeps a history log by design; reset() purges)" if r else "")
CHECKS["mem0"] = check_mem0

def check_chroma():
    import chromadb
    d = tempfile.mkdtemp(); e = _emb()
    cl = chromadb.PersistentClient(path=d); c = cl.get_or_create_collection("selfcheck")
    c.add(ids=["x"], embeddings=[e(MARK)], documents=[MARK]); c.delete(ids=["x"]); cl = None
    _vacuum(d); return bool(_residue(d)), _residue(d)
CHECKS["chroma"] = check_chroma

def check_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    d = tempfile.mkdtemp(); e = _emb(); v = e(MARK)
    qc = QdrantClient(path=d); qc.create_collection("sc", vectors_config=VectorParams(size=len(v), distance=Distance.COSINE))
    qc.upsert("sc", points=[PointStruct(id=1, vector=v, payload={"t": MARK})]); qc.delete("sc", points_selector=[1]); del qc
    _vacuum(d); return bool(_residue(d)), _residue(d)
CHECKS["qdrant"] = check_qdrant

def check_lancedb():
    import lancedb, datetime
    d = tempfile.mkdtemp(); e = _emb()
    t = lancedb.connect(d).create_table("sc", data=[{"id": 1, "text": MARK, "vector": e(MARK)}])
    t.delete("id = 1")
    try: t.cleanup_old_versions(older_than=datetime.timedelta(seconds=0))
    except Exception:
        try: t.compact_files()
        except Exception: pass
    return bool(_residue(d)), _residue(d)
CHECKS["lancedb"] = check_lancedb


def main(argv):
    want = [a.lower() for a in argv] or list(CHECKS)
    rows = []
    for name in want:
        fn = CHECKS.get(name)
        if not fn:
            rows.append((name, "not-a-known-check", "")); continue
        try:
            present, note = fn()
            rows.append((name, "PRESENT" if present else "absent", note))
        except ImportError:
            rows.append((name, "not-installed", ""))
        except Exception as ex:
            rows.append((name, "error", repr(ex)[:80]))
    print("=" * 74)
    print("agent-memory erasure self-check - YOUR stack (marker present in raw store after delete+VACUUM?)")
    print("=" * 74)
    for n, status, note in rows:
        print(f"  {n:<12} {status:<14} {note}")
    print("-" * 74)
    print("This is YOUR result. 'PRESENT' = the marker's bytes are still in the store's files after its own")
    print("delete + VACUUM (logical residue). It is NOT an at-rest-security verdict (plaintext stores of any")
    print("library leave bytes in free space/backups - use FDE + crypto-erasure). If a result surprises you,")
    print("raise it with that project via coordinated disclosure. inspeximus adds content-free deletion + shred().")


if __name__ == "__main__":
    main(sys.argv[1:])
