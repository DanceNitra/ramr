"""Every file we ship, hashed and checked against data/manifest.json.

WHY THIS EXISTS, and it is not a hypothetical. The README says the dataset is "frozen to disk with a
sha256-pinned manifest". On 2026-08-12 that manifest pinned exactly ONE file, and its hash
(00f1587680672690...) matched **no committed version of that file** -- not the blob, not a CRLF
variant, not an LF variant, not a whitespace-stripped variant. The file has been committed once, in
June, as 036983181534ed72..., so the pin was wrong from the first commit and was never checked by
anything. Meanwhile the three trace artifacts we actually hand to people were not pinned at all.

A hash nobody verifies is not a pin, it is a claim -- and this one was false for seven weeks while the
README advertised it as the reproducibility guarantee.

Two rules this enforces, both learned the hard way in the same day:
  * every file listed in the manifest matches its recorded sha256, and
  * every artifact we publish is listed. An unpinned file cannot go stale loudly, only quietly.

The basis is the file bytes as stored in git; `.gitattributes` pins *.jsonl / *.json to eol=lf so the
working copy equals the blob on every platform. Without that, this repo quotes hashes that verify on
the machine that wrote them and nowhere else -- which it has already done, in a public thread.

    python verify_manifest.py        # exit 0 = every shipped artifact matches its pin
"""
import hashlib
import io
import json
import os
import sys

MANIFEST = os.path.join("data", "manifest.json")

#: Artifacts that MUST appear in the manifest. Listing them here rather than globbing is deliberate:
#: a glob would silently accept a new unpinned file by never noticing it was missing.
REQUIRED = [
    "data/ramr_chains_v0.1.0.jsonl",
    "ramr_traces_v0.1.jsonl",
    "ramr_traces_v0.2_blind.jsonl",
    "ramr_traces_v0.2_labels.jsonl",
    "ramr_traces_v0.3_blind.jsonl",
    "ramr_traces_v0.3_labels.jsonl",
    "data/ramr_chains_v0.2.0.jsonl",
    "ramr_traces_v0.5_blind.jsonl",
    "ramr_traces_v0.5_labels.jsonl",
]


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    if not os.path.exists(MANIFEST):
        print("MISSING: %s" % MANIFEST)
        return 2
    man = json.load(io.open(MANIFEST, encoding="utf-8"))
    files = man.get("files") or {}
    problems = []

    listed_paths = {(meta.get("path") or name).replace("\\", "/")
                    for name, meta in files.items()}
    for req in REQUIRED:
        if req not in listed_paths:
            problems.append("%s is published but NOT pinned in the manifest" % req)

    print("%-34s %-10s %s" % ("artifact", "status", "sha256"))
    print("-" * 74)
    for name, meta in files.items():
        path = (meta.get("path") or name).replace("\\", "/")
        if not os.path.exists(path):
            problems.append("%s is pinned but missing from the repo" % path)
            print("%-34s %-10s %s" % (name[:34], "MISSING", "-"))
            continue
        actual = sha256(path)
        want = meta.get("sha256")
        ok = actual == want
        if not ok:
            problems.append("%s: manifest says %s..., file is %s..."
                            % (path, str(want)[:16], actual[:16]))
        print("%-34s %-10s %s" % (name[:34], "ok" if ok else "MISMATCH", actual[:32]))

        n = meta.get("n")
        if n is not None:
            actual_n = sum(1 for line in io.open(path, encoding="utf-8") if line.strip())
            if actual_n != n:
                problems.append("%s: manifest says n=%s, file holds %d" % (path, n, actual_n))

    print()
    if problems:
        print("FAIL -- the pin does not hold:")
        for p in problems:
            print("  * %s" % p)
        return 1
    print("OK -- %d artifact(s) pinned and matching." % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
