"""
contamination_window.py

Infini-gram count queries of each document's text (default index: RedPajama, v4_rpj_llama_s4; --index
v4_olmo-mix-1124_llama for OLMo-2's pre-training corpus). The API accepts at most 1000 characters
(WINDOW) and 500 (Llama-2) tokens per query.

Short texts (<= WINDOW characters) are queried whole; a query over 500 tokens is retried on its first
500 tokens and marked "trimmed_to_tokens".

With --chunks (long texts) the document is cut into consecutive chunks of at most WINDOW characters and
500 tokens (cut locally with the Llama-2 tokenizer) and every chunk is queried ("chunks"). A partial last
chunk is replaced by the document's last WINDOW characters (cut to the last 500 tokens if longer), so the
end of the text is checked as a full-size chunk instead of a short leftover.

The query text is the stripped document text with no other normalisation (see query_normalized.py for
HoC, DMSAFN and ZSTFNS). Queries containing " AND " / " OR " are evaluated by the API as CNF, not
verbatim, and are flagged ("cnf_operator") so they can be excluded.

Per-document results: <out-root>/per_example/<dataset>/<split>-original-contaminated.jsonl
(resumable: documents already written are skipped).

    python src/leakage/contamination_window.py --datasets hallmarks_of_cancer Lc37Fnd AYSLAX
    python src/leakage/contamination_window.py --datasets N2C2_2008 --chunks --out-root ./outputs/leakage
    python src/leakage/contamination_window.py --index v4_olmo-mix-1124_llama --out-root ./outputs/leakage \
        --datasets PsyTar ZSTFNS DMSAFN hallmarks_of_cancer
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://api.infini-gram.io/"
INDEX = "v4_rpj_llama_s4"
FORBIDDEN_PAUSE = 600      # seconds to back off globally after an HTTP 403 (rate block)
MAX_FORBIDDEN_PAUSES = 6   # give up after this many consecutive blocks
MAX_TOKENS = 500           # the API's other limit, in the index's (Llama-2) tokens
RETRY_ERRORS = ("tokens", "request failed")  # rows with these errors are re-queried on resume

_pause_until = [0.0]
_consec_blocks = [0]
_lock = threading.Lock()
_stop = threading.Event()


def query(text, delay):
    payload = json.dumps({"index": INDEX, "query_type": "count", "query": text}).encode()
    for attempt in range(6):
        if _stop.is_set():
            return None
        wait = _pause_until[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            with _lock:
                _consec_blocks[0] = 0
            time.sleep(delay)
            return body
        except urllib.error.HTTPError as e:
            if e.code == 403:
                with _lock:
                    if _pause_until[0] <= time.time():
                        _consec_blocks[0] += 1
                        if _consec_blocks[0] > MAX_FORBIDDEN_PAUSES:
                            print("blocked by the API too often, stopping", flush=True)
                            _stop.set()
                            return None
                        print(f"HTTP 403: pausing {FORBIDDEN_PAUSE}s (block {_consec_blocks[0]})", flush=True)
                        _pause_until[0] = time.time() + FORBIDDEN_PAUSE
                continue
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return {"error": "request failed"}


_tok = []


def tokenizer():
    if not _tok:
        from transformers import AutoTokenizer
        _tok.append(AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer"))  # Llama-2 tokenizer
    return _tok[0]


def chunks(text, window):
    """Consecutive (start, end) chunks of text, each <= window chars and <= MAX_TOKENS tokens."""
    out, i = [], 0
    while i < len(text):
        c = text[i:i + window]
        enc = tokenizer()(c, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) > MAX_TOKENS:
            c = c[:enc["offset_mapping"][MAX_TOKENS - 1][1]]
        out.append((i, i + len(c)))
        i += len(c)
    return out


def shard_range(n, shard):
    """Documents [lo, hi) of shard "K/N": the K-th (0-based) of N consecutive, near-equal parts."""
    if not shard:
        return 0, n
    k, parts = map(int, shard.split("/"))
    return k * n // parts, (k + 1) * n // parts


def tail_window(text, window):
    """(start, end) of the last `window` characters of text, cut to its last MAX_TOKENS tokens."""
    start = max(0, len(text) - window)
    enc = tokenizer()(text[start:], add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) > MAX_TOKENS:
        start += enc["offset_mapping"][-MAX_TOKENS][0]
    return start, len(text)


def query_chunks(text, window):
    """chunks(text, window), with a partial last chunk replaced by tail_window(text, window)."""
    c = chunks(text, window)
    if len(c) > 1:
        a, b = c[-1]
        if b - a < window and len(tokenizer()(text[a:b], add_special_tokens=False).input_ids) < MAX_TOKENS - 1:
            c[-1] = tail_window(text, window)
    return c


def run(dataset, split, data_root, out_root, window, workers, delay, by_chunks=False, shard=None):
    src = f"{data_root}/{dataset}/original/{split}-original.jsonl"
    dst = f"{out_root}/per_example/{dataset}/{split}-original-contaminated.jsonl"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    texts = [json.loads(line)["text"] for line in open(src, encoding="utf-8") if line.strip()]
    done = set()
    if os.path.exists(dst):
        rows = [json.loads(line) for line in open(dst, encoding="utf-8") if line.strip()]
        keep = [r for r in rows if not any(e in (r.get("error") or "") for e in RETRY_ERRORS)]
        if len(keep) < len(rows):  # drop rows to re-query
            with open(dst, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in keep)
        done = {r["_idx"] for r in keep}
    lo, hi = shard_range(len(texts), shard)
    done &= set(range(lo, hi))
    todo = [i for i in range(lo, hi) if i not in done]
    print(f"{dataset}/{split}: {len(texts)} docs" + (f", shard {shard} = [{lo}, {hi})" if shard else "")
          + f", {len(done)} done, {len(todo)} to query", flush=True)

    def one(i):
        full = texts[i].strip()
        q = full[:window]
        body = query(q, delay) if q else {"error": "empty text"}
        if body is None:
            return None
        trimmed = False
        if "tokens" in (body.get("error") or ""):
            enc = tokenizer()(q, add_special_tokens=False, return_offsets_mapping=True)
            q = q[:enc["offset_mapping"][MAX_TOKENS - 1][1]]
            trimmed = True
            body = query(q, delay)
            if body is None:
                return None
        err = body.get("error")
        return {"_idx": i, "chars": len(full), "query_chars": len(q), "truncated": len(full) > len(q),
                "trimmed_to_tokens": trimmed,
                "cnf_operator": (" AND " in q or " OR " in q), "ntok": len(body.get("token_ids", [])),
                "count": -1 if err else int(body.get("count", -1)), "approx": body.get("approx"), "error": err}

    def one_chunks(i):
        full = texts[i].strip()
        res = []
        for a, b in query_chunks(full, window):
            q = full[a:b]
            body = query(q, delay)
            if body is None:
                return None
            err = body.get("error")
            res.append({"start": a, "end": b, "ntok": len(body.get("token_ids", [])),
                        "cnf_operator": (" AND " in q or " OR " in q),
                        "count": -1 if err else int(body.get("count", -1)), "approx": body.get("approx"), "error": err})
        failed = all(r["count"] < 0 for r in res)
        return {"_idx": i, "chars": len(full), "chunks": res, "error": "all chunks failed" if failed else None}

    fn = one_chunks if by_chunks else one
    with open(dst, "a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as ex:
        for n, rec in enumerate(ex.map(fn, todo), 1):
            if rec is None:
                break
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 500 == 0:
                print(f"  {dataset}: {len(done) + n}/{hi - lo}", flush=True)
    rows = [json.loads(line) for line in open(dst, encoding="utf-8") if line.strip()]
    qs = [q for r in rows if lo <= r["_idx"] < hi for q in r.get("chunks", [r])]
    ok = [q for q in qs if q["count"] >= 0]
    hits = sum(q["count"] > 0 for q in ok)
    print(f"{dataset}/{split}: {len(qs)} queries, {len(qs) - len(ok)} errors, "
          f"count>0 in {hits} ({100 * hits / max(len(ok), 1):.3f}% of answered)", flush=True)


def main():
    global INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["train"])
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--index", default=INDEX, help="Infini-gram index, e.g. v4_olmo-mix-1124_llama")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.3, help="pause after each request, per worker")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="./outputs/leakage")
    ap.add_argument("--chunks", action="store_true", help="query the whole document in consecutive chunks")
    ap.add_argument("--shard", help='only documents of shard "K/N" (K-th of N consecutive parts, 0-based), '
                                    "e.g. 0/2 on one machine and 1/2 on another")
    args = ap.parse_args()
    INDEX = args.index
    print(f"index: {INDEX}", flush=True)
    for ds in args.datasets:
        for split in args.split:
            if _stop.is_set():
                return
            run(ds, split, args.data_root, args.out_root, args.window, args.workers, args.delay, args.chunks,
                args.shard)


if __name__ == "__main__":
    main()
