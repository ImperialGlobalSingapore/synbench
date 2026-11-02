"""Whole-document Infini-gram counts of normalized short texts (HoC, DMSAFN detokenized; ZSTFNS URL-free).

Some datasets are stored in a form that never occurs verbatim on the web, so a verbatim query of the raw
text undercounts leakage:
  detok     pre-tokenized text ("lean rats .", "company 's", ``quotes''): HoC, DMSAFN. Spaces before
            punctuation / inside brackets / before clitics are removed and PTB quotes restored.
  strip-url tweets ending in a unique t.co link: ZSTFNS. URLs are removed; texts without a URL keep their
            raw-text result (--raw), and texts that are only a URL are flagged, not queried (an empty query
            counts every token of the index).
Output rows match contamination_window.py's whole-document rows (`_idx`, `count`, `cnf_operator`, `error`,
`ntok`, plus the `query` sent). Resumable; on HTTP 403/429 (Infini-gram rate limit, per IP) it waits 10 min.
Keep to one process per IP at --delay >= 0.5 s.

    python src/leakage/query_normalized.py --mode detok --input data/hallmarks_of_cancer/original/train-original.jsonl \
        --output results/leakage/redpajama/hallmarks_of_cancer.jsonl
    python src/leakage/query_normalized.py --mode strip-url --input data/ZSTFNS/original/train-original.jsonl \
        --raw <raw whole-document results> --output results/leakage/redpajama/ZSTFNS.jsonl
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

API = "https://api.infini-gram.io/"


def detok(t):
    t = re.sub(r"``\s*", '"', t)
    t = re.sub(r"\s*''", '"', t)
    t = re.sub(r"\s+([.,;:!?%)\]])", r"\1", t)  # "rats ." -> "rats."
    t = re.sub(r"([(\[])\s+", r"\1", t)  # "( T" -> "(T"
    t = re.sub(r"\s+('s|'re|'ll|'d|'ve|'m|n't)\b", r"\1", t)
    t = re.sub(r"\$\s+(\d)", r"$\1", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_url(t):
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", " ", t)).strip()


def count(index, q):
    req = urllib.request.Request(API, headers={"Content-Type": "application/json"},
                                 data=json.dumps({"index": index, "query_type": "count", "query": q}).encode())
    while True:
        try:
            b = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return (b.get("count", -1), str(b["error"]) if b.get("error") else None, bool(b.get("approx", False)),
                    len(b["token_ids"]) if "token_ids" in b else None)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print(time.strftime("%H:%M:%S"), f"HTTP {e.code}, waiting 600 s", flush=True)
                time.sleep(600)
                continue
            return -1, f"HTTP {e.code}", False, None
        except Exception as e:  # timeout / connection reset
            print(time.strftime("%H:%M:%S"), "request failed:", e, flush=True)
            time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detok", "strip-url"], required=True)
    ap.add_argument("--input", required=True, help="train-original.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw", help="strip-url: raw-text whole-document results reused for texts without a URL")
    ap.add_argument("--index", default="v4_rpj_llama_s4", help="Infini-gram index (RedPajama; v4_olmo-mix-1124_llama)")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    raw = {r["_idx"]: r for r in map(json.loads, open(args.raw))} if args.raw else {}
    done = set()
    if os.path.exists(args.output):
        done = {r["_idx"] for r in map(json.loads, open(args.output, encoding="utf-8"))
                if not (r["error"] or "").startswith("HTTP")}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            if i in done:
                continue
            t = r["text"]
            rec = {"_idx": i, "chars": len(t), "query_chars": None, "query": None, "cnf_operator": False,
                   "ntok": None, "count": -1, "approx": False, "error": None, "source": "query"}
            if args.mode == "strip-url" and not re.search(r"https?://", t):
                o = raw[i]
                rec.update(query_chars=len(t), query=t, cnf_operator=o["cnf_operator"], ntok=o["ntok"],
                           count=o["count"], approx=o.get("approx", False), error=o.get("error"), source="raw")
            else:
                q = detok(t) if args.mode == "detok" else strip_url(t)
                rec.update(query_chars=len(q), query=q, cnf_operator=" AND " in f" {q} " or " OR " in f" {q} ")
                if not q:
                    rec["error"] = "empty after URL removal"
                elif rec["cnf_operator"]:  # the API parses upper-case AND / OR as boolean operators
                    rec["error"] = "cnf_operator"
                else:
                    rec["count"], rec["error"], rec["approx"], rec["ntok"] = count(args.index, q)
                    time.sleep(args.delay)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
