"""Per-dataset RedPajama leakage (the paper's measure) from the per-example Infini-gram results.

Input: results/leakage/redpajama/<ds>.jsonl, one row per document (src/leakage/contamination_window.py,
src/leakage/query_normalized.py). A short text (<= 1000 chars) is one query on the whole text; a long text's
row holds its `chunks`: consecutive 1000-char chunks, a partial last chunk replaced by a query on the last
1000 chars. For AYSLAX only the partial last chunks with a match were re-queried on the last 1000 chars,
none of which matched; its other partial last chunks are left out.

Measure: share of queried chunks with a verbatim match (count > 0) in RedPajama; a short text is one chunk.
No token floor; chunks repeated across documents are kept. Queries the API parsed as AND/OR expressions
(`cnf_operator`) or that failed are left out. HoC and DMSAFN were queried detokenized ("rats ." -> "rats.")
and ZSTFNS with tweet URLs removed: their raw text barely matches the web originals.

    python src/leakage/summarize_leakage.py [--dir results/leakage/redpajama] [--out results/leakage/redpajama_summary.csv]
"""
import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["hallmarks_of_cancer", "N2C2_2008", "PsyTar", "DMSAFN", "Lc37Fnd", "ZSTFNS", "AYSLAX", "EURLEX"]


def ok(q):
    return q.get("count", -1) >= 0 and not q.get("cnf_operator") and not q.get("error")


def chunks(d, ds):
    """List of per-document lists of usable chunk queries."""
    rows = map(json.loads, open(os.path.join(d, f"{ds}.jsonl"), encoding="utf-8"))
    return [[q for q in r.get("chunks", [r]) if ok(q)] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "leakage", "redpajama"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "leakage", "redpajama_summary.csv"))
    args = ap.parse_args()
    rows = []
    for ds in DATASETS:
        docs = [c for c in chunks(args.dir, ds) if c]
        n = sum(len(c) for c in docs)
        hit = sum(q["count"] > 0 for c in docs for q in c)
        rows.append({"dataset": ds, "documents": len(docs), "chunks": n, "leaked_chunks": hit,
                     "leakage_pct": round(100 * hit / n, 4)})
        print(rows[-1])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
