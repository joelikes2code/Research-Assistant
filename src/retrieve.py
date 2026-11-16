#!/usr/bin/env python3
"""
retrieve.py - hybrid (dense + BM25) retrieval over the chunk index.

Usage (CLI, for eyeballing results):
    python src/retrieve.py "how does Task Shield detect misaligned tool calls?"
                           [-k 5] [--papers P003] [--mode hybrid|dense|bm25]

Library:
    from retrieve import Retriever
    r = Retriever()                       # loads chroma_db/ built by index.py
    hits = r.search("question", k=5, paper_ids=["P003"])
    # -> [{"chunk_id", "paper_id", "section", "page", "text", "score"}, ...]

Hybrid = reciprocal-rank fusion (RRF, k=60) of the dense and BM25 rankings. Dense
handles paraphrase; BM25 catches exact names, acronyms and numbers (benchmark
names, metric names) that small embedders blur. "text" is returned exactly as in
chunks.jsonl so it can go straight into an example's context.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index import COLLECTION, CHUNKER, DEFAULT_DB, META_FILE, load_chunks, sha256  # noqa: E402

# bge v1.5 recommends this prefix on short queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, db: Path = DEFAULT_DB, allow_stale: bool = False):
        import chromadb
        from fastembed import TextEmbedding
        from rank_bm25 import BM25Okapi

        meta_path = Path(db) / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"no index at {db}; run: python src/index.py")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks_path = Path(self.meta["chunks_path"])
        stale = []
        if sha256(chunks_path) != self.meta["chunks_sha256"]:
            stale.append("chunks.jsonl changed")
        if sha256(CHUNKER) != self.meta["chunker_sha256"]:
            stale.append("chunk_papers.py changed")
        if stale and not allow_stale:
            raise RuntimeError(f"index is stale ({'; '.join(stale)}); re-run src/index.py")

        self.col = chromadb.PersistentClient(path=str(db)).get_collection(COLLECTION)
        self.model = TextEmbedding(self.meta["model"])
        self.prefix = BGE_QUERY_PREFIX if "bge" in self.meta["model"].lower() else ""

        self.chunks = load_chunks(chunks_path)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _dense(self, query, n, paper_ids):
        q = next(iter(self.model.embed([self.prefix + query]))).tolist()
        where = {"paper_id": {"$in": list(paper_ids)}} if paper_ids else None
        res = self.col.query(query_embeddings=[q], n_results=n, where=where,
                             include=["distances"])
        return [(cid, 1 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def _bm25(self, query, n, paper_ids):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((c["chunk_id"], s) for c, s in zip(self.chunks, scores)
             if s > 0 and (not paper_ids or c["paper_id"] in paper_ids)),
            key=lambda x: -x[1])
        return ranked[:n]

    def search(self, query: str, k: int = 5, paper_ids=None, mode: str = "hybrid"):
        paper_ids = set(paper_ids) if paper_ids else None
        pool = max(k * 4, 20)                    # candidates per ranker before fusion
        if mode == "dense":
            ranked = self._dense(query, k, paper_ids)
        elif mode == "bm25":
            ranked = self._bm25(query, k, paper_ids)
        elif mode == "hybrid":
            fused = {}
            for ranking in (self._dense(query, pool, paper_ids),
                            self._bm25(query, pool, paper_ids)):
                for rank, (cid, _) in enumerate(ranking):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [{**{f: self.by_id[cid][f] for f in ("chunk_id", "paper_id", "section",
                                                     "page", "text")},
                 "score": round(float(s), 4)} for cid, s in ranked]


def main():
    ap = argparse.ArgumentParser(description="Search the chunk index.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--papers", nargs="*", help="restrict to these paper IDs")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--json", action="store_true", help="print raw JSON lines")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hits = Retriever(args.db).search(args.query, args.k, args.papers, args.mode)
    for h in hits:
        if args.json:
            print(json.dumps(h, ensure_ascii=False))
        else:
            print(f"{h['score']:.4f}  {h['chunk_id']}  p{h['page']}  [{h['section']}]")
            print("        " + h["text"][:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
