#!/usr/bin/env python3
"""
index.py - build the retrieval index over chunks.jsonl.

Usage:
    python src/index.py [--chunks data/processed/chunks.jsonl] [--db chroma_db]
                        [--model BAAI/bge-small-en-v1.5]

Builds two indexes over the SAME chunk IDs produced by chunk_papers.py:
    dense   Chroma collection (cosine), embeddings from fastembed (ONNX, no torch)
    sparse  BM25 is cheap at this size, so retrieve.py rebuilds it from chunks.jsonl
            at load time instead of persisting it.

The collection is rebuilt from scratch on every run (no stale chunks survive a
re-chunk). index_meta.json records the embedding model and SHA-256 of chunks.jsonl
and chunk_papers.py so retrieve.py can refuse to run against a stale index.

chroma_db/ is gitignored: it contains text from papers that must not be released
(see papers.csv notes, e.g. P001).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_DB = ROOT / "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"      # 512-token window; chunks target ~400
COLLECTION = "chunks"
META_FILE = "index_meta.json"
CHUNKER = ROOT / "src" / "chunk_papers.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path: Path) -> list[dict]:
    chunks = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [c["chunk_id"] for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"duplicate chunk_ids in {path}: {sorted(dupes)[:5]}")
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import chromadb
    from fastembed import TextEmbedding

    chunks = load_chunks(args.chunks)
    print(f"{len(chunks)} chunks from {args.chunks}")

    model = TextEmbedding(args.model)
    texts = [c["text"] for c in chunks]
    vecs = [v.tolist() for v in model.embed(texts, batch_size=args.batch)]

    # Safety net: chunker v0.9+ caps chunks at 512 tokens of this tokenizer, so this should
    # only fire if --model differs from chunk_papers.TOKENIZER.
    tok = model.model.tokenizer
    max_len = tok.truncation["max_length"] if tok.truncation else None
    if max_len:
        tok.no_truncation()
        long = [(c["chunk_id"], n) for c in chunks
                if (n := len(tok.encode(c["text"]).ids)) > max_len]
        tok.enable_truncation(max_len)
        for cid, n in long:
            print(f"  warning: {cid} has {n} tokens > {max_len}; embedding sees only the start")

    args.db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass                                      # first build: nothing to delete
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                   embedding_function=None)
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vecs,
        documents=texts,
        metadatas=[{"paper_id": c["paper_id"], "section": c["section"], "page": c["page"]}
                   for c in chunks],
    )

    meta = {
        "model": args.model,
        "n_chunks": len(chunks),
        "chunks_path": str(args.chunks.resolve()),
        "chunks_sha256": sha256(args.chunks),
        "chunker_sha256": sha256(CHUNKER),
    }
    (args.db / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    papers = sorted({c["paper_id"] for c in chunks})
    print(f"indexed {col.count()} chunks from {len(papers)} papers ({', '.join(papers)}) "
          f"into {args.db} with {args.model}")


if __name__ == "__main__":
    main()
