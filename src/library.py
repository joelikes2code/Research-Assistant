#!/usr/bin/env python3
"""
library.py - papers the user adds (from the app or the command line).

    python src/library.py add path/to/paper.pdf
    python src/library.py list
    python src/library.py remove U001

The experiment's index (chroma_db/, papers P001-P008) is never changed, so evaluation
runs stay comparable with the committed results. Added papers get IDs U001, U002, ...
and live in data/user/ (gitignored: they may be copyrighted):

    data/user/pdfs/U001.pdf
    data/user/papers.jsonl     one line per added paper (title, file, sha256, warnings)
    data/user/chunks.jsonl     chunks of the added papers only
    data/user/library.jsonl    P-paper chunks + added chunks (what the library index holds)

and in a second index, chroma_library/, built from library.jsonl. Adding a paper embeds
only its own chunks. The library index is rebuilt in full only when the P-paper chunks
or the chunker change (index_meta.json records both).

Chunks are made by the frozen chunker (chunk_papers.process_pdf, same settings as the
experiment). Its heading rules were tuned on 8 layouts, so check the preview of a new
paper's first chunks; a badly read paper can be removed again.
"""
import argparse
import gc
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index import (CHUNKER, COLLECTION, DEFAULT_CHUNKS, DEFAULT_MODEL,  # noqa: E402
                   META_FILE, build_index, embed_and_add, load_chunks, sha256, write_meta)

ROOT = Path(__file__).resolve().parent.parent
USER_DIR = ROOT / "data" / "user"
LIBRARY_DB = ROOT / "chroma_library"


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] \
        if Path(path).exists() else []


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


class Library:
    def __init__(self, user_dir=USER_DIR, db=LIBRARY_DB, core_chunks=DEFAULT_CHUNKS,
                 model=DEFAULT_MODEL):
        self.dir, self.db, self.core, self.model = Path(user_dir), Path(db), Path(core_chunks), model
        self.pdfs = self.dir / "pdfs"
        self.papers_path = self.dir / "papers.jsonl"
        self.chunks_path = self.dir / "chunks.jsonl"
        self.library_path = self.dir / "library.jsonl"

    def papers(self):
        return read_jsonl(self.papers_path)

    def _meta(self):
        p = self.db / META_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def ensure_index(self):
        """Make chroma_library/ match P-paper chunks + added chunks; rebuild if not."""
        m = self.dir.exists() and self._meta()
        if (m and self.library_path.exists() and m.get("core_sha256") == sha256(self.core)
                and m.get("chunker_sha256") == sha256(CHUNKER)
                and m.get("chunks_sha256") == sha256(self.library_path)):
            return self.db
        self.dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.library_path, load_chunks(self.core) + read_jsonl(self.chunks_path))
        build_index(self.library_path, self.db, self.model, core_sha256=sha256(self.core))
        return self.db

    def _collection(self):
        import chromadb
        return chromadb.PersistentClient(path=str(self.db)).get_collection(COLLECTION)

    def _save(self, papers, user_chunks):
        """Write the user files and refresh the fingerprint; the collection is updated
        by the caller."""
        write_jsonl(self.papers_path, papers)
        write_jsonl(self.chunks_path, user_chunks)
        lib = load_chunks(self.core) + user_chunks
        write_jsonl(self.library_path, lib)
        write_meta(self.db, self.library_path, self.model, len(lib),
                   core_sha256=sha256(self.core))

    def add(self, data: bytes, filename: str) -> dict:
        """Chunk and index one PDF. Returns its paper record plus "chunks" and
        "duplicate" (True if the same file was already added)."""
        import pymupdf
        from chunk_papers import EMBED_WINDOW, n_tokens, process_pdf
        from fastembed import TextEmbedding

        digest = hashlib.sha256(data).hexdigest()
        papers = self.papers()
        for p in papers:
            if p["sha256"] == digest:
                chunks = [c for c in read_jsonl(self.chunks_path) if c["paper_id"] == p["paper_id"]]
                return {**p, "chunks": chunks, "duplicate": True}

        try:                                     # check in memory: nothing written yet
            with pymupdf.open(stream=data, filetype="pdf"):
                pass
        except Exception as e:
            raise ValueError(f"could not read {filename} as a PDF ({e})") from e

        self.ensure_index()
        used = [int(p["paper_id"][1:]) for p in papers]
        pid = f"U{max(used, default=0) + 1:03d}"
        self.pdfs.mkdir(parents=True, exist_ok=True)
        pdf = self.pdfs / f"{pid}.pdf"
        pdf.write_bytes(data)
        # process_pdf (frozen) never closes its document; on Windows an open file cannot
        # be deleted, so collect it before any cleanup
        try:
            title, n_pages, raw, warnings = process_pdf(pdf, 400, EMBED_WINDOW, 80)
        except Exception as e:
            _delete(pdf)
            raise ValueError(f"could not extract text from {filename} ({e})") from e
        if not raw:
            _delete(pdf)
            raise ValueError(f"no text found in {filename}: it may be a scanned PDF "
                             "(images only), which this tool cannot read")
        if len(raw) < 8:
            warnings.append("few chunks: check extraction")
        chunks = [{"chunk_id": f"{pid}_c{i:02d}", "paper_id": pid, "section": c["section"],
                   "page": c["page"], "text": c["text"], "n_tokens": n_tokens(c["text"])}
                  for i, c in enumerate(raw)]

        embed_and_add(self._collection(), chunks, TextEmbedding(self.model))
        rec = {"paper_id": pid, "title": (title or "").strip() or Path(filename).stem,
               "filename": filename, "sha256": digest, "n_pages": n_pages,
               "n_chunks": len(chunks), "warnings": warnings, "added": date.today().isoformat()}
        self._save(papers + [rec], read_jsonl(self.chunks_path) + chunks)
        return {**rec, "chunks": chunks, "duplicate": False}

    def remove(self, pid: str):
        papers = self.papers()
        if pid not in {p["paper_id"] for p in papers}:
            raise KeyError(f"{pid} is not an added paper")
        self.ensure_index()
        self._collection().delete(where={"paper_id": pid})
        self._save([p for p in papers if p["paper_id"] != pid],
                   [c for c in read_jsonl(self.chunks_path) if c["paper_id"] != pid])
        _delete(self.pdfs / f"{pid}.pdf")


def _delete(path):
    gc.collect()                                  # releases PDFs process_pdf left open
    Path(path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("pdf", type=Path)
    sub.add_parser("list")
    sub.add_parser("remove").add_argument("paper_id")
    args = ap.parse_args()
    lib = Library()
    if args.cmd == "add":
        rec = lib.add(args.pdf.read_bytes(), args.pdf.name)
        print(f"{'already added' if rec['duplicate'] else 'added'}: {rec['paper_id']} "
              f"{rec['title']!r}, {rec['n_chunks']} chunks"
              + (f"  warnings: {'; '.join(rec['warnings'])}" if rec["warnings"] else ""))
    elif args.cmd == "list":
        for p in lib.papers():
            print(f"{p['paper_id']}  {p['n_chunks']:3d} chunks  {p['title']}")
    else:
        lib.remove(args.paper_id)
        print(f"removed {args.paper_id}")


if __name__ == "__main__":
    main()
