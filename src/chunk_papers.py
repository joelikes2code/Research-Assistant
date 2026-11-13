#!/usr/bin/env python3
"""
chunk_papers.py - turn research-paper PDFs into section-aware chunks with stable IDs.

FREEZE THIS FILE before generating any training or test data. Every gold citation
refers to the chunk IDs this script produces; changing it later invalidates them.

Usage:
    python chunk_papers.py PDF_DIR OUT_DIR [--target 400] [--max 550] [--min 80]

Rename PDFs to short alphanumeric IDs first (P001.pdf, P002.pdf, ...) and keep a
papers.csv mapping each ID to title, venue, year, licence and URL.

Outputs (in OUT_DIR):
    chunks.jsonl  one chunk per line: chunk_id, paper_id, section, page, text, est_tokens
    papers.jsonl  one paper per line: paper_id, title, n_pages, n_chunks, warnings

Known limits (report these): tables and equations extract badly, and two-column
layouts can occasionally interleave. Heading detection is a heuristic (numbered or
known section names that are also bold/larger). Always eyeball the output for a
sample of papers.

Change log (v0.2, NOT yet validated on ACL two-column PDFs):
  - text comes from the "words" extractor (correct spacing); font size/bold from "dict",
    matched to words by bounding box (block numbers differ between the two modes).
  - headings must be numbered or a known name, and bold or larger than body text.
  - small-font blocks (footnotes) are dropped after the front matter.
  - short text repeated across many pages (running headers) is dropped.
  - paragraphs split by a page/column break are re-joined.
  - run-in "Abstract ..." text is split off the front matter as its own section.
v0.3: captions/tables kept; only page-1 footnotes dropped.
v0.4: scope is MAIN TEXT ONLY (Abstract up to the first References / Appendix /
      Declarations heading). ACL appendices come after References, so this keeps
      all venues consistent. Numbered headings may wrap to 3 lines.
v0.5: run-in numbered headings (bold "5.1.1 Title" followed by body text in the
      same block, common in ACL PDFs) are split into heading + body.
v0.6: v0.5 never fired: the leading bold spans were joined with "" ("5.1.1Structure
      Identifier"), so the whitespace the heading regex requires was missing. Now the heading word count comes
      from the words extractor by bounding box (spans are separate words). Verified on P002:
      5.1.1, 5.1.2, 5.2.1-5.2.4, 7.1.1 now appear; P001 unchanged at 55 chunks.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

STOP_SECTIONS = {"references", "bibliography"}          # stop processing here
SKIP_SECTIONS = {"acknowledgments", "acknowledgements", "acknowledgment"}
NUM_PREFIX = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.\d*)\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")
KNOWN = {"abstract", "introduction", "conclusion", "conclusions", "limitations",
         "ethics statement", "ethical considerations", "related work", "background",
         "discussion", "acknowledgments", "acknowledgements", "references",
         "bibliography", "appendix"}


def est_tokens(text: str) -> int:
    """Cheap token estimate (words * 1.3). Replace with a real tokenizer if wanted."""
    return int(len(text.split()) * 1.3) + 1


def join_lines(lines):
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln          # de-hyphenate line-break words
        else:
            out += " " + ln
    return out.strip()


def build_blocks(doc):
    blocks = []
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        words = page.get_text("words")   # (x0, y0, x1, y1, word, block, line, wordno)
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            spans = [s for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            bx0, by0, bx1, by1 = b["bbox"]
            lines = {}
            for x0, y0, x1, y1, w, bn, ln, wn in words:
                if bx0 <= (x0 + x1) / 2 <= bx1 and by0 <= (y0 + y1) / 2 <= by1:
                    lines.setdefault((bn, ln), []).append(w)
            if not lines:
                continue
            text = unicodedata.normalize(
                "NFKC", join_lines([" ".join(ws) for ws in lines.values()]))
            if re.fullmatch(r"\d{1,3}", text):                      # page number
                continue
            if by0 > 0.95 * height or by1 < 0.04 * height:          # header/footer
                continue
            is_b = lambda s: bool(s["flags"] & 16) or "bold" in s["font"].lower()
            k = 0                                   # number of leading bold spans
            while k < len(spans) and is_b(spans[k]):
                k += 1
            # leading bold words, counted by bounding box (spans are separate words)
            lead_n = 0
            if 0 < k < len(spans):
                for x0, y0, x1, y1, w, bn, ln, wn in words:
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if any(s["bbox"][0] <= cx <= s["bbox"][2] and s["bbox"][1] <= cy <= s["bbox"][3]
                           for s in spans[:k]):
                        lead_n += 1
            toks = text.split()
            lead_text = " ".join(toks[:lead_n])
            if (0 < k < len(spans) and 0 < lead_n < len(toks) and len(lead_text) <= 100
                    and re.match(r"\d+(\.\d+)+\s+[A-Za-z]", lead_text)):
                # run-in numbered heading ("5.1.1 Title Body text ...") -> split it
                n = lead_n
                if len(toks) > n:
                    blocks.append({"page": pno, "text": " ".join(toks[:n]),
                                   "size": max(s["size"] for s in spans[:k]),
                                   "bold": True, "nlines": 1})
                    blocks.append({"page": pno, "text": " ".join(toks[n:]),
                                   "size": max(s["size"] for s in spans[k:]),
                                   "bold": False, "nlines": len(lines)})
                    continue
            blocks.append({
                "page": pno, "text": text,
                "size": max(s["size"] for s in spans),
                "bold": all(is_b(s) for s in spans),
                "nlines": len(lines)})
    return blocks


def body_size(blocks):
    weights = Counter()
    for b in blocks:
        weights[round(b["size"], 1)] += len(b["text"])
    return weights.most_common(1)[0][0] if weights else 10.0


def heading_key(t):
    return NUM_PREFIX.sub("", t).strip().lower().rstrip(":")


def is_stop(key):
    """Main-text boundary: references, appendices, declarations."""
    return key in STOP_SECTIONS or key.startswith(("appendix", "declarations"))


def is_heading(b, body):
    t = b["text"]
    if b["nlines"] > 3 or not 3 <= len(t) <= 120:
        return False
    if not (b["bold"] or b["size"] >= body + 0.5):
        return False
    m = NUM_PREFIX.match(t)
    if m:                                   # numbered headings may wrap to 3 lines
        rest = t[m.end():]
        return bool(rest) and rest[0].isalpha() and not t.endswith(".")
    if b["nlines"] > 2:
        return False
    key = heading_key(t)
    return key in KNOWN or is_stop(key)


def drop_running_text(blocks, n_pages):
    """Remove short blocks that repeat on many pages (running headers/footers)."""
    key = lambda t: re.sub(r"\d+", "#", t.lower())
    pages = {}
    for b in blocks:
        pages.setdefault(key(b["text"]), set()).add(b["page"])
    cut = max(3, int(0.3 * n_pages))
    return [b for b in blocks if len(b["text"]) > 150 or len(pages[key(b["text"])]) < cut]


def split_long(paragraph, target):
    """Split an over-long paragraph on sentence boundaries into <= target pieces."""
    pieces, cur = [], ""
    for sent in SENT_SPLIT.split(paragraph):
        if cur and est_tokens(cur + " " + sent) > target:
            pieces.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_section(paras, target, hard_max, min_tokens):
    units = []
    for p in paras:
        units += split_long(p, target) if est_tokens(p) > hard_max else [p]
    chunks, cur = [], []
    for u in units:
        if cur and est_tokens(" ".join(cur + [u])) > target:
            chunks.append(" ".join(cur))
            cur = [u]
        else:
            cur.append(u)
    if cur:
        chunks.append(" ".join(cur))
    if len(chunks) > 1 and est_tokens(chunks[-1]) < min_tokens:   # merge tiny tail
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


def process_pdf(path, target, hard_max, min_tokens):
    doc = pymupdf.open(path)
    blocks = drop_running_text(build_blocks(doc), len(doc))
    body = body_size(blocks)
    first_page = [b for b in blocks if b["page"] == 1]
    title = max(first_page, key=lambda b: b["size"])["text"] if first_page else path.stem

    sections, current = [], {"name": "Front matter", "page": 1, "paras": []}
    for b in blocks:
        if is_heading(b, body):
            sections.append(current)
            current = {"name": b["text"], "page": b["page"], "paras": []}
            if is_stop(heading_key(b["text"])):
                break
            continue
        in_front = current["name"] == "Front matter"
        small = b["size"] < body - 1.0
        if small and b["page"] == 1 and not in_front:
            continue                                   # page-1 affiliation footnotes only
        paras = current["paras"]
        if (not in_front and not small and paras and paras[-1][-1:] not in ".!?:"
                and b["text"][:1].islower()):
            paras[-1] = join_lines([paras[-1], b["text"]])   # paragraph split by page/column
        else:
            paras.append(b["text"])
    sections.append(current)

    # run-in abstract ("Abstract Agentic AI ...") sitting in the front matter
    if not any(s["name"].lower() == "abstract" for s in sections):
        for p in sections[0]["paras"]:
            if p.lower().startswith("abstract"):
                text = re.sub(r"^abstract[\s:.\u2014-]*", "", p, flags=re.I)
                sections.insert(1, {"name": "Abstract", "page": 1, "paras": [text]})
                break

    chunks = []
    for sec in sections:
        key = heading_key(sec["name"])
        if sec["name"] == "Front matter" or key in SKIP_SECTIONS or is_stop(key):
            continue
        for text in chunk_section(sec["paras"], target, hard_max, min_tokens):
            if est_tokens(text) < 15:
                continue
            chunks.append({"section": sec["name"], "page": sec["page"],
                           "text": f"[Section: {sec['name']}] {text}"})
    return title, len(doc), chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--max", type=int, default=550)
    ap.add_argument("--min", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs found.")

    with open(out / "chunks.jsonl", "w", encoding="utf-8") as cf, \
         open(out / "papers.jsonl", "w", encoding="utf-8") as pf:
        for pdf in pdfs:
            pid = pdf.stem
            if not re.fullmatch(r"[A-Za-z0-9]+", pid):
                sys.exit(f"Bad paper ID '{pid}': use letters/digits only (e.g. P001).")
            title, n_pages, chunks = process_pdf(pdf, args.target, args.max, args.min)
            warnings = []
            if len(chunks) < 8:
                warnings.append("few chunks: check extraction")
            for i, c in enumerate(chunks):
                rec = {"chunk_id": f"{pid}_c{i:02d}", "paper_id": pid,
                       "section": c["section"], "page": c["page"], "text": c["text"],
                       "est_tokens": est_tokens(c["text"])}
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.write(json.dumps({"paper_id": pid, "title": title, "n_pages": n_pages,
                                 "n_chunks": len(chunks), "warnings": warnings},
                                ensure_ascii=False) + "\n")
            print(f"{pid}: {len(chunks)} chunks  {'; '.join(warnings)}")


if __name__ == "__main__":
    main()
