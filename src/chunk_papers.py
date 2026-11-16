#!/usr/bin/env python3
"""
chunk_papers.py - turn research-paper PDFs into section-aware chunks with stable IDs.

FREEZE THIS FILE before generating any training or test data. Every gold citation
refers to the chunk IDs this script produces; changing it later invalidates them.

Usage:
    python chunk_papers.py PDF_DIR OUT_DIR [--target 400] [--max 512] [--min 80]

Token counts are exact, from the embedder's tokenizer (TOKENIZER below), and include
the "[Section: ...]" prefix and [CLS]/[SEP]. --max may not exceed EMBED_WINDOW (512).

Rename PDFs to short alphanumeric IDs first (P001.pdf, P002.pdf, ...) and keep a
papers.csv mapping each ID to title, venue, year, licence and URL.

Outputs (in OUT_DIR):
    chunks.jsonl  one chunk per line: chunk_id, paper_id, section, page, text, n_tokens
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
v0.7: words are built from rawdict characters (page_words) and also split where the gap
      between two characters in the same span exceeds 0.10 em. P001 has no space
      characters on many lines ("Buildingonthethreatmodelingframeworks"); its gaps are
      0.12-0.15 em vs <= 0.06 em inside words. Verified: non-space text identical for
      P001-P003; P002 unchanged; P003 only "∈{Ls," -> "∈ {Ls,". P001 now 56 chunks
      (correct word counts split section 2.2 once more), so P001_c10+ IDs shifted by one.
      Not fixed: line-break de-hyphenation also joins real compounds
      ("environment-\\nmodifying" -> "environmentmodifying").
v0.8: layouts of P004-P008 (USENIX, DeepMind, ICML, IEEE). Verified P001-P003 output
      byte-identical to v0.7.
  - arXiv margin stamps ("arXiv:2503.18813v2 [cs.CR] 24 Jun 2025") are dropped.
  - body font size is estimated from the first 10 pages (P005's 100+ appendix pages of
    small code gave 6.6pt, so 8pt figure text became "headings").
  - headings smaller than 0.8 x body are rejected (bold labels inside figures, P004).
  - IEEE headings at body size are accepted on one line: "IV. DEFENSES" (Roman, caps),
    italic "A. Sessions", and all-caps known names ("REFERENCES").
  - unlabelled abstract: if none is found, the longest page-1 front-matter paragraph
    (>= 100 words) becomes "Abstract" and papers.jsonl gets a warning (P005).
  Not fixed: papers.jsonl "title" is the largest page-1 text and is wrong for P005
  ("1. Introduction"); papers.csv is the source of truth for titles.
v0.9: exact token counts. est_tokens (words x 1.3) undercounted tables, math and code,
      so 12 chunks exceeded the embedder's 512-token window and were truncated. Now
      n_tokens() uses the bge-small-en-v1.5 tokenizer and measures the chunk WITH its
      "[Section: ...]" prefix; --max defaults to (and is capped at) 512; sentences longer
      than --target (table text) are split on words; the tiny-tail merge only happens if
      the result fits. chunks.jsonl field est_tokens -> n_tokens. Chunk IDs of all papers
      change, since boundaries move.
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
SKIP_SECTIONS = {"acknowledgments", "acknowledgements", "acknowledgment",
                 "acknowledgement", "acknowlegement"}       # last: typo in P007
WORD_GAP_EM = 0.10     # letter gaps in P001-P003 are <= 0.06 em; missing spaces are 0.12-0.15 em
NUM_PREFIX = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.\d*)\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")
KNOWN = {"abstract", "introduction", "conclusion", "conclusions", "limitations",
         "ethics statement", "ethical considerations", "related work", "background",
         "discussion", "references", "bibliography", "appendix"} | SKIP_SECTIONS
ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]\s+\d{1,2} \w{3} \d{4}$")
ROMAN_HEAD = re.compile(r"^[IVX]+\.\s+[A-Z][A-Z0-9 ,:&/-]+$")       # IEEE "IV. DEFENSES"
LETTER_HEAD = re.compile(r"^[A-H]\.\s+[A-Z][^.]{2,80}$")           # IEEE italic "A. Sessions"
BODY_PAGES = 10        # estimate body font size from the first pages only (long appendices skew it)


TOKENIZER = "BAAI/bge-small-en-v1.5"    # must match the embedding model in index.py
EMBED_WINDOW = 512                      # its max sequence length, special tokens included
_tokenizer = None


def n_tokens(text: str) -> int:
    """Exact token count under the embedder's tokenizer, incl. [CLS] and [SEP]."""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_pretrained(TOKENIZER)   # cached after first download
        _tokenizer.no_truncation()
        _tokenizer.no_padding()
    return len(_tokenizer.encode(text).ids)


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


def page_words(page, gap_em=WORD_GAP_EM):
    """Like page.get_text("words") (same tuples), but also breaks a word wherever the
    horizontal gap between two characters exceeds gap_em * font size. Some PDFs (P001)
    encode inter-word spaces as positioning only, with no space character, and MuPDF's
    words extractor then glues the whole line together."""
    out = []
    for bn, b in enumerate(page.get_text("rawdict")["blocks"]):
        for ln, line in enumerate(b.get("lines", [])):
            cur, prev, wn = [], None, 0

            def flush():
                nonlocal cur, wn
                if cur:
                    out.append((min(c["bbox"][0] for c in cur), min(c["bbox"][1] for c in cur),
                                max(c["bbox"][2] for c in cur), max(c["bbox"][3] for c in cur),
                                "".join(c["c"] for c in cur), bn, ln, wn))
                    wn += 1
                    cur = []

            for s in line["spans"]:
                prev = None                    # gap rule only within one span: math glyphs
                for c in s["chars"]:           # come from other fonts with wide side bearings
                    if c["c"].isspace():
                        flush()
                        prev = None
                        continue
                    if prev is not None and c["bbox"][0] - prev["bbox"][2] > gap_em * s["size"]:
                        flush()
                    cur.append(c)
                    prev = c
            flush()
    return out


def build_blocks(doc):
    blocks = []
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        words = page_words(page)         # (x0, y0, x1, y1, word, block, line, wordno)
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
            if ARXIV_STAMP.match(text):                             # arXiv margin stamp
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
                "italic": all(s["flags"] & 2 for s in spans),
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
    if b["size"] < 0.8 * body:              # bold labels inside figures (P004 "5. Response")
        return False
    if not (b["bold"] or b["size"] >= body + 0.5):
        # IEEE style: small-caps or italic headings at body size, one line
        return b["nlines"] == 1 and bool(
            ROMAN_HEAD.match(t)
            or (b.get("italic") and LETTER_HEAD.match(t))
            or (t.isupper() and heading_key(t) in KNOWN))
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


def split_words(text, target, measure):
    """Last resort for a 'sentence' over target (table text, no periods): split on words."""
    out, cur = [], []
    for w in text.split():
        if cur and measure(" ".join(cur + [w])) > target:
            out.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def split_long(paragraph, target, measure):
    """Split an over-long paragraph on sentence boundaries into <= target pieces."""
    pieces, cur = [], ""
    for sent in SENT_SPLIT.split(paragraph):
        parts = split_words(sent, target, measure) if measure(sent) > target else [sent]
        for part in parts:
            if cur and measure(cur + " " + part) > target:
                pieces.append(cur)
                cur = part
            else:
                cur = (cur + " " + part).strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_section(paras, target, hard_max, min_tokens, prefix=""):
    """Pack paragraphs into chunks. Sizes are measured WITH the "[Section: ...] " prefix,
    because that is the text the embedder sees; every chunk is <= hard_max tokens."""
    measure = lambda t: n_tokens(prefix + t)
    units = []
    for p in paras:
        units += split_long(p, target, measure) if measure(p) > hard_max else [p]
    chunks, cur = [], []
    for u in units:
        if cur and measure(" ".join(cur + [u])) > target:
            chunks.append(" ".join(cur))
            cur = [u]
        else:
            cur.append(u)
    if cur:
        chunks.append(" ".join(cur))
    if (len(chunks) > 1 and measure(chunks[-1]) < min_tokens          # merge tiny tail
            and measure(chunks[-2] + " " + chunks[-1]) <= hard_max):
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


def process_pdf(path, target, hard_max, min_tokens):
    doc = pymupdf.open(path)
    blocks = drop_running_text(build_blocks(doc), len(doc))
    body = body_size([b for b in blocks if b["page"] <= BODY_PAGES])
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
    # unlabelled abstract (P005, Google DeepMind template): longest page-1 front-matter
    # paragraph of >= 100 words. Reported in papers.jsonl warnings: check it by hand.
    inferred_abstract = False
    if not any(s["name"].lower() == "abstract" for s in sections):
        cands = [p for p in sections[0]["paras"] if len(p.split()) >= 100]
        if cands:
            sections.insert(1, {"name": "Abstract", "page": 1,
                                "paras": [max(cands, key=lambda p: len(p.split()))]})
            inferred_abstract = True

    chunks = []
    for sec in sections:
        key = heading_key(sec["name"])
        if sec["name"] == "Front matter" or key in SKIP_SECTIONS or is_stop(key):
            continue
        prefix = f"[Section: {sec['name']}] "
        for text in chunk_section(sec["paras"], target, hard_max, min_tokens, prefix):
            if n_tokens(text) < 15:
                continue
            chunks.append({"section": sec["name"], "page": sec["page"],
                           "text": prefix + text})
    notes = ["abstract inferred (no 'Abstract' label): check it"] if inferred_abstract else []
    return title, len(doc), chunks, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--max", type=int, default=EMBED_WINDOW)
    ap.add_argument("--min", type=int, default=80)
    args = ap.parse_args()
    if args.max > EMBED_WINDOW:
        sys.exit(f"--max {args.max} > {EMBED_WINDOW}: the embedder would truncate chunks.")

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
            title, n_pages, chunks, warnings = process_pdf(pdf, args.target, args.max, args.min)
            if len(chunks) < 8:
                warnings.append("few chunks: check extraction")
            for i, c in enumerate(chunks):
                rec = {"chunk_id": f"{pid}_c{i:02d}", "paper_id": pid,
                       "section": c["section"], "page": c["page"], "text": c["text"],
                       "n_tokens": n_tokens(c["text"])}
                if rec["n_tokens"] > args.max:
                    warnings.append(f"{rec['chunk_id']} has {rec['n_tokens']} tokens > {args.max}")
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.write(json.dumps({"paper_id": pid, "title": title, "n_pages": n_pages,
                                 "n_chunks": len(chunks), "warnings": warnings},
                                ensure_ascii=False) + "\n")
            print(f"{pid}: {len(chunks)} chunks  {'; '.join(warnings)}")


if __name__ == "__main__":
    main()
