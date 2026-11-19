"""
Research assistant UI: ask a question about the indexed papers and get an answer whose
every sentence cites a passage, with the passages shown for checking.

    uv run streamlit run app/streamlit_app.py

Needs the index (src/index.py) and Ollama running. Uses the same generate.answer() as
the evaluation, so what you see here is what was measured.

Papers added in the sidebar (src/library.py) go to data/user/ and a separate library
index, chroma_library/; the experiment's index chroma_db/ is never changed. With no
added papers the app searches chroma_db/ directly.
"""
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import generate  # noqa: E402
from library import Library  # noqa: E402
from validate_examples import ABSTAIN, CITE_RE  # noqa: E402

st.set_page_config(page_title="Research Assistant", layout="wide")
library = Library()


@st.cache_resource(show_spinner="Loading the search index...")
def get_retriever(db: str):
    from retrieve import Retriever
    return Retriever(db=Path(db))


def current_retriever():
    if library.papers():
        with st.spinner("Updating the library index..."):
            return get_retriever(str(library.ensure_index()))
    from index import DEFAULT_DB
    return get_retriever(str(DEFAULT_DB))


@st.cache_data
def load_core_papers():
    with open(ROOT / "data" / "papers.csv", encoding="utf-8", newline="") as f:
        return {r["paper_id"]: r for r in csv.DictReader(f)}


def load_papers():
    return {**load_core_papers(), **{p["paper_id"]: p for p in library.papers()}}


def remove_paper(pid):
    """Button callback: runs before the widgets are drawn, so the paper selection can
    still be changed (a selected paper that no longer exists would break it)."""
    library.remove(pid)
    st.session_state["chosen"] = [p for p in st.session_state.get("chosen", []) if p != pid]
    for key in ("added", "rec", "add_error"):
        st.session_state.pop(key, None)
    get_retriever.clear()


@st.cache_data(ttl=60)
def ollama_models():
    try:
        with urllib.request.urlopen(f"{generate.OLLAMA}/api/tags", timeout=5) as r:
            return sorted(m["name"] for m in json.loads(r.read())["models"])
    except OSError:
        return []


def explain(err):
    """Plain-language reading of a check_citations error code."""
    code, _, detail = err.partition(":")
    return {
        "sentence_without_citation": "a sentence has no citation",
        "no_citations": "the answer cites nothing",
        "cites_chunk_not_in_context": f"cites a passage it was not given ({detail})",
        "number_not_in_cited_chunk": f"the number {detail} is not in the cited passage",
        "quote_not_in_cited_chunk": "a quotation is not in the cited passage",
        "abstention_has_extra_text": "it declined but added other text",
        "abstention_has_citation": "it declined but still cited passages",
        "compare_missing_paper": f"no citation from {detail}",
    }.get(code, err)


papers = load_papers()
models = ollama_models()

with st.sidebar:
    st.header("Settings")
    if not models:
        st.error("Ollama is not reachable at " + generate.OLLAMA)
    default = generate.DEFAULT_MODEL if generate.DEFAULT_MODEL in models else (models or [""])[0]
    model = st.selectbox("Model", models or [generate.DEFAULT_MODEL],
                         index=(models.index(default) if default in models else 0))
    task = st.radio("Task", ["qa", "compare", "analysis"], horizontal=True,
                    format_func={"qa": "Question", "compare": "Compare", "analysis": "Summary"}.get)
    chosen = st.multiselect("Papers (empty = all)", list(papers), key="chosen",
                            format_func=lambda p: f"{p} · {papers[p]['title'][:48]}")
    k = st.slider("Passages per search", 2, 10, 5)
    fewshot = st.checkbox("Few-shot examples", help="Show the model two worked examples first")

    st.header("Your papers")
    upload = st.file_uploader("Add a paper (PDF with a text layer)", type="pdf",
                              help="Stays on this computer (data/user/, not in git).")
    if upload and st.button("Add paper"):
        with st.spinner("Reading and indexing the paper (up to 2 minutes the first time, "
                        "when the library index is built)..."):
            try:
                st.session_state["added"] = library.add(upload.getvalue(), upload.name)
                st.session_state.pop("add_error", None)
            except ValueError as e:
                st.session_state["add_error"] = str(e)
        get_retriever.clear()
        st.rerun()
    if "add_error" in st.session_state:
        st.error(st.session_state["add_error"])
    for p in library.papers():
        c1, c2 = st.columns([5, 1])
        c1.caption(f"**{p['paper_id']}** · {p['title'][:60]} · {p['n_chunks']} passages")
        c2.button("✕", key=f"rm_{p['paper_id']}", help=f"Remove {p['paper_id']}",
                  on_click=remove_paper, args=(p["paper_id"],))

hint = {"qa": "What does MELON compare to detect an attack?",
        "compare": "How do CaMeL and MELON differ?", "analysis": ""}[task]
st.title("Research Assistant")

added = st.session_state.get("added")
if added:
    verb = "was already in your library" if added["duplicate"] else "added"
    st.success(f"{added['paper_id']} {verb}: *{added['title']}*, {added['n_pages']} pages, "
               f"{added['n_chunks']} passages. Pick it under Papers to ask only about it.")
    for w in added["warnings"]:
        st.warning(w)
    with st.expander("Check how the paper was read (first 3 passages)"):
        st.caption("Garbled text, a wrong title or headings read as body text mean this "
                   "layout extracts badly; remove the paper with ✕ in the sidebar.")
        for c in added["chunks"][:3]:
            st.markdown(f"**{c['chunk_id']}** · {c['section']} · page {c['page']}")
            st.text(re.sub(r"^\[Section: [^\]]*\]\s*", "", c["text"])[:700])
    if st.button("Dismiss"):
        st.session_state.pop("added")
        st.rerun()

question =st.text_input("Question", placeholder=hint, disabled=task == "analysis",
                         help="Summary mode needs no question: pick exactly one paper.")
need = {"compare": "pick at least two papers", "analysis": "pick exactly one paper"}
ok_papers = (task == "qa" or (task == "compare" and len(chosen) >= 2)
             or (task == "analysis" and len(chosen) == 1))
if not ok_papers:
    st.info(f"For this task, {need[task]} in the sidebar.")

if st.button("Ask", type="primary", disabled=not ok_papers or (task != "analysis" and not question)):
    with st.spinner(f"Searching and asking {model}..."):
        try:
            rec = generate.answer(question, task, k, chosen or None, model,
                                  retriever=current_retriever(), fewshot=fewshot)
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
            st.stop()
    st.session_state["rec"] = rec

rec = st.session_state.get("rec")
if rec:
    ans, errs = rec["answer"], rec["citation_errors"]
    if isinstance(ans, dict):
        st.subheader("Summary")
        st.table({f: {"value": v.get("value", ""), "evidence": ", ".join(v.get("evidence", []))}
                  for f, v in ans.items() if isinstance(v, dict)})
        cited = {i for v in ans.values() if isinstance(v, dict) for i in v.get("evidence", [])}
    else:
        st.subheader("Answer")
        if ABSTAIN in ans:
            st.warning(ans)
        else:
            st.markdown(CITE_RE.sub(lambda m: f"**[{m.group(1)}]**", ans))
        cited = {i.strip() for m in CITE_RE.finditer(ans) for i in m.group(1).split(",")}
    if errs:
        st.error("Citation check failed: " + "; ".join(explain(e) for e in errs))
    else:
        st.success("Citation check passed. This means the rules were followed, "
                   "not that the answer is correct: check it against the passages below.")

    st.subheader(f"Passages given to the model ({len(rec['context'])})")
    for c in rec["context"]:
        pid = c["id"].split("_c")[0]
        sec = re.match(r"\[Section: ([^\]]*)\]", c["text"])
        mark = "✅ cited · " if c["id"] in cited else ""
        title = papers.get(pid, {}).get("title", pid)
        with st.expander(f"{mark}{c['id']} · {title[:60]} · {sec.group(1) if sec else ''}",
                         expanded=c["id"] in cited):
            st.write(re.sub(r"^\[Section: [^\]]*\]\s*", "", c["text"]))
