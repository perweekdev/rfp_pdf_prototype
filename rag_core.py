from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer


# -------------------------
# Data structures
# -------------------------
@dataclass(frozen=True)
class Evidence:
    page: int
    snippet: str

@dataclass
class IndexBundle:
    doc_id: str
    pages: List[Dict[str, Any]]          # [{"page": int, "text": str}, ...]
    embed_model_name: str
    embed_model: SentenceTransformer
    index: Any                          # faiss.IndexFlatL2
    id_to_page: List[int]               # faiss row id -> page number (1-based)


# -------------------------
# Core functions (PRD signatures)
# -------------------------
def load_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Returns pages as [{ "page": 1, "text": "..." }, ...]
    Chunking: C2 (page-based). Page number is the ground-truth unit.
    """
    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


def build_index(
    pages: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
) -> Tuple[Any, List[int]]:
    """
    Build FAISS IndexFlatL2 from per-page texts.
    Returns: (faiss_index, id_to_page)
    """
    texts = [(p.get("text") or "") for p in pages]
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype("float32")

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    id_to_page = [int(p["page"]) for p in pages]
    return index, id_to_page


def retrieve_pages(
    query: str,
    pages: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    index: Any,
    id_to_page: List[int],
    top_k: int = 5,
    snippet_chars: int = 600,
) -> List[Dict[str, Any]]:
    """
    Returns evidence list: [{page:int, snippet:str}, ...]
    snippet must be verbatim from page text (no paraphrase).
    """
    if not query or not query.strip():
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    _D, I = index.search(q_emb, min(top_k, len(id_to_page)))

    out: List[Dict[str, Any]] = []
    seen_pages = set()
    for idx in I[0].tolist():
        if idx < 0 or idx >= len(id_to_page):
            continue
        page_no = id_to_page[idx]
        if page_no in seen_pages:
            continue
        seen_pages.add(page_no)

        page_text = (pages[page_no - 1].get("text") or "")
        snippet = page_text[:snippet_chars]  # verbatim slice
        if snippet.strip():
            out.append({"page": int(page_no), "snippet": snippet})
    return out


def make_summary(
    bundle: IndexBundle,
    llm_client: Any,
    max_evidence: int = 5,
) -> Dict[str, Any]:
    """
    Output rules:
    - {summary_text: str, evidence: [{page:int, snippet:str}]*<=5}
    - If no evidence/context => summary_text="NOT_FOUND"
    - LLM fail/parsing fail => log "GEN_FAIL", UI shows retry message (handled in app.py)
    """
    # v0 strategy: retrieve a few pages with a fixed "summary query"
    summary_query = "이 문서의 목적, 범위, 일정, 예산/금액, 평가 기준의 핵심을 요약해줘."
    evidence = retrieve_pages(
        query=summary_query,
        pages=bundle.pages,
        embed_model=bundle.embed_model,
        index=bundle.index,
        id_to_page=bundle.id_to_page,
        top_k=max_evidence,
        snippet_chars=900,
    )

    if not evidence:
        return {"summary_text": "NOT_FOUND", "evidence": []}

    context = _build_context_from_evidence(bundle.pages, evidence, max_chars=4000)

    try:
        # Minimal: free-form summary (not JSON) + we attach evidence separately by policy
        prompt = (
            "너는 RFP PDF에서 정보를 찾는 보조자다.\n"
            "규칙: CONTEXT에 근거한 내용만 요약하고, 추측하지 마라.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "요약을 6~10줄로 작성하라."
        )
        summary_text = _call_llm_text(llm_client, prompt).strip()
        if not summary_text:
            _log_internal("GEN_FAIL", {"where": "make_summary", "reason": "empty"})
            return {"summary_text": "GEN_FAIL", "evidence": evidence[:max_evidence]}
        return {"summary_text": summary_text, "evidence": evidence[:max_evidence]}
    except Exception as e:
        _log_internal("GEN_FAIL", {"where": "make_summary", "exc": repr(e)})
        return {"summary_text": "GEN_FAIL", "evidence": evidence[:max_evidence]}


def answer_question(
    bundle: IndexBundle,
    question: str,
    llm_client: Any,
    max_evidence: int = 3,
) -> Dict[str, Any]:
    """
    Output rules:
    - {answer: str, evidence: [{page:int, snippet:str}]*<=3}
    - If no evidence/context => answer="NOT_FOUND"
    - LLM fail/parsing fail => log "GEN_FAIL", UI shows retry message (handled in app.py)
    """
    evidence = retrieve_pages(
        query=question,
        pages=bundle.pages,
        embed_model=bundle.embed_model,
        index=bundle.index,
        id_to_page=bundle.id_to_page,
        top_k=max_evidence,
        snippet_chars=900,
    )
    if not evidence:
        return {"answer": "NOT_FOUND", "evidence": []}

    context = _build_context_from_evidence(bundle.pages, evidence, max_chars=4000)

    try:
        prompt = (
            "너는 RFP PDF의 CONTEXT 발췌에서만 답변한다.\n"
            "규칙: CONTEXT에 없는 내용은 답하지 말고 NOT_FOUND라고만 써라.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "답변:"
        )
        ans = _call_llm_text(llm_client, prompt).strip()
        if not ans:
            _log_internal("GEN_FAIL", {"where": "answer_question", "reason": "empty"})
            return {"answer": "GEN_FAIL", "evidence": evidence[:max_evidence]}
        return {"answer": ans, "evidence": evidence[:max_evidence]}
    except Exception as e:
        _log_internal("GEN_FAIL", {"where": "answer_question", "exc": repr(e)})
        return {"answer": "GEN_FAIL", "evidence": evidence[:max_evidence]}


# -------------------------
# Internal helpers
# -------------------------
def _build_context_from_evidence(pages: List[Dict[str, Any]], evidence: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts: List[str] = []
    for ev in evidence:
        p = int(ev["page"])
        txt = (pages[p - 1].get("text") or "")
        parts.append(f"[페이지 {p}]\n{txt}\n")
    ctx = "\n".join(parts)
    return ctx[:max_chars]


def _call_llm_text(llm_client: Any, prompt: str) -> str:
    """
    Adapter point.
    - In your old code, OpenAI Responses API was used.
    - Keep this minimal: return text only.
    """
    # Example for openai>=1.x Responses API style (you can replace later)
    resp = llm_client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        max_output_tokens=1200,
        reasoning={"effort": "minimal"},
    )
    return (getattr(resp, "output_text", "") or "")


def _log_internal(tag: str, payload: Dict[str, Any]) -> None:
    # v0: print is enough; later swap to logging module / file
    print(f"[{tag}] {payload}")