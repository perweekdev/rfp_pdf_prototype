from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import gradio as gr
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

from rag_core import (
    IndexBundle,
    load_pdf_pages,
    build_index,
    make_summary,
    answer_question,
)

PDF_DIR = Path(__file__).parent / "data" / "pdfs"
EMBED_MODEL_NAME = "nlpai-lab/KoE5"  # PRD fixed
TOP_K_SUMMARY = 5
TOP_K_QA = 3


def list_pdfs() -> List[str]:
    if not PDF_DIR.exists():
        return []
    return sorted([p.name for p in PDF_DIR.glob("*.pdf")])


def ensure_bundle(doc_name: str, cache: Dict[str, Any]) -> Tuple[Optional[IndexBundle], str]:
    """
    P1 cache: same document -> skip rebuild.
    Returns (bundle, status_msg)
    """
    if not doc_name:
        return None, "문서를 선택하세요."

    if cache.get("doc_id") == doc_name and cache.get("bundle") is not None:
        return cache["bundle"], f"캐시 사용: {doc_name}"

    pdf_path = PDF_DIR / doc_name
    if not pdf_path.exists():
        return None, f"파일 없음: {doc_name}"

    pages = load_pdf_pages(pdf_path)

    # Embed model: keep one global instance per process for speed
    embed_model = cache.get("embed_model")
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        cache["embed_model"] = embed_model

    index, id_to_page = build_index(pages, embed_model)

    bundle = IndexBundle(
        doc_id=doc_name,
        pages=pages,
        embed_model_name=EMBED_MODEL_NAME,
        embed_model=embed_model,
        index=index,
        id_to_page=id_to_page,
    )
    cache["doc_id"] = doc_name
    cache["bundle"] = bundle
    return bundle, f"로딩 완료: {doc_name} (pages={len(pages)})"


def ui_make_summary(doc_name: str, cache: Dict[str, Any]) -> Tuple[str, str]:
    bundle, msg = ensure_bundle(doc_name, cache)
    if bundle is None:
        return msg, ""

    client = cache.get("llm_client")
    if client is None:
        client = OpenAI()
        cache["llm_client"] = client

    result = make_summary(bundle=bundle, llm_client=client, max_evidence=TOP_K_SUMMARY)

    if result["summary_text"] == "GEN_FAIL":
        return "요약 생성에 실패했습니다. 다시 시도해주세요.", ""

    summary_text = result["summary_text"]
    ev_md = evidence_to_markdown(result.get("evidence", []))
    return summary_text, ev_md


from typing import Any, Dict, List, Optional

from openai import OpenAI

# history: gr.Chatbot(type="messages")이면
# - List[{"role": "user"|"assistant", "content": str}, ...] 형태를 기대함 [web:44][web:124]
def ui_answer(
    doc_name: str,
    question: str,
    history: Optional[List[Dict[str, Any]]],
    cache: Dict[str, Any],
):
    history = history or []
    question = (question or "").strip()
    if not question:
        return "", history

    bundle, _msg = ensure_bundle(doc_name, cache)
    if bundle is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "문서를 먼저 선택하세요."},
        ]
        return "", history

    client = cache.get("llm_client")
    if client is None:
        client = OpenAI()
        cache["llm_client"] = client

    result = answer_question(
        bundle=bundle,
        question=question,
        llm_client=client,
        max_evidence=TOP_K_QA,
    )

    if result.get("answer") == "GEN_FAIL":
        bot = "답변 생성에 실패했습니다. 다시 시도해주세요."
    else:
        bot = result.get("answer", "") or "NOT_FOUND"

    ev_md = evidence_to_markdown(result.get("evidence", []))
    if ev_md.strip():
        bot = bot + "\n\n" + ev_md

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": bot},
    ]
    return "", history

def evidence_to_markdown(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    lines = ["### Evidence"]
    for ev in evidence:
        p = int(ev["page"])
        snip = (ev.get("snippet") or "").replace("\n", " ")
        lines.append(f"- p.{p}: {snip}")
    return "\n".join(lines)


def main():
    pdf_choices = list_pdfs()

    with gr.Blocks() as demo:
        gr.Markdown("# RFP PDF 프로토타입 v0")

        cache = gr.State({"doc_id": None, "bundle": None, "embed_model": None, "llm_client": None})

        doc_dd = gr.Dropdown(choices=pdf_choices, label="문서 선택 (사전 적재 PDF)", value=pdf_choices[0] if pdf_choices else None)

        with gr.Tabs():
            with gr.TabItem("요약"):
                sum_btn = gr.Button("요약 생성")
                summary_out = gr.Textbox(label="요약", lines=10)
                summary_ev = gr.Markdown()

                sum_btn.click(fn=ui_make_summary, inputs=[doc_dd, cache], outputs=[summary_out, summary_ev])

            with gr.TabItem("Q&A"):
                chatbot = gr.Chatbot(label="문서 기반 Q&A")
                msg = gr.Textbox(label="질문", placeholder="질문을 입력하고 Enter")
                clear = gr.ClearButton([msg, chatbot])

                msg.submit(fn=ui_answer, inputs=[doc_dd, msg, chatbot, cache], outputs=[msg, chatbot])


    demo.launch()


if __name__ == "__main__":
    main()