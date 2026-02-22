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
EMBED_MODEL_NAME = "nlpai-lab/KoE5"
TOP_K_SUMMARY = 5
TOP_K_QA = 3

CSS = """
#summary_textbox textarea {
  font-size: calc(1rem + 2pt) !important;
  line-height: 1.2 !important;
}
"""


def list_pdfs() -> List[str]:
    if not PDF_DIR.exists():
        return []
    return sorted([p.name for p in PDF_DIR.glob("*.pdf")])


def ensure_bundle(doc_name: str, cache: Dict[str, Any]) -> Tuple[Optional[IndexBundle], str]:
    if not doc_name:
        return None, "문서를 선택하세요."

    if cache.get("doc_id") == doc_name and cache.get("bundle") is not None:
        return cache["bundle"], f"캐시 사용: {doc_name}"

    pdf_path = PDF_DIR / doc_name
    if not pdf_path.exists():
        return None, f"파일 없음: {doc_name}"

    pages = load_pdf_pages(pdf_path)

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


# -----------------------------
# Summary UI handlers
# -----------------------------
def ui_make_summary(doc_name: str, cache: Dict[str, Any]):
    """
    outputs:
      summary_text(str),
      summary_ev_list (Radio update),
      summary_ev_detail (str)
    """
    try:
        bundle, msg = ensure_bundle(doc_name, cache)
        if bundle is None:
            cache["summary_evidence"] = []
            return msg, gr.update(choices=[], value=None), ""

        client = cache.get("llm_client")
        if client is None:
            client = OpenAI()
            cache["llm_client"] = client

        result = make_summary(bundle=bundle, llm_client=client, max_evidence=TOP_K_SUMMARY)

        if result.get("summary_text") == "GEN_FAIL":
            summary_text = "요약 생성에 실패했습니다. 다시 시도해주세요."
        else:
            summary_text = result.get("summary_text", "") or "NOT_FOUND"

        evidence = result.get("evidence", []) or []
        cache["summary_evidence"] = evidence

        choices = [f"p.{int(ev['page']):02d}  [자세히 보기]" for ev in evidence]
        return summary_text, gr.update(choices=choices, value=None), ""

    except Exception as e:
        # 클릭은 됐는데 내부 예외로 “안 눌리는 것처럼 보이는” 문제를 방지
        cache["summary_evidence"] = []
        return f"요약 처리 중 오류가 발생했습니다. 터미널 로그를 확인해주세요.\n\n{repr(e)}", gr.update(choices=[], value=None), ""


def ui_show_summary_evidence(selected_label: str, cache: Dict[str, Any]) -> str:
    evidence = cache.get("summary_evidence", []) or []
    if not selected_label:
        return ""

    try:
        page_no = int(selected_label.split()[0].replace("p.", ""))
    except Exception:
        return ""

    for ev in evidence:
        if int(ev.get("page", -1)) == page_no:
            snip = ev.get("snippet") or ""
            return f"**p.{page_no:02d}**\n\n```\n{snip}\n```"
    return ""


# -----------------------------
# Q&A UI handlers
# -----------------------------
def toggle_ask_btn(text: str):
    text = (text or "").strip()
    return gr.update(interactive=bool(text))


def ui_answer(
    doc_name: str,
    question: str,
    history: Optional[List[Dict[str, Any]]],
    cache: Dict[str, Any],
):
    history = history or []
    question = (question or "").strip()
    if not question:
        return "", history, gr.update(), "", gr.update(interactive=False)

    bundle, _msg = ensure_bundle(doc_name, cache)
    if bundle is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "문서를 먼저 선택하세요."},
        ]
        _append_turn_record(cache, question=question, answer="문서를 먼저 선택하세요.", evidence=[])
        turn_update = _qa_turn_choices_update(cache, value=None)
        return "", history, turn_update, "", gr.update(interactive=False)

    client = cache.get("llm_client")
    if client is None:
        client = OpenAI()
        cache["llm_client"] = client

    result = answer_question(bundle=bundle, question=question, llm_client=client, max_evidence=TOP_K_QA)

    if result.get("answer") == "GEN_FAIL":
        answer_text = "답변 생성에 실패했습니다. 다시 시도해주세요."
        evidence = result.get("evidence", []) or []
    else:
        answer_text = result.get("answer", "") or "NOT_FOUND"
        evidence = result.get("evidence", []) or []

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text},
    ]

    _append_turn_record(cache, question=question, answer=answer_text, evidence=evidence)

    turn_update = _qa_turn_choices_update(cache, value=None)
    return "", history, turn_update, "", gr.update(interactive=False)


def _append_turn_record(cache: Dict[str, Any], question: str, answer: str, evidence: List[Dict[str, Any]]) -> None:
    turns = cache.get("qa_turns")
    if not isinstance(turns, list):
        turns = []
        cache["qa_turns"] = turns

    turn_id = len(turns) + 1
    turns.append(
        {
            "turn_id": turn_id,
            "question": question,
            "answer": answer,
            "evidence": evidence,
        }
    )


def _qa_turn_choices_update(cache: Dict[str, Any], value: Optional[str]):
    turns = cache.get("qa_turns") or []
    choices = [f"{t['turn_id']:02d}. Q: {t['question'][:40]}" for t in turns]
    return gr.update(choices=choices, value=value)


def ui_show_qa_evidence(selected_turn_label: str, cache: Dict[str, Any]) -> str:
    turns = cache.get("qa_turns") or []
    if not selected_turn_label:
        return ""

    try:
        turn_id = int(selected_turn_label.split(".")[0])
    except Exception:
        return ""

    target = None
    for t in turns:
        if int(t.get("turn_id", -1)) == turn_id:
            target = t
            break
    if not target:
        return ""

    evs = target.get("evidence", []) or []
    if not evs:
        return "근거 없음"

    lines = []
    for ev in evs:
        p = int(ev["page"])
        snip = ev.get("snippet") or ""
        lines.append(f"**p.{p:02d}**\n\n```\n{snip}\n```")
    return "\n\n---\n\n".join(lines)


def ui_clear_chat(cache: Dict[str, Any]):
    cache["qa_turns"] = []
    return [], gr.update(choices=[], value=None), "", gr.update(interactive=False)


def main():
    pdf_choices = list_pdfs()

    # 1) 앱 시작 시 임베딩 모델 선로드
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2) 웜업: 첫 encode가 느린 문제 완화
    _ = embed_model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)

    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# RFP PDF 프로토타입 v0")

        cache = gr.State(
            {
                "doc_id": None,
                "bundle": None,
                "embed_model": None,
                "llm_client": None,
                "summary_evidence": [],
                "qa_turns": [],
            }
        )

        doc_dd = gr.Dropdown(
            choices=pdf_choices,
            label="문서 선택 (사전 적재 PDF)",
            value=pdf_choices[0] if pdf_choices else None,
        )

        with gr.Tabs():
            with gr.TabItem("요약"):
                sum_btn = gr.Button("요약 생성")
                summary_out = gr.Textbox(label="요약", lines=10, elem_id="summary_textbox")

                with gr.Row():
                    with gr.Column(scale=1):
                        summary_ev_list = gr.Radio(choices=[], label="Evidence", value=None)
                    with gr.Column(scale=2):
                        summary_ev_detail = gr.Markdown("")

                sum_btn.click(
                    fn=ui_make_summary,
                    inputs=[doc_dd, cache],
                    outputs=[summary_out, summary_ev_list, summary_ev_detail],
                    show_progress="full",
                )

                summary_ev_list.change(
                    fn=ui_show_summary_evidence,
                    inputs=[summary_ev_list, cache],
                    outputs=[summary_ev_detail],
                    show_progress="hidden",
                )

            with gr.TabItem("Q&A"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="문서 기반 Q&A")
                        msg = gr.Textbox(label="질문", placeholder="질문을 입력하세요")
                        with gr.Row():
                            ask_btn = gr.Button("질문하기", variant="primary", interactive=False)
                            clear_btn = gr.Button("대화 초기화")
                    with gr.Column(scale=1):
                        qa_turn_list = gr.Radio(choices=[], label="이전 질문(턴) 선택", value=None)
                        qa_evidence_detail = gr.Markdown("")

                msg.change(fn=toggle_ask_btn, inputs=[msg], outputs=[ask_btn], show_progress="hidden")

                ask_btn.click(
                    fn=ui_answer,
                    inputs=[doc_dd, msg, chatbot, cache],
                    outputs=[msg, chatbot, qa_turn_list, qa_evidence_detail, ask_btn],
                    show_progress="hidden",
                )
                msg.submit(
                    fn=ui_answer,
                    inputs=[doc_dd, msg, chatbot, cache],
                    outputs=[msg, chatbot, qa_turn_list, qa_evidence_detail, ask_btn],
                    show_progress="hidden",
                )

                qa_turn_list.change(
                    fn=ui_show_qa_evidence,
                    inputs=[qa_turn_list, cache],
                    outputs=[qa_evidence_detail],
                    show_progress="hidden",
                )

                clear_btn.click(
                    fn=ui_clear_chat,
                    inputs=[cache],
                    outputs=[chatbot, qa_turn_list, qa_evidence_detail, ask_btn],
                    show_progress="hidden",
                )

        demo.launch()

if __name__ == "__main__":
    main()