"""
Career Copilot — Session 02 Gradio UI (v2)
==========================================
Imports all logic from solution.py — students add code there and the UI
reflects their work immediately.

New in v2: resume upload + job postings upload + match & analyse tab.

Run:
    python demos/02_ingestion/career_copilot/app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

def _find_env() -> Path:
    for p in _HERE.parents:
        if (p / ".env").exists():
            return p / ".env"
    return _HERE / ".env"

from dotenv import load_dotenv
load_dotenv(_find_env())
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gradio as gr

# ── import from solution (students edit this file) ────────────────────────────
try:
    from solution import (
        get_llm,
        get_embeddings,
        load_resume,
        load_job_postings,
        chunk_documents,
        build_job_vectorstore,
        find_best_matches,
        analyze_skill_gap,
        run_copilot_v2,
        SAMPLE_RESUME,
        SAMPLE_JOBS_DIR_CONTENT,
        JobMatch,
        SkillGapAnalysis,
        CopilotV2Result,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except NotImplementedError as e:
    _IMPORT_OK = False
    _IMPORT_ERR = f"solution.py has unfinished TODOs: {e}"
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = f"Error importing solution.py: {e}"

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = (
    "You are Career Copilot, a friendly and professional AI career coach. "
    "Help the candidate land their dream job. Keep replies concise."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def lc_history_from_gradio(gr_history: list) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for msg in gr_history:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def _get_llm(model: str, temperature: float = 0.5):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    return ChatGroq(model=model, api_key=api_key, temperature=temperature, max_retries=3)


# ── Gradio handlers ───────────────────────────────────────────────────────────

def chat(user_message: str, history: list, model: str, temperature: float):
    if not user_message.strip():
        yield history
        return
    llm = _get_llm(model, temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    lc_history = lc_history_from_gradio(history)
    history = history + [{"role": "user", "content": user_message}]
    history.append({"role": "assistant", "content": ""})
    for chunk in chain.stream({"history": lc_history, "question": user_message}):
        history[-1]["content"] += chunk
        yield history


def run_match_pipeline(
    resume_text: str,
    resume_file,
    job_files,
    model: str,
    temperature: float,
    top_k: int,
):
    if not _IMPORT_OK:
        return f"**Import error:** {_IMPORT_ERR}"

    try:
        # Write sample jobs to a temp dir if no files uploaded
        if job_files:
            jobs_dir = tempfile.mkdtemp(prefix="cc2_custom_")
            for f in job_files:
                src = Path(f.name)
                (Path(jobs_dir) / src.name).write_bytes(src.read_bytes())
        else:
            jobs_dir = tempfile.mkdtemp(prefix="cc2_sample_")
            for fname, content in SAMPLE_JOBS_DIR_CONTENT.items():
                Path(jobs_dir, fname).write_text(content, encoding="utf-8")

        # Use resume text or uploaded file
        resume_source = resume_text.strip() if resume_text.strip() else SAMPLE_RESUME
        if resume_file:
            resume_source = str(resume_file.name)

        result: CopilotV2Result = run_copilot_v2(resume_source, jobs_dir)

        md = f"**Resume chunks indexed:** {result.resume_chunks} | **Jobs:** {result.job_count}\n\n"
        md += "### Top Job Matches\n\n"
        for i, m in enumerate(result.top_matches, 1):
            bar = "|" * int(m.similarity / 5)
            md += f"**{i}. {m.title} @ {m.company}** — {m.similarity}%  {bar}\n\n"
        md += f"---\n### Skill Gap vs. **{result.top_matches[0].title}**\n\n"
        md += f"**Matched:** {', '.join(result.skill_gap.matched_skills)}\n\n"
        md += f"**Missing:** {', '.join(result.skill_gap.missing_skills)}\n\n"
        md += f"**Summary:** {result.skill_gap.fit_summary}\n\n"
        md += "**Tips:**\n" + "\n".join(f"- {t}" for t in result.skill_gap.improvement_tips)
        return md

    except NotImplementedError as e:
        return f"**TODO not implemented:** {e}"
    except Exception as e:
        return f"**Error:** {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Career Copilot v2 — Ingestion & Matching") as demo:
    gr.Markdown(
        "# Career Copilot\n"
        "### Session 02 — Resume & Job Ingestion\n"
        "Upload your resume and job postings. The copilot will find your best matches "
        "and show a skill-gap analysis."
    )

    if not _IMPORT_OK:
        gr.Markdown(f"> **Import warning:** `{_IMPORT_ERR}`\n\n"
                    "_Complete the TODOs in `solution.py` and reload the app._")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            model_dd = gr.Dropdown(GROQ_MODELS, value=GROQ_MODELS[0], label="Model")
            temp_sl = gr.Slider(0.0, 1.5, value=0.5, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K matches")
            if not os.getenv("GROQ_API_KEY"):
                gr.Markdown("> **Warning:** GROQ_API_KEY not found.")
            if not os.getenv("HUGGING_FACE_API_KEY"):
                gr.Markdown("> **Warning:** HUGGING_FACE_API_KEY not found — embeddings will fail.")
            gr.Markdown("---")
            gr.Markdown("_Logic lives in `solution.py`._")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(label="Career Copilot", height=400)
                    with gr.Row():
                        msg_box = gr.Textbox(placeholder="Ask a career question...", label="Message", scale=5)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear")

                with gr.Tab("Match & Analyse"):
                    gr.Markdown("#### Your Resume")
                    resume_file = gr.File(label="Upload resume (.txt or .pdf)", file_types=[".txt", ".pdf"])
                    resume_text = gr.Textbox(
                        label="Or paste resume text (used if no file uploaded)",
                        value=SAMPLE_RESUME if _IMPORT_OK else "",
                        lines=6,
                    )
                    gr.Markdown("#### Job Postings")
                    job_files = gr.File(
                        label="Upload .txt job files (leave empty to use built-in samples)",
                        file_types=[".txt"],
                        file_count="multiple",
                    )
                    run_btn = gr.Button("Find My Best Matches", variant="primary")
                    results_md = gr.Markdown("_Click 'Find My Best Matches' to start._")

    send_btn.click(chat, [msg_box, chatbot, model_dd, temp_sl], chatbot).then(lambda: "", outputs=msg_box)
    msg_box.submit(chat, [msg_box, chatbot, model_dd, temp_sl], chatbot).then(lambda: "", outputs=msg_box)
    clear_btn.click(lambda: [], outputs=chatbot)

    run_btn.click(
        run_match_pipeline,
        inputs=[resume_text, resume_file, job_files, model_dd, temp_sl, top_k],
        outputs=results_md,
    )

    gr.Markdown(
        "_Imports: `load_resume`, `load_job_postings`, `chunk_documents`, `build_job_vectorstore`, "
        "`find_best_matches`, `analyze_skill_gap`, `run_copilot_v2` from `solution.py`_"
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
