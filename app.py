"""
AI Career Copilot — Session 02: Ingestion, Embeddings & Matching (Web App)
==========================================================================
Imports logic from starter.py — as you implement each TODO in starter.py,
the file upload, semantic matching, and skill-gap analysis update dynamically.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Import logic from starter.py ─────────────────────────────────────────────
try:
    from starter import (
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
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = f"Error importing starter.py: {e}"

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = (
    "You are Career Copilot, a friendly and professional AI career coach. "
    "Help the candidate land their dream job. Keep replies concise and actionable."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def lc_history_from_gradio(gr_history: list) -> List[BaseMessage]:
    """Convert Gradio chat history format to LangChain BaseMessage list."""
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
        raise ValueError("GROQ_API_KEY not set in .env")
    return ChatGroq(model=model, api_key=api_key, temperature=temperature, max_retries=3)


# ── Gradio Handlers ───────────────────────────────────────────────────────────

def chat(user_message: str, history: list, model: str, temperature: float):
    """Handle general career coaching chat turns."""
    if not user_message.strip():
        yield history
        return

    try:
        llm = _get_llm(model, temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
    except Exception as e:
        yield history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"⚠️ **Error initializing model:** {e}"},
        ]
        return

    lc_history = lc_history_from_gradio(history)
    history = history + [{"role": "user", "content": user_message}]
    history.append({"role": "assistant", "content": ""})

    try:
        for chunk in chain.stream({"history": lc_history, "question": user_message}):
            history[-1]["content"] += chunk
            yield history
    except Exception as e:
        history[-1]["content"] = f"⚠️ **Streaming Error:** {e}"
        yield history


def run_match_pipeline(
    resume_text: str,
    resume_file,
    job_files,
    model: str,
    temperature: float,
    top_k: int,
):
    """Run full ingestion, semantic vector search, and skill-gap analysis."""
    if not _IMPORT_OK:
        return f"⚠️ **Import error:** `{_IMPORT_ERR}`"

    try:
        # Prepare job postings directory
        if job_files:
            jobs_dir = tempfile.mkdtemp(prefix="cc2_custom_")
            for f in job_files:
                src = Path(f.name)
                (Path(jobs_dir) / src.name).write_bytes(src.read_bytes())
        else:
            jobs_dir = tempfile.mkdtemp(prefix="cc2_sample_")
            for fname, content in SAMPLE_JOBS_DIR_CONTENT.items():
                Path(jobs_dir, fname).write_text(content, encoding="utf-8")

        # Select resume source
        resume_source = resume_text.strip() if resume_text.strip() else SAMPLE_RESUME
        if resume_file:
            resume_source = str(resume_file.name)

        result: CopilotV2Result = run_copilot_v2(resume_source, jobs_dir)

        md = f"**📄 Resume Chunks Indexed:** `{result.resume_chunks}` | **🏢 Job Postings:** `{result.job_count}`\n\n"
        md += "### 🏆 Top Job Matches\n\n"
        for i, m in enumerate(result.top_matches, 1):
            bar = "█" * int(m.similarity / 10) + "░" * (10 - int(m.similarity / 10))
            md += f"**{i}. {m.title} @ {m.company}** — `{m.similarity}%` `[{bar}]`\n\n"

        md += f"---\n### 🔍 Skill Gap Analysis vs. **{result.top_matches[0].title}**\n\n"
        md += f"**✅ Matched Skills:** {', '.join(result.skill_gap.matched_skills) if result.skill_gap.matched_skills else 'None'}\n\n"
        md += f"**❌ Missing Skills:** {', '.join(result.skill_gap.missing_skills) if result.skill_gap.missing_skills else 'None'}\n\n"
        md += f"**📋 Assessment:** {result.skill_gap.fit_summary}\n\n"
        md += "### 💡 Recommended Action Items:\n" + "\n".join(f"- {t}" for t in result.skill_gap.improvement_tips)
        return md

    except NotImplementedError as e:
        return f"⚠️ **TODO Not Implemented Yet:** `{e}`\n\nOpen `starter.py` and implement this function to enable resume matching."
    except Exception as e:
        return f"⚠️ **Error running pipeline:** {e}\n\nEnsure `GROQ_API_KEY` and `HUGGING_FACE_API_KEY` are configured in `.env`."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Career Copilot v2 — Ingestion & Matching") as demo:
    gr.Markdown(
        "# 🧭 AI Career Copilot — Session 02\n"
        "### Ingestion, Embeddings, Chroma Vector Search & Skill-Gap Analysis\n"
        "Upload your resume and target job postings to discover top job matches and personalized skill-gap insights."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            model_dd = gr.Dropdown(GROQ_MODELS, value=GROQ_MODELS[0], label="LLM Model")
            temp_sl = gr.Slider(0.0, 1.5, value=0.5, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K Matches")
            if not os.getenv("GROQ_API_KEY"):
                gr.Markdown("> ⚠️ **Warning:** `GROQ_API_KEY` not found in `.env`.")
            if not os.getenv("HUGGING_FACE_API_KEY"):
                gr.Markdown("> ⚠️ **Warning:** `HUGGING_FACE_API_KEY` not found in `.env` (required for embeddings).")
            gr.Markdown("---")
            gr.Markdown("_Core logic is imported from `starter.py`._")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("🎯 Resume Match & Skill Gap"):
                    gr.Markdown("#### 📄 1. Your Resume")
                    resume_file = gr.File(label="Upload Resume (.txt or .pdf)", file_types=[".txt", ".pdf"])
                    resume_text = gr.Textbox(
                        label="Or Paste Resume Text (used if no file uploaded)",
                        value=SAMPLE_RESUME if _IMPORT_OK else "",
                        lines=6,
                    )
                    gr.Markdown("#### 🏢 2. Target Job Postings")
                    job_files = gr.File(
                        label="Upload .txt Job Files (leave empty to use built-in sample tech jobs)",
                        file_types=[".txt"],
                        file_count="multiple",
                    )
                    run_btn = gr.Button("🚀 Find My Best Matches & Analyze Skill Gap", variant="primary")
                    results_md = gr.Markdown("_Click the button above to run the ingestion & matching pipeline._")

                with gr.Tab("💬 Career Coach Chat"):
                    chatbot = gr.Chatbot(label="Career Copilot", height=400)
                    with gr.Row():
                        msg_box = gr.Textbox(placeholder="Ask any career, CV, or interview question...", label="Message", scale=5)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat")

    send_btn.click(chat, [msg_box, chatbot, model_dd, temp_sl], chatbot).then(lambda: "", outputs=msg_box)
    msg_box.submit(chat, [msg_box, chatbot, model_dd, temp_sl], chatbot).then(lambda: "", outputs=msg_box)
    clear_btn.click(lambda: [], outputs=chatbot)

    run_btn.click(
        run_match_pipeline,
        inputs=[resume_text, resume_file, job_files, model_dd, temp_sl, top_k],
        outputs=results_md,
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
