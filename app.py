"""
AI Career Copilot — Session 04: LangGraph Stateful Workflow (Web Dashboard)
==========================================================================
Interactive Gradio Dashboard featuring:
- Dynamic action routing (Tailor Resume / Cover Letter / Mock Interview)
- Automated self-correction review cycles
- Human-in-the-Loop approval gate (interrupt_before)
- Live LangGraph Checkpoint State Inspector (MemorySaver)
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gradio as gr

# ── Import logic from starter.py ─────────────────────────────────────────────
try:
    from starter import (
        build_graph,
        SAMPLE_RESUME,
        SAMPLE_JOB,
        CopilotState,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = f"Error importing starter.py: {e}"
    SAMPLE_RESUME = ""
    SAMPLE_JOB = ""

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# ── Graph Cache ───────────────────────────────────────────────────────────────
_graph_cache: dict = {}


def get_graph():
    """Build (or return cached) compiled LangGraph state machine from starter.py."""
    if not _IMPORT_OK:
        raise RuntimeError(_IMPORT_ERR)
    key = "graph"
    if key not in _graph_cache:
        _graph_cache[key] = build_graph()
    return _graph_cache[key]


# ── Gradio Handlers ───────────────────────────────────────────────────────────

def run_workflow(resume: str, job: str, action: str, max_iter: int, thread_id: str):
    """Execute the LangGraph workflow up to completion or human interrupt."""
    if not _IMPORT_OK:
        return f"⚠️ **Import error:** `{_IMPORT_ERR}`", "{}", gr.update(visible=False), "Import error."

    try:
        app = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "resume": resume or SAMPLE_RESUME,
            "job_posting": job or SAMPLE_JOB,
            "desired_action": action,
            "target_role": "",
            "company": "",
            "tailored_bullets": "",
            "cover_letter_draft": "",
            "editor_feedback": "",
            "quality_score": 0.0,
            "approved": False,
            "interview_questions": "",
            "iteration": 0,
            "max_iterations": int(max_iter),
            "thread_id": thread_id,
        }

        log_lines = [f"🚀 Starting workflow on thread '{thread_id}' with action: '{action}'..."]
        for event in app.stream(initial, config, stream_mode="values"):
            it = event.get("iteration", 0)
            qs = event.get("quality_score", 0.0)
            if event.get("cover_letter_draft") and it > 0:
                log_lines.append(f"✍️ Draft revision {it} generated — Editor Quality Score: {qs:.2f}/1.0")
            if event.get("tailored_bullets"):
                log_lines.append("📄 Resume bullet tailoring completed.")
            if event.get("interview_questions"):
                log_lines.append("🎯 Interview questions generated.")

        state_snap = app.get_state(config)
        paused = bool(state_snap.next)
        vals = state_snap.values
        result_md = _render_result(action, vals)
        state_json = json.dumps({k: v for k, v in vals.items() if v}, indent=2, default=str)
        approval_visible = paused and "hitl_approval" in (state_snap.next or [])

        if approval_visible:
            log_lines.append("⏸️ Workflow PAUSED at 'hitl_approval' node — Human approval required to proceed.")
        else:
            log_lines.append("✅ Workflow reached END state successfully.")

        log = "\n".join(log_lines)
        return result_md, state_json, gr.update(visible=approval_visible), log

    except NotImplementedError as e:
        return (
            f"⚠️ **TODO Not Implemented Yet:** `{e}`\n\nOpen `starter.py` and implement this function to enable the workflow.",
            "{}",
            gr.update(visible=False),
            str(e),
        )
    except Exception as e:
        return f"⚠️ **Execution Error:** {e}", "{}", gr.update(visible=False), str(e)


def approve_workflow(thread_id: str):
    """Resume an interrupted LangGraph session after human approval."""
    if not _IMPORT_OK:
        return f"⚠️ **Import error:** `{_IMPORT_ERR}`", "{}", gr.update(visible=False), "Import error."

    try:
        app = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        app.invoke(None, config)  # Resume without state updates
        vals = app.get_state(config).values
        return (
            _render_result("cover_letter", vals),
            json.dumps({k: v for k, v in vals.items() if v}, indent=2, default=str),
            gr.update(visible=False),
            "✅ Human approved — workflow resumed from checkpoint and completed.",
        )
    except Exception as e:
        return f"⚠️ **Error during approval:** {e}", "{}", gr.update(visible=False), str(e)


def _render_result(action: str, vals: dict) -> str:
    """Render markdown representation of current state results."""
    if vals.get("cover_letter_draft"):
        cl = vals["cover_letter_draft"]
        qs = vals.get("quality_score", 0.0)
        it = vals.get("iteration", 0)
        fb = vals.get("editor_feedback", "")
        status_tag = "✅ Approved by Human/Editor" if vals.get("approved") else "⏳ Pending Review"
        md = f"### ✉️ Cover Letter (Iteration {it} | Quality: `{qs:.2f}/1.0` | {status_tag})\n\n{cl}"
        if fb:
            md += f"\n\n---\n**📝 Editor Feedback:** {fb}"
        return md
    if vals.get("tailored_bullets"):
        return f"### 📄 Tailored Resume Bullets\n\n{vals['tailored_bullets']}"
    if vals.get("interview_questions"):
        return f"### 🎯 Role-Specific Interview Questions\n\n{vals['interview_questions']}"
    return "_Click 'Run Workflow' to execute the LangGraph state machine._"


# ── Gradio Dashboard UI ───────────────────────────────────────────────────────

with gr.Blocks(title="AI Career Copilot v4 — LangGraph Stateful Workflow") as demo:
    gr.Markdown(
        "# 🧭 AI Career Copilot — Session 04\n"
        "### Stateful Multi-Step LangGraph Workflow with Self-Correction & Human-in-the-Loop\n"
        "Route career requests dynamically, run automated editor feedback loops, and approve drafts with state persistence."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Workflow Settings")
            action_radio = gr.Radio(
                ["cover_letter", "tailor_resume", "mock_interview"],
                value="cover_letter",
                label="Select Action / Specialist Node",
            )
            max_iter_sl = gr.Slider(1, 5, value=3, step=1, label="Max Self-Correction Iterations")
            thread_id_box = gr.Textbox(value="candidate-session-01", label="Thread / Session ID (MemorySaver)")
            if not os.getenv("GROQ_API_KEY"):
                gr.Markdown("> ⚠️ **Warning:** `GROQ_API_KEY` not found in `.env`.")
            gr.Markdown("---")
            gr.Markdown("_State machine compiled via `build_graph()` in `starter.py`._")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("🚀 Workflow Execution"):
                    resume_box = gr.Textbox(
                        label="Candidate Resume (leave blank to use built-in sample)",
                        value=SAMPLE_RESUME if _IMPORT_OK else "",
                        lines=5,
                    )
                    job_box = gr.Textbox(
                        label="Target Job Posting (leave blank to use built-in sample)",
                        value=SAMPLE_JOB if _IMPORT_OK else "",
                        lines=4,
                    )
                    run_btn = gr.Button("▶️ Run LangGraph Workflow", variant="primary")
                    workflow_log = gr.Textbox(label="Execution Log", lines=3, interactive=False)

                    approval_row = gr.Row(visible=False)
                    with approval_row:
                        gr.Markdown("### 🛑 Human-in-the-Loop Approval Required")
                        approve_btn = gr.Button("👍 Approve & Finalize Cover Letter", variant="primary")

                    result_md = gr.Markdown("_Click 'Run LangGraph Workflow' above to begin._")

                with gr.Tab("🔍 Live State Inspector"):
                    gr.Markdown("#### Real-time JSON snapshot from LangGraph `MemorySaver` checkpoint store:")
                    state_json_box = gr.Code(language="json", label="Graph State Snapshot")

    run_btn.click(
        run_workflow,
        inputs=[resume_box, job_box, action_radio, max_iter_sl, thread_id_box],
        outputs=[result_md, state_json_box, approval_row, workflow_log],
    )

    approve_btn.click(
        approve_workflow,
        inputs=[thread_id_box],
        outputs=[result_md, state_json_box, approval_row, workflow_log],
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
