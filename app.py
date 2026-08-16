"""
AI Career Copilot — Interactive Web Application
================================================
Imports all logic from starter.py — as you implement each TODO in starter.py,
the UI reflects your working code immediately.
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ── Import from starter.py ───────────────────────────────────────────────────
try:
    from starter import (
        get_llm,
        build_chat_chain,
        stream_reply,
        extract_profile,
        answer_career_question,
        CandidateProfile,
        CareerAdvice,
        SYSTEM_PROMPT,
        SAMPLE_CONVERSATION,
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


def _profile_md(profile) -> str:
    """Format CandidateProfile as clean Markdown for the sidebar."""
    if profile is None:
        return "_Profile details will appear as you converse with the copilot._"
    skills = ", ".join(profile.top_skills) if profile.top_skills else "—"
    return (
        f"**🎯 Target Role:** {profile.target_role}\n\n"
        f"**⏳ Experience:** {profile.years_of_experience} yr(s)\n\n"
        f"**🛠️ Skills:** {skills}\n\n"
        f"**💼 Situation:** {profile.current_situation}"
    )


# ── Event Handlers ────────────────────────────────────────────────────────────

def chat(user_message: str, history: list, model: str, temperature: float):
    """Handle chat interaction with streaming output and profile extraction."""
    if not _IMPORT_OK:
        yield history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"⚠️ **Import Error:** `{_IMPORT_ERR}`"},
        ], _profile_md(None)
        return

    if not user_message.strip():
        yield history, _profile_md(None)
        return

    try:
        os.environ["GROQ_MODEL"] = model
        llm = get_llm("groq", temperature)
        chain = build_chat_chain(llm)
    except NotImplementedError as e:
        yield history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"⚠️ **TODO Not Implemented Yet:** `{e}`\n\nOpen `starter.py` and implement `build_chat_chain()` to enable chat."},
        ], _profile_md(None)
        return
    except Exception as e:
        yield history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"⚠️ **Error initializing LLM:** {e}\n\nPlease check your `GROQ_API_KEY` in `.env`."},
        ], _profile_md(None)
        return

    lc_history = lc_history_from_gradio(history)

    history = history + [{"role": "user", "content": user_message}]
    history.append({"role": "assistant", "content": ""})

    try:
        for chunk in stream_reply(chain, lc_history, user_message):
            history[-1]["content"] += chunk
            yield history, _profile_md(None)
    except NotImplementedError as e:
        history[-1]["content"] = f"⚠️ **TODO Not Implemented Yet:** `{e}`\n\nOpen `starter.py` and implement `stream_reply()` to enable streaming."
        yield history, _profile_md(None)
        return

    # Extract profile if implemented
    updated_lc = lc_history_from_gradio(history)
    try:
        profile = extract_profile(llm, updated_lc)
    except (NotImplementedError, Exception):
        profile = None

    yield history, _profile_md(profile)


def get_career_advice(question: str, role: str, model: str, temperature: float) -> str:
    """Generate structured career advice for a specific role and query."""
    if not _IMPORT_OK:
        return f"⚠️ **Import Error:** `{_IMPORT_ERR}`"
    if not question.strip():
        return "_Please enter a question above._"
    try:
        os.environ["GROQ_MODEL"] = model
        llm = get_llm("groq", temperature)
        advice = answer_career_question(llm, role or "Software Engineer", question)
        md = f"### 💡 Guidance\n{advice.advice}\n\n"
        md += "### ✅ Action Items\n" + "\n".join(f"- {a}" for a in advice.action_items) + "\n\n"
        md += "### 📚 Recommended Resources\n" + "\n".join(f"- {r}" for r in advice.resources)
        return md
    except NotImplementedError as e:
        return f"⚠️ **TODO Not Implemented Yet:** `{e}`\n\nOpen `starter.py` and implement `answer_career_question()`."
    except Exception as e:
        return f"⚠️ **Error:** {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Career Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧭 AI Career Copilot\n"
        "A streaming career coach powered by LangChain LCEL & structured Pydantic extraction. "
        "Logic is imported from `starter.py`."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("💬 Live Coaching Chat"):
                    chatbot = gr.Chatbot(label="Career Copilot", height=440)
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="e.g. I'm a junior Python developer targeting backend roles in fintech...",
                            label="Your message",
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Conversation")

                with gr.Tab("🎯 Targeted Career Advice"):
                    gr.Markdown("Ask a specific career, interview, or resume question.")
                    adv_role = gr.Textbox(value="Backend Engineer", label="Target Role")
                    adv_question = gr.Textbox(
                        placeholder="e.g. How do I prepare for system design interviews with 1 year of experience?",
                        label="Your Question",
                    )
                    adv_btn = gr.Button("Get Structured Advice", variant="primary")
                    adv_output = gr.Markdown("_Submit a question above to receive tailored advice._")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Candidate Profile")
            profile_md = gr.Markdown(_profile_md(None))
            gr.Markdown("---")
            gr.Markdown("### ⚙️ Model Settings")
            model_dd = gr.Dropdown(GROQ_MODELS, value=GROQ_MODELS[0], label="LLM Model")
            temp_sl = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
            if not os.getenv("GROQ_API_KEY"):
                gr.Markdown("> ⚠️ **Note:** `GROQ_API_KEY` is not set in `.env`.")

    send_btn.click(
        chat,
        inputs=[msg_box, chatbot, model_dd, temp_sl],
        outputs=[chatbot, profile_md],
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, model_dd, temp_sl],
        outputs=[chatbot, profile_md],
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(lambda: ([], _profile_md(None)), outputs=[chatbot, profile_md])

    adv_btn.click(get_career_advice, [adv_question, adv_role, model_dd, temp_sl], adv_output)


if __name__ == "__main__":
    demo.launch(share=False)
