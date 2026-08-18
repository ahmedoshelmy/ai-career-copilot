"""
AI Career Copilot — Session 04: LangGraph Stateful Workflow (STARTER)
======================================================================
Concepts practised:
- StateGraph + TypedDict state with typed fields
- add_conditional_edges (Intake & Editor Routing)
- Self-correcting feedback loop (editor → writer cycle)
- Human-in-the-loop (interrupt_before approval node)
- MemorySaver checkpointing & thread session resume

Fill in every TODO. The __main__ block will tell you what is missing.
"""

import os
import sys
from typing import Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def safe_print(*args, **kwargs):
    """Print helper handling potential Unicode console issues."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))


def get_llm(provider: str = "groq", temperature: float = 0.7):
    """Create chat LLM instance."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in .env first.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=api_key,
            temperature=temperature,
            max_retries=5,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ── State Definition ──────────────────────────────────────────────────────────

class CopilotState(TypedDict):
    # Inputs
    resume: str
    job_posting: str
    desired_action: str           # "tailor_resume" | "cover_letter" | "mock_interview"
    target_role: str
    company: str

    # Outputs
    tailored_bullets: str         # Rewritten resume bullets
    cover_letter_draft: str       # Generated cover letter text
    editor_feedback: str          # Editor critique / suggestions
    quality_score: float          # 0.0–1.0 score
    approved: bool                # Editor or human approved
    interview_questions: str      # Numbered list of interview questions

    # Control
    iteration: int
    max_iterations: int
    thread_id: str


# ── Pydantic Schemas for Structured Outputs ───────────────────────────────────

class IntakeDecision(BaseModel):
    action: Literal["tailor_resume", "cover_letter", "mock_interview"] = Field(
        description="What the candidate wants the copilot to do"
    )
    target_role: str = Field(description="The job title extracted from the posting")
    company: str = Field(description="Company name extracted from the posting")


class EditorReview(BaseModel):
    quality_score: float = Field(description="Quality score between 0.0 and 1.0")
    feedback: str = Field(description="Specific, actionable feedback for the writer")
    approved: bool = Field(description="True if quality_score >= 0.8")


# ── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_RESUME = """
Alex Kim — Software Engineer
Skills: Python, FastAPI, PostgreSQL, Docker, Git, REST API design.
Experience:
- Built a FastAPI microservice handling 10k req/day.
- Optimised PostgreSQL queries, cutting p99 latency by 40%.
- Containerised services with Docker and GitHub Actions CI.
Education: BSc Computer Science, 2023.
"""

SAMPLE_JOB = """
Backend Software Engineer @ Stripe
We build the economic infrastructure of the internet.
Requirements: expert Python, distributed systems, PostgreSQL or MySQL at scale,
clean REST API design, Docker/Kubernetes. Nice-to-have: Go, Kafka, AWS.
"""


# ============================================================
# TODO 1: intake_node
# ============================================================

def intake_node(state: CopilotState) -> dict:
    """Read the job posting and desired action; return routing and extracted fields.

    TODO 1:
      - structured_llm = get_llm().with_structured_output(IntakeDecision)
      - Build a prompt:
          system: "Extract the action, target role, and company from the user's request."
          human: "Job posting:\n{job_posting}\n\nDesired action: {desired_action}"
      - decision = structured_llm.invoke(...)
      - Return {"desired_action": decision.action,
                "target_role": decision.target_role,
                "company": decision.company}
    """
    raise NotImplementedError("TODO 1: implement intake_node")


# ============================================================
# TODO 2: tailor_resume_node
# ============================================================

def tailor_resume_node(state: CopilotState) -> dict:
    """Rewrite resume bullet points to target the job description.

    TODO 2:
      - Call llm.invoke([SystemMessage(...), HumanMessage(...)])
      - System: "You are a resume coach. Rewrite the bullet points to highlight
                 experience most relevant to the job. Keep the same factual content
                 but sharpen the language and add relevant keywords."
      - Human: Resume:\n{resume}\n\nJob:\n{job_posting}
      - Return {"tailored_bullets": response.content}
    """
    raise NotImplementedError("TODO 2: implement tailor_resume_node")


# ============================================================
# TODO 3: draft_cover_letter_node
# ============================================================

def draft_cover_letter_node(state: CopilotState) -> dict:
    """Write (or revise) a personalised cover letter.

    TODO 3:
      - If state["iteration"] > 0 and state["editor_feedback"]:
          include a revision note asking to address the editor feedback.
      - Call llm.invoke([SystemMessage(...), HumanMessage(...)])
      - System: cover letter writer persona
      - Human: Resume + Job posting + (optional revision note)
      - Return {"cover_letter_draft": response.content,
                "iteration": state.get("iteration", 0) + 1}
    """
    raise NotImplementedError("TODO 3: implement draft_cover_letter_node")


# ============================================================
# TODO 4: editor_node
# ============================================================

def editor_node(state: CopilotState) -> dict:
    """Score the cover letter. Set approved=True if quality_score >= 0.8.

    TODO 4:
      - review_llm = get_llm(temperature=0.2).with_structured_output(EditorReview)
      - System: strict editor persona, score 0-1, approve if >= 0.8
      - Human: cover_letter_draft + iteration info
      - review = review_llm.invoke(...)
      - Return {"quality_score": review.quality_score,
                "editor_feedback": review.feedback,
                "approved": review.approved}
    """
    raise NotImplementedError("TODO 4: implement editor_node")


# ============================================================
# TODO 5: mock_interview_node
# ============================================================

def mock_interview_node(state: CopilotState) -> dict:
    """Generate 5 role-specific interview questions.

    TODO 5:
      - Call llm.invoke with a prompt asking for 5 numbered behavioural +
        technical questions tailored to the job posting and the candidate's resume.
      - Return {"interview_questions": response.content}
    """
    raise NotImplementedError("TODO 5: implement mock_interview_node")


# ============================================================
# TODO 6: route_after_intake
# ============================================================

def route_after_intake(state: CopilotState) -> Literal["tailor_resume", "cover_letter", "mock_interview"]:
    """Return which specialist node to route to.

    TODO 6:
      - Read state["desired_action"] and return the matching literal.
      - Default to "cover_letter" if unrecognised.
    """
    raise NotImplementedError("TODO 6: implement route_after_intake")


# ============================================================
# TODO 7: route_after_editor
# ============================================================

def route_after_editor(state: CopilotState) -> Literal["revise", "approve"]:
    """Loop back to writer or proceed to human approval.

    TODO 7:
      - If not state["approved"] AND state.get("iteration", 0) < state.get("max_iterations", 3):
            return "revise"
      - Else: return "approve"
    """
    raise NotImplementedError("TODO 7: implement route_after_editor")


# ============================================================
# TODO 8: build_graph
# ============================================================

def build_graph():
    """Wire the full Career Copilot LangGraph state machine.

    TODO 8:
      - from langgraph.graph import StateGraph, START, END
      - from langgraph.checkpoint.memory import MemorySaver
      - Create workflow = StateGraph(CopilotState)
      - Add nodes: "intake", "tailor_resume", "draft_cover_letter",
                   "editor", "mock_interview", "hitl_approval"
      - Add edges:
          START → "intake"
          "intake" --conditional--> route_after_intake
              {"tailor_resume": "tailor_resume",
               "cover_letter": "draft_cover_letter",
               "mock_interview": "mock_interview"}
          "draft_cover_letter" → "editor"
          "editor" --conditional--> route_after_editor
              {"revise": "draft_cover_letter", "approve": "hitl_approval"}
          "tailor_resume" → END
          "mock_interview" → END
          "hitl_approval" → END
      - Compile with checkpointer=MemorySaver(), interrupt_before=["hitl_approval"]
      - Return the compiled graph
    """
    raise NotImplementedError("TODO 8: implement build_graph")


def run_demo():
    safe_print("=" * 60)
    safe_print("CAREER COPILOT v4 — LangGraph Stateful Workflow Demo")
    safe_print("=" * 60)

    app = build_graph()
    config = {"configurable": {"thread_id": "demo-cover-letter"}}

    initial_state: CopilotState = {
        "resume": SAMPLE_RESUME,
        "job_posting": SAMPLE_JOB,
        "desired_action": "cover_letter",
        "target_role": "",
        "company": "",
        "tailored_bullets": "",
        "cover_letter_draft": "",
        "editor_feedback": "",
        "quality_score": 0.0,
        "approved": False,
        "interview_questions": "",
        "iteration": 0,
        "max_iterations": 3,
        "thread_id": "demo-cover-letter",
    }

    safe_print("\n[1] Running workflow (will pause at human approval)...")
    for event in app.stream(initial_state, config, stream_mode="values"):
        if event.get("cover_letter_draft"):
            safe_print(f"  Draft ready — iteration {event.get('iteration', 0)}, "
                       f"quality {event.get('quality_score', 0):.2f}")

    state = app.get_state(config)
    if state.next:
        safe_print(f"\n[2] Workflow paused at: {state.next}")
        safe_print("    Simulating human approval...")
        app.invoke(None, config)  # resume with no update = approved

    final = app.get_state(config).values
    safe_print(f"\n[3] Final cover letter (quality {final.get('quality_score', 0):.2f}):")
    safe_print(final.get("cover_letter_draft", "(none)"))

    # Quick demo of mock interview mode
    safe_print("\n" + "=" * 60)
    safe_print("Mock Interview mode")
    safe_print("=" * 60)
    config2 = {"configurable": {"thread_id": "demo-interview"}}
    initial2 = {**initial_state, "desired_action": "mock_interview", "thread_id": "demo-interview"}
    result2 = app.invoke(initial2, config2)
    safe_print(result2.get("interview_questions", "(none)"))


if __name__ == "__main__":
    safe_print("Career Copilot — Session 04: LangGraph Workflow\n")
    try:
        run_demo()
    except NotImplementedError as e:
        safe_print(f"Not implemented yet -> {e}")
        safe_print("\nComplete the TODOs above in starter.py and re-run.")
