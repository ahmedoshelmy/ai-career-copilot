# 🧭 AI Career Copilot — Session 04: LangGraph Stateful Workflow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-v0.2+-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous, multi-step **AI Career Workflow Engine** built with **LangGraph**, featuring dynamic specialist routing, an automated self-correcting editor loop, human-in-the-loop approval gates, and persistent state checkpointing.

---

## 🎯 What's New in Session 04

Session 04 transitions the copilot from sequential chains into a full **Stateful Graph Agent Architecture**:

- 🔀 **Dynamic Action Routing**: Intelligently inspects the job posting and candidate goals to route to specialist nodes:
  - `tailor_resume`: Rewrites resume bullets targeting key job requirements.
  - `draft_cover_letter`: Writes and refines personalized cover letters.
  - `mock_interview`: Generates 5 tailored technical and behavioral interview questions.
- 🔄 **Self-Correcting Feedback Loop**: An automated `editor_node` evaluates cover letter quality (0.0 to 1.0). If quality is below `0.8`, the graph loops back with constructive feedback to revise the draft up to `max_iterations`.
- 🛑 **Human-in-the-Loop (HitL)**: Uses `interrupt_before=["hitl_approval"]` to safely pause execution and wait for candidate approval before finalizing.
- 💾 **State Checkpointing**: Employs `MemorySaver` to persist state snapshots across unique `thread_id` sessions, allowing seamless workflow resumption.
- 🖥️ **Workflow Dashboard & State Inspector**: Interactive Gradio v4 interface with real-time execution logs, human approval buttons, and live JSON state inspection.

---

## 🏗️ State Machine Architecture

```
                       START
                         │
                         ▼
                   [intake_node]
            (Extracts role, company & action)
                         │
         ┌───────────────┼───────────────┐
         │ (tailor)      │ (cover_letter)│ (mock_interview)
         ▼               ▼               ▼
  [tailor_resume]  [draft_cover]   [mock_interview]
         │               │               │
         │               ▼               │
         │         [editor_node]         │
         │         (Quality Score)       │
         │          ┌────┴────┐          │
         │   quality<0.8   quality>=0.8  │
         │    (Revise)     (Approve)     │
         │       ▲            ▼          │
         │       └────── [hitl_node]     │
         │             (Human Approves)  │
         ▼                    ▼          ▼
        END                  END        END
```

---

## 🛠️ Step-by-Step Implementation Guide (`starter.py`)

All workflow logic is implemented in `starter.py` through guided `# TODO` items:

| Step | Component | Objective |
|---|---|---|
| **TODO 1** | `intake_node(state)` | Parse job posting and extract `action`, `target_role`, and `company` using structured outputs. |
| **TODO 2** | `tailor_resume_node(state)` | Rewrite resume bullet points to emphasize relevant skills and achievements. |
| **TODO 3** | `draft_cover_letter_node(state)` | Draft or revise cover letters, incorporating editor feedback on subsequent iterations. |
| **TODO 4** | `editor_node(state)` | Critique the draft, score quality (0.0–1.0), and approve if score $\ge 0.8$. |
| **TODO 5** | `mock_interview_node(state)` | Generate 5 role-specific behavioral and technical interview questions. |
| **TODO 6** | `route_after_intake(state)` | Conditional edge routing to `"tailor_resume"`, `"draft_cover_letter"`, or `"mock_interview"`. |
| **TODO 7** | `route_after_editor(state)` | Conditional edge returning `"revise"` (loop back) or `"approve"` (proceed to HitL). |
| **TODO 8** | `build_graph()` | Assemble `StateGraph`, define edges/cycles, attach `MemorySaver`, and set `interrupt_before`. |

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ahmedoshelmy/ai-career-copilot.git
cd ai-career-copilot
```

### 2. Set Up Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure API Key
Copy `.env.example` to `.env` and set your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Free fast LLM inference: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

### 4. Run the Workflow

#### Interactive Web Dashboard
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) to test routing, trigger self-correction revisions, and test human approval gates.

#### CLI Practice Runner
```bash
python starter.py
```

---

## 📂 Project Structure

```
ai-career-copilot/
├── app.py               # Gradio v4 workflow dashboard with live state inspector
├── starter.py           # Practice scaffold with CopilotState, TODOs 1-8, and demo runner
├── requirements.txt     # Session 04 dependencies (LangGraph, LangChain, Groq, Gradio)
├── .env.example         # Environment template
├── .gitignore           # Python, cache, and checkpoint ignore rules
├── LICENSE              # MIT License
└── README.md            # Architecture documentation & state machine specification
```

---

## 🗺️ Project Roadmap

| Stage | Module | Architecture Focus | Status |
|---|---|---|:---:|
| **Part 01** | `01_chat_foundation` | Streaming Chat, LCEL Pipelines, Structured Profile Extraction | ✅ Complete |
| **Part 02** | `02_ingestion` | Resume Parsing, Job Description Ingestion, Embeddings & Chroma Matching | ✅ Complete |
| **Part 03** | `03_rag` | Grounded Career RAG Assistant with Re-ranking & Citations | ✅ Complete |
| **Part 04** | `04_langgraph` | Stateful Multi-Step Workflows, Self-Correction & Human-in-the-Loop | 🚀 **Current** |
| **Part 05** | `05_multiagents` | Specialized Multi-Agent Coaching Crew & Supervisor | 🔜 Upcoming |
| **Part 06** | `06_production_llms` | Production Gateway, Guardrails, Fallbacks & Observability | 🔜 Upcoming |

---

## 🛠️ Technology Stack

- **Workflow Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) (`StateGraph`, `MemorySaver`, `interrupt_before`)
- **LLM Framework**: [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-groq`)
- **Inference Engine**: [Groq Cloud](https://groq.com/)
- **Data Validation**: [Pydantic v2](https://docs.pydantic.dev/) & [typing-extensions](https://typing-extensions.readthedocs.io/)
- **UI Dashboard**: [Gradio](https://gradio.app/)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
