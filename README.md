# 🧭 AI Career Copilot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, real-time **AI Career Coach & Copilot** built with LangChain Expression Language (LCEL), Pydantic structured output validation, and an interactive Gradio web interface.

---

## 🎯 Features & Architecture

- ⚡ **Real-Time Token Streaming**: Low-latency token streaming using LCEL chains (`chain.stream()`).
- 📋 **Dynamic Profile Extraction**: Automatically parses conversation history into structured `CandidateProfile` Pydantic models (target role, years of experience, top skills, current situation).
- 💡 **Targeted Career Advisory Engine**: Dedicated structured chain generating actionable advice (`CareerAdvice`) with step-by-step action items and recommended resources.
- 🖥️ **Interactive Gradio Web UI**: Clean interface with live streaming chat, dynamic candidate profile card, and model settings.
- 🔌 **Multi-Provider Support**: Default support for ultra-fast Groq models, with support for local/cloud Ollama models.

```
User Message
     │
     ▼
ChatPromptTemplate (System Coach Persona + Conversational History)
     │
     ▼
ChatGroq / ChatOllama (Token Streaming)
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
StrOutputParser (Live Streamed Response)     with_structured_output(CandidateProfile)
     │                                         │
     ▼                                         ▼
Gradio Chat UI                            Candidate Profile Sidebar Card
```

---

## 🛠️ Step-by-Step Build Guide (`starter.py`)

The core logic lives in `starter.py`. Follow the guided `# TODO` comments to build each component:

1. **`build_chat_chain(llm)`**: Compose `system_prompt | llm | StrOutputParser()`.
2. **`stream_reply(chain, history, question)`**: Stream output tokens progressively from `chain.stream()`.
3. **`extract_profile(llm, conversation)`**: Use `llm.with_structured_output(CandidateProfile)` to extract key profile data.
4. **`answer_career_question(llm, role, question)`**: Build a specialized one-shot chain returning structured `CareerAdvice`.
5. **`run_demo()`**: Connect all components for a simulated multi-turn career coaching interview.

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ahmedoshelmy/ai-career-copilot.git
cd ai-career-copilot
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your Groq API key (free at [console.groq.com](https://console.groq.com/keys)):

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

### 4. Run the Project

#### Interactive Gradio Web App
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser. As you complete TODOs in `starter.py`, reload the app to see your features come alive!

#### CLI Hands-On Runner
```bash
python starter.py
```

---

## 📂 Project Structure

```
ai-career-copilot/
├── app.py               # Interactive Gradio web interface (imports from starter.py)
├── starter.py           # Practice scaffold with guided TODOs & core LCEL logic
├── requirements.txt     # Minimal, necessary dependencies
├── .env.example         # Environment configuration template
├── .gitignore           # Git ignore rules for Python & env files
├── LICENSE              # MIT License
└── README.md            # Documentation & step-by-step guide
```

---

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://python.langchain.com/) (`langchain`, `langchain-core`, `langchain-groq`, `langchain-ollama`)
- **LLM Inference**: [Groq](https://groq.com/) / [Ollama](https://ollama.com/)
- **Data Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **Web UI**: [Gradio](https://gradio.app/)
- **Environment**: [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
