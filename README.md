# 🧭 AI Career Copilot — Session 02: Resume & Job Ingestion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/VectorStore-Chroma-red.svg)](https://docs.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, end-to-end **AI Career Coach & Job Matcher** powered by LangChain, HuggingFace embeddings, Chroma vector search, and structured Pydantic skill-gap analysis.

---

## 🎯 What's New in Session 02

Session 02 upgrades the copilot from a basic conversational assistant into a full **Document Ingestion & Semantic Matching Engine**:

- 📄 **Multi-Format Resume Ingestion**: Ingest candidate resumes directly from raw text, `.txt`, or `.pdf` documents using `TextLoader` and `PyPDFLoader`.
- 🏢 **Batch Job Description Loading**: Automatically ingest and parse directories of job postings with `DirectoryLoader` and metadata tagging (title, company).
- ✂️ **Smart Text Chunking**: Segment documents into optimal retrieval blocks using `RecursiveCharacterTextSplitter`.
- 🧠 **Serverless Embeddings**: Embed documents into high-dimensional semantic vectors using `HuggingFaceEndpointEmbeddings` (`all-MiniLM-L6-v2`).
- 🗄️ **Chroma Vector Store**: Index and rank job postings by semantic similarity using Chroma vector search (`similarity_search_with_score`).
- 🔍 **Structured Skill-Gap Analysis**: Run deterministic LLM evaluations (`SkillGapAnalysis`) comparing candidate experience against job requirements, highlighting matched skills, missing skills, fit summary, and actionable tips.
- 🖥️ **Interactive Gradio v2 UI**: Features drag-and-drop resume upload, batch job file uploads, interactive ranking bars, and a live coaching chat window.

---

## 🏗️ Architecture

```
Candidate Resume (.pdf / .txt / text)          Job Postings (.txt directory)
              │                                                │
       TextLoader / PyPDFLoader                         DirectoryLoader
              │                                                │
       RecursiveCharacterTextSplitter                   RecursiveCharacterTextSplitter
              │                                                │
       embed_query()                                    HuggingFaceEndpointEmbeddings
              │                                                │
              └────────────────► Chroma Vector Index ◄─────────┘
                                       │
                               similarity_search_with_score()
                                       │
                                       ▼
                              Top-N Ranked Job Matches
                                       │
                                       ▼
                       LLM Structured Skill-Gap Analysis
                      (Matched, Missing, Summary, Action Tips)
```

---

## 🛠️ Step-by-Step Implementation Guide (`starter.py`)

The core logic lives in `starter.py`. Follow the guided `# TODO` items in order:

| Step | Function | Goal |
|---|---|---|
| **TODO 1** | `load_resume(source)` | Load resume from PDF, TXT file, or raw text string into a LangChain `Document`. |
| **TODO 2** | `load_job_postings(directory)` | Load all job `.txt` files from a directory using `DirectoryLoader` with metadata. |
| **TODO 3** | `chunk_documents(docs)` | Chunk documents with `RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)`. |
| **TODO 4** | `build_job_vectorstore(job_docs)` | Embed and index job documents into a Chroma vector store. |
| **TODO 5** | `find_best_matches(vectorstore, resume)` | Compute similarity scores (`1 / (1 + distance) * 100`) and return ranked `JobMatch` list. |
| **TODO 6** | `analyze_skill_gap(llm, resume, job)` | Structured comparison of candidate resume vs. target job returning `SkillGapAnalysis`. |
| **TODO 7** | `run_copilot_v2(resume, jobs_dir)` | Orchestrate the complete end-to-end ingestion and analysis pipeline. |

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

### 3. Configure API Keys
Copy `.env.example` to `.env` and provide your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Fast free inference: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Free embeddings: https://huggingface.co/settings/tokens
HUGGING_FACE_API_KEY=hf_your_huggingface_token_here
```

### 4. Run the Project

#### Interactive Web Application
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser. As you implement TODOs in `starter.py`, the web interface updates dynamically!

#### CLI Hands-On Practice Runner
```bash
python starter.py
```

---

## 📂 Project Structure

```
ai-career-copilot/
├── app.py               # Gradio v2 web interface with resume upload & matching tabs
├── starter.py           # Practice scaffold with guided TODOs 1-7 & data schemas
├── requirements.txt     # Session 02 dependencies (Chroma, HuggingFace, PyPDF, etc.)
├── .env.example         # Environment template with Groq & Hugging Face configs
├── .gitignore           # Git ignore rules for Python, Chroma DB, and cache files
├── LICENSE              # MIT License
└── README.md            # Comprehensive project documentation & architecture guide
```

---

## 🗺️ Project Roadmap

| Stage | Module | Focus | Status |
|---|---|---|:---:|
| **Part 01** | `01_chat_foundation` | Streaming Chat, LCEL Pipelines, Structured Profile Extraction | ✅ Complete |
| **Part 02** | `02_ingestion` | Resume Parsing, Job Description Ingestion, Embeddings & Chroma Matching | 🚀 **Current** |
| **Part 03** | `03_rag` | Grounded Career RAG Assistant with Re-ranking & Citations | 🔜 Upcoming |
| **Part 04** | `04_langgraph` | Stateful Multi-Step Workflows & Human-in-the-Loop Review | 🔜 Upcoming |
| **Part 05** | `05_multiagents` | Specialized Multi-Agent Coaching Crew | 🔜 Upcoming |
| **Part 06** | `06_production_llms` | Production Gateway, Guardrails, Fallbacks & Observability | 🔜 Upcoming |

---

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
- **LLM Inference**: [Groq](https://groq.com/) / [Ollama](https://ollama.com/)
- **Embeddings**: [Hugging Face Endpoint](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [ChromaDB](https://docs.trychroma.com/) (`langchain-chroma`)
- **Document Loaders**: [PyPDF](https://pypdf.readthedocs.io/) & TextLoader
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **UI**: [Gradio](https://gradio.app/)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
