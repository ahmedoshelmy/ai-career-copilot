"""
AI Career Copilot — Session 02: Resume & Job Ingestion (STARTER)
================================================================
Concepts practised:
- TextLoader, PyPDFLoader, DirectoryLoader
- RecursiveCharacterTextSplitter
- HuggingFaceEndpointEmbeddings
- Chroma vector store (from_documents, similarity_search_with_score)
- with_structured_output (SkillGapAnalysis)

Fill in every TODO. The __main__ block will tell you what is missing even
when some steps are incomplete.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def safe_print(*args, **kwargs):
    """Print helper handling potential Unicode console issues."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))


# ── LLM / Embeddings Factories ───────────────────────────────────────────────

def get_llm(provider: str = "groq", temperature: float = 0.7):
    """Create a chat model for 'groq' (default) or 'ollama'."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Copy .env.example to .env first.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=api_key,
            temperature=temperature,
            max_retries=5,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not found.")
        return ChatOllama(
            model="gpt-oss:120b-cloud",
            base_url="https://ollama.com",
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
        )
    raise ValueError(f"Unknown provider: {provider}")


def get_embeddings():
    """Create HuggingFace serverless embeddings client."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    api_key = os.getenv("HUGGING_FACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGING_FACE_API_KEY not found. Set it in .env to use embeddings.")
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=api_key,
        model="sentence-transformers/all-MiniLM-L6-v2",
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class JobMatch(BaseModel):
    title: str = Field(description="Job title from metadata")
    company: str = Field(description="Company name from metadata")
    similarity: float = Field(description="Similarity score 0-100")


class SkillGapAnalysis(BaseModel):
    matched_skills: List[str] = Field(description="Skills the resume already shows that the job wants")
    missing_skills: List[str] = Field(description="Skills the job requires that are NOT in the resume")
    fit_summary: str = Field(description="One-paragraph overall fit assessment")
    improvement_tips: List[str] = Field(description="2-3 specific tips to close the skill gap")


class CopilotV2Result(BaseModel):
    resume_chunks: int
    job_count: int
    top_matches: List[JobMatch]
    skill_gap: SkillGapAnalysis


# ── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_RESUME = """
Alex Kim — Software Engineer

Skills: Python, FastAPI, PostgreSQL, Docker, Git, basic React, REST API design.

Experience:
- Built and deployed a FastAPI microservice for a small e-commerce startup (6 months).
- Wrote SQL migrations and optimised slow queries for a PostgreSQL database.
- Containerised services with Docker and Docker Compose.

Education: BSc Computer Science, graduated 2023.
Currently looking for a mid-level backend role in a product company.
"""

SAMPLE_JOBS_DIR_CONTENT = {
    "backend_engineer_stripe.txt": (
        "Job: Backend Software Engineer @ Stripe\n"
        "Requirements: Python (expert), distributed systems, PostgreSQL or MySQL, "
        "REST API design, Docker/Kubernetes. Nice-to-have: Go, Kafka, AWS."
    ),
    "fullstack_shopify.txt": (
        "Job: Full-Stack Engineer @ Shopify\n"
        "Requirements: React, Node.js, Ruby on Rails, PostgreSQL, GraphQL, "
        "strong TypeScript. Backend Python is a plus."
    ),
    "devops_cloudflare.txt": (
        "Job: DevOps Engineer @ Cloudflare\n"
        "Requirements: Kubernetes, Terraform, CI/CD pipelines, Prometheus, "
        "Go or Python scripting, cloud (AWS/GCP)."
    ),
    "ml_engineer_huggingface.txt": (
        "Job: ML Engineer @ HuggingFace\n"
        "Requirements: Python, PyTorch, Transformers library, CUDA, MLOps, "
        "Docker. Experience with LLM fine-tuning or deployment is a strong plus."
    ),
    "backend_django_startup.txt": (
        "Job: Backend Python Engineer @ EarlyStage.io\n"
        "Requirements: Python, Django or FastAPI, PostgreSQL, REST APIs, "
        "Docker, basic AWS (EC2/S3). Team player in a fast-paced startup."
    ),
}


# ============================================================
# TODO 1: load_resume
# ============================================================

def load_resume(source: str) -> Document:
    """Load the resume as a single LangChain Document.

    TODO 1:
      - If `source` is a file path that exists:
          * If it ends with '.pdf': use PyPDFLoader(source).load() and join pages
          * Otherwise: use TextLoader(source).load()[0]
      - If `source` is raw text (not a path): wrap it in Document(page_content=source, metadata={"source": "pasted_resume"})
      - Return a single Document with metadata["source"] set.
    """
    raise NotImplementedError("TODO 1: implement load_resume")


# ============================================================
# TODO 2: load_job_postings
# ============================================================

def load_job_postings(directory: str) -> List[Document]:
    """Load all .txt job description files from a directory.

    TODO 2:
      - Use DirectoryLoader(directory, glob="**/*.txt") to load all files.
      - Set metadata["title"] and ["company"] from the filename
        (e.g. "backend_engineer_stripe.txt" -> title="Backend Engineer", company="Stripe").
      - Return the list of Documents.
    """
    raise NotImplementedError("TODO 2: implement load_job_postings")


# ============================================================
# TODO 3: chunk_documents
# ============================================================

def chunk_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Split documents into chunks.

    TODO 3:
      - Create RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      - Return splitter.split_documents(docs)
    """
    raise NotImplementedError("TODO 3: implement chunk_documents")


# ============================================================
# TODO 4: build_job_vectorstore
# ============================================================

def build_job_vectorstore(job_docs: List[Document], persist_directory: str):
    """Embed and index job documents in Chroma.

    TODO 4:
      - embeddings = get_embeddings()
      - Use Chroma.from_documents(documents=job_docs, embedding=embeddings,
          persist_directory=persist_directory, collection_name="career_copilot_jobs")
      - Return the Chroma instance.
    """
    raise NotImplementedError("TODO 4: implement build_job_vectorstore")


# ============================================================
# TODO 5: find_best_matches
# ============================================================

def find_best_matches(vectorstore, resume_text: str, k: int = 3) -> List[JobMatch]:
    """Rank job postings by semantic similarity to the resume.

    TODO 5:
      - results = vectorstore.similarity_search_with_score(resume_text, k=k)
      - For each (doc, distance): similarity = round(1 / (1 + distance) * 100, 1)
      - Build JobMatch objects using doc.metadata for title/company
      - Sort by similarity descending and return the list.
    """
    raise NotImplementedError("TODO 5: implement find_best_matches")


# ============================================================
# TODO 6: analyze_skill_gap
# ============================================================

def analyze_skill_gap(llm, resume_text: str, job_title: str, job_description: str) -> SkillGapAnalysis:
    """Compare resume against one job description and return structured gap analysis.

    TODO 6:
      - structured_llm = llm.with_structured_output(SkillGapAnalysis)
      - Build a ChatPromptTemplate with a system message (career coach persona)
        and a human message with resume + job details
      - Return SkillGapAnalysis
    """
    raise NotImplementedError("TODO 6: implement analyze_skill_gap")


# ============================================================
# TODO 7: run_copilot_v2
# ============================================================

def run_copilot_v2(resume_source: str, jobs_dir: str) -> CopilotV2Result:
    """Orchestrate the full session-02 ingestion pipeline.

    TODO 7:
      - Load resume: resume_doc = load_resume(resume_source)
      - Load & chunk jobs: job_docs = chunk_documents(load_job_postings(jobs_dir))
      - Build vectorstore
      - Find top matches
      - Analyze gap for the top match
      - Return CopilotV2Result
    """
    raise NotImplementedError("TODO 7: implement run_copilot_v2")


def print_result(result: CopilotV2Result) -> None:
    safe_print("=" * 60)
    safe_print(f"Resume indexed : {result.resume_chunks} chunk(s)")
    safe_print(f"Jobs indexed   : {result.job_count}")
    safe_print("\nTOP JOB MATCHES")
    for i, m in enumerate(result.top_matches, 1):
        safe_print(f"  {i}. {m.title} @ {m.company}  ({m.similarity}% match)")
    safe_print("\nSKILL GAP vs. Top Match")
    safe_print(f"  Matched  : {', '.join(result.skill_gap.matched_skills)}")
    safe_print(f"  Missing  : {', '.join(result.skill_gap.missing_skills)}")
    safe_print(f"  Summary  : {result.skill_gap.fit_summary}")
    safe_print("  Tips:")
    for tip in result.skill_gap.improvement_tips:
        safe_print(f"    - {tip}")
    safe_print("=" * 60)


if __name__ == "__main__":
    safe_print("Career Copilot — Session 02: Resume & Job Ingestion\n")

    # Write sample job files to a temp dir for the demo
    jobs_dir = tempfile.mkdtemp(prefix="cc_jobs_")
    for fname, content in SAMPLE_JOBS_DIR_CONTENT.items():
        Path(jobs_dir, fname).write_text(content, encoding="utf-8")

    try:
        result = run_copilot_v2(SAMPLE_RESUME, jobs_dir)
        print_result(result)
    except NotImplementedError as e:
        safe_print(f"Not implemented yet -> {e}")
        safe_print("\nComplete the TODOs above in starter.py and re-run.")
