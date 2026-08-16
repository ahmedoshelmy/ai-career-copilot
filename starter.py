"""
Career Copilot — Session 01: Chat Foundation (STARTER)
Module: 01_langchain

Concepts practised:
- ChatPromptTemplate + MessagesPlaceholder
- LCEL pipe operator (|)
- StrOutputParser (streaming)
- with_structured_output (Pydantic structured extraction)
- Message types: SystemMessage, HumanMessage, AIMessage

Fill in every TODO below. The __main__ block will tell you what's next
even when some steps are still unfinished.
"""

import os
import sys
from typing import Generator, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def safe_print(*args, **kwargs):
    """Print that survives Unicode issues on Windows consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))


def get_llm(provider: str = "groq", temperature: float = 0.7):
    """Create a chat model for 'groq' (default) or 'ollama'."""
    if provider == "groq":
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
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not found.")
        return ChatOllama(
            model="gpt-oss:120b-cloud",
            base_url="https://ollama.com",
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
            temperature=temperature,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# Schemas
# ============================================================

class CandidateProfile(BaseModel):
    """Structured profile extracted from the conversation."""
    target_role: str = Field(description="The job title the candidate is targeting, e.g. 'Senior Backend Engineer'")
    years_of_experience: int = Field(description="Estimated total years of relevant work experience (0 if student)")
    top_skills: List[str] = Field(description="Up to 5 key technical or domain skills mentioned")
    current_situation: str = Field(
        description="One-sentence summary: student / employed looking / actively job-hunting / career-changer"
    )


class CareerAdvice(BaseModel):
    """Structured career advice for a specific question."""
    advice: str = Field(description="Concise, actionable advice in 2-3 sentences")
    action_items: List[str] = Field(description="2-4 specific, immediate next steps the candidate can take")
    resources: List[str] = Field(description="1-3 recommended resources (courses, tools, communities)")


SYSTEM_PROMPT = (
    "You are Career Copilot, a friendly and professional AI career coach. "
    "Your goal is to help candidates land their dream job. "
    "Ask about their target role and experience if not provided. "
    "Give specific, actionable advice tailored to their situation. "
    "Keep responses concise — 2-4 sentences unless the user asks for detail."
)

SAMPLE_CONVERSATION = [
    HumanMessage(content="Hi! I'm a junior Python developer with 1 year of experience. I want to become a backend engineer at a fintech company."),
    AIMessage(content="Great to meet you! Fintech backend roles typically require strong Python, REST API design, and some experience with databases like PostgreSQL. With 1 year under your belt, focusing on system design basics and contributing to open source will make your CV stand out. What specific areas do you feel least confident in?"),
    HumanMessage(content="Honestly, system design. I've never done a system design interview before."),
]


# ============================================================
# TODO 1: build_chat_chain
# ============================================================

def build_chat_chain(llm):
    """Build the streaming LCEL chat chain.

    TODO 1:
      - Create a ChatPromptTemplate with:
          * a system message using SYSTEM_PROMPT
          * a MessagesPlaceholder named "history"
          * a human turn using "{question}"
      - Return:  prompt | llm | StrOutputParser()
    """
    raise NotImplementedError("TODO 1: implement build_chat_chain")


# ============================================================
# TODO 2: stream_reply
# ============================================================

def stream_reply(chain, history: List[BaseMessage], question: str) -> Generator[str, None, None]:
    """Stream the copilot's reply token by token.

    TODO 2:
      - Call chain.stream({"history": history, "question": question})
      - yield each chunk (already a string because of StrOutputParser)
    """
    raise NotImplementedError("TODO 2: implement stream_reply")


# ============================================================
# TODO 3: extract_profile
# ============================================================

def extract_profile(llm, conversation: List[BaseMessage]) -> CandidateProfile:
    """Extract a structured CandidateProfile from the conversation so far.

    TODO 3:
      - structured_llm = llm.with_structured_output(CandidateProfile)
      - Build a prompt: system + the conversation messages + a final human
        message asking to extract the profile
      - Return the CandidateProfile
    """
    raise NotImplementedError("TODO 3: implement extract_profile")


# ============================================================
# TODO 4: answer_career_question
# ============================================================

def answer_career_question(llm, role: str, question: str) -> CareerAdvice:
    """Answer a specific career question with structured advice.

    TODO 4:
      - structured_llm = llm.with_structured_output(CareerAdvice)
      - Build a ChatPromptTemplate with system + human containing
        the role context and question
      - chain = prompt | structured_llm
      - Return CareerAdvice
    """
    raise NotImplementedError("TODO 4: implement answer_career_question")


# ============================================================
# TODO 5: run_demo
# ============================================================

def run_demo():
    """Demonstrate the full session-01 Career Copilot.

    TODO 5:
      - llm = get_llm("groq")
      - chain = build_chat_chain(llm)
      - Replay SAMPLE_CONVERSATION to show streaming:
          * For each HumanMessage: call stream_reply(chain, history_so_far, msg.content)
            collect and print the streamed chunks
          * Append human + AI messages to a running history list
      - After the conversation, call extract_profile(llm, history) and print it
      - Call answer_career_question(llm, "Backend Engineer", "How do I prepare for system design interviews?")
        and print the CareerAdvice
    """
    raise NotImplementedError("TODO 5: implement run_demo")


if __name__ == "__main__":
    safe_print("Career Copilot — Session 01: Chat Foundation\n")
    try:
        run_demo()
    except NotImplementedError as e:
        safe_print(f"Not implemented yet -> {e}")
        safe_print("\nComplete the TODOs above and re-run.")
