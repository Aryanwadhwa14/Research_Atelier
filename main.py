"""
main.py — FastAPI Backend

Exposes a single POST endpoint /ask that:
1. Receives a user question and optional thread_id (JSON body)
2. Passes them to the LangGraph ReAct agent
3. Returns the agent's answer + reasoning trace (tool calls made)

The thread_id enables conversational memory: the same thread_id will retain
conversation history, allowing the agent to understand follow-up questions.

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import run_agent
import uuid
from typing import Optional

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Research Assistant",
    description="A research assistant powered by LangGraph ReAct agent with web search.",
    version="1.0.0",
)


# ── Request / Response Schemas ─────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    """Schema for the incoming question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The user's natural language question or query (3-5000 characters)",
        example="What are the latest breakthroughs in quantum computing?"
    )
    thread_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=256,
        description="Optional conversation session identifier. If not provided, a new UUID is auto-generated. Use the same thread_id to maintain conversation history across multiple turns. (1-256 characters)",
        example="550e8400-e29b-41d4-a716-446655440000"
    )

    class Config:
        """Pydantic model configuration."""
        str_strip_whitespace = True  # Automatically strip whitespace from string fields


class ToolStep(BaseModel):
    """Schema for a single reasoning step (one tool call)."""
    tool: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the tool invoked (e.g., 'web_search', 'github_repo_info', 'arxiv_tool') (1-100 characters)",
        example="web_search"
    )
    query: dict = Field(
        ...,
        min_items=1,
        description="Arguments passed to the tool (must contain at least one key-value pair)",
        example={"query": "latest Python features"}
    )
    result: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Raw result returned by the tool (1-50000 characters)",
        example="Python 3.13 introduces new features such as..."
    )

    class Config:
        """Pydantic model configuration."""
        str_strip_whitespace = True


class AnswerResponse(BaseModel):
    """Schema for the full response sent back to the frontend."""
    answer: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The agent's final natural language answer synthesized from tools and knowledge (1-50000 characters)",
        example="Python 3.13 is the latest stable release..."
    )
    steps: list[ToolStep] = Field(
        ...,
        min_items=0,
        max_items=50,
        description="Trace of tool calls made during reasoning, showing the agent's thought process (0-50 steps)",
        example=[]
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="The conversation session identifier used for memory/checkpointing (1-256 characters)",
        example="550e8400-e29b-41d4-a716-446655440000"
    )

    class Config:
        """Pydantic model configuration."""
        str_strip_whitespace = True


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Simple health check — useful for verifying the server is running."""
    return {"status": "ok", "message": "Agentic Research Assistant is running."}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint: takes a question and optional thread_id, runs the agent, returns answer + reasoning.

    The agent will:
    - Analyze the question
    - Recall previous messages in the conversation (if thread_id matches a prior session)
    - Decide if it needs to search the web (calls DuckDuckGo if so)
    - Synthesize a final answer from its knowledge + search results

    Args:
        request.question: The user's question (required)
        request.thread_id: Optional session identifier. If not provided, a new UUID is generated.
                          Use the same thread_id to maintain conversation history across multiple turns.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Generate a new thread_id if not provided (ensures each new session has its own history)
    thread_id = request.thread_id if request.thread_id else str(uuid.uuid4())

    try:
        # Run the LangGraph agent with the thread_id for memory/checkpointing
        result = run_agent(request.question, thread_id)
        return AnswerResponse(
            answer=result["answer"],
            steps=result["steps"],
            thread_id=thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
