"""
main.py — FastAPI Backend

Exposes a single POST endpoint /ask that:
1. Receives a user question (JSON body)
2. Passes it to the LangGraph ReAct agent
3. Returns the agent's answer + reasoning trace (tool calls made)

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Research Assistant",
    description="A research assistant powered by LangGraph ReAct agent with web search.",
    version="1.0.0",
)


# ── Request / Response Schemas ─────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    """Schema for the incoming question."""
    question: str


class ToolStep(BaseModel):
    """Schema for a single reasoning step (one tool call)."""
    tool: str       # Name of the tool called (e.g., "web_search")
    query: dict     # Arguments passed to the tool
    result: str     # Raw result returned by the tool


class AnswerResponse(BaseModel):
    """Schema for the full response sent back to the frontend."""
    answer: str           # The agent's final natural language answer
    steps: list[ToolStep] # Trace of tool calls made during reasoning


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Simple health check — useful for verifying the server is running."""
    return {"status": "ok", "message": "Agentic Research Assistant is running."}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint: takes a question, runs the agent, returns answer + reasoning.

    The agent will:
    - Analyze the question
    - Decide if it needs to search the web (calls DuckDuckGo if so)
    - Synthesize a final answer from its knowledge + search results
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Run the LangGraph agent and get the structured result
        result = run_agent(request.question)
        return AnswerResponse(
            answer=result["answer"],
            steps=result["steps"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
