# 🔍 Agentic Research Assistant

A simple **agentic AI research assistant** built from scratch. Ask any question — an AI agent decides which tool to use (web search, GitHub repository lookup, or arXiv paper search), calls the appropriate tool, and synthesizes a final answer. The UI shows the agent's full reasoning trace so you can see it "thinking."

## Architecture

```
┌──────────────┐       POST /ask        ┌──────────────────┐
│              │ ─────────────────────▶  │                  │
│  Streamlit   │   { question: "..." }  │   FastAPI         │
│  Frontend    │                        │   Backend         │
│              │ ◀─────────────────────  │                  │
└──────────────┘  { answer, steps[] }   └────────┬─────────┘
                                                 │
                                                 │ invokes
                                                 ▼
                                        ┌──────────────────┐
                                        │  LangGraph ReAct  │
                                        │  Agent            │
                                        │                  │
                                        │  LLM: Llama 3.3  │
                                        │  (via Groq)      │
                                        └────────┬─────────┘
                                                 │
                                          decides │ to call
                                                 ▼
                               ┌─────────────────┼─────────────────┐
                               ▼                 ▼                 ▼
                      ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                      │  DuckDuckGo      │ │  GitHub Repo     │ │  arXiv Paper     │
                      │  Web Search      │ │  Info Tool       │ │  Search Tool     │
                      │  (no API key)    │ │  (Public API)    │ │  (arXiv API)     │
                      └──────────────────┘ └──────────────────┘ └──────────────────┘
```

**Flow:**
1. User types a question in the Streamlit UI
2. Streamlit sends a POST request to FastAPI's `/ask` endpoint
3. FastAPI invokes the LangGraph **ReAct agent**
4. The agent reasons about the question:
   - If it can answer from its own knowledge → responds directly
   - If it needs current/factual web info → calls `web_search` (DuckDuckGo)
   - If it needs repository details (stars, forks, links) → calls `github_repo_info` (GitHub Public API)
   - If it needs academic papers, scientific articles, or research publications → calls `arxiv`
   - It can call tools multiple times if needed
5. The final answer + reasoning trace (tool calls) are returned to Streamlit
6. The UI displays both the answer and each tool call the agent made

## Agent Tools

The agent is equipped with the following tools:

1. **`web_search`** (DuckDuckGo): Searches the web for current events, facts, articles, and general research without requiring an API key.
   - *Example question:* "What are the latest breakthroughs in fusion energy?"
2. **`github_repo_info`** (GitHub REST API): Fetches real-time repository metadata including star count, forks, open issues, description, primary language, license, and repository URL without requiring authentication.
   - *Example questions:*
     - "How many stars does langchain-ai/langchain have?"
     - "Give me the link to facebook/react"
     - "What is the primary language and description of tiangolo/fastapi?"
3. **`arxiv`** (arXiv.org REST API): Searches arXiv for scientific papers, academic preprints, and research publications across physics, computer science, mathematics, quantitative biology, and related fields.
   - *Example questions:*
     - "Find recent papers on retrieval augmented generation"
     - "Summarize a paper about LLM agents"

## Tech Stack

| Component        | Technology                                       |
|-----------------|--------------------------------------------------|
| Backend          | FastAPI + Uvicorn                                |
| Agent Framework  | LangGraph (`create_react_agent`)                 |
| LLM              | Llama 3.3 70B via Groq (ChatGroq)               |
| Tools            | DuckDuckGo Search, GitHub REST API, arXiv API    |
| Frontend         | Streamlit                                        |
| Env Management   | python-dotenv                                    |

## Project Structure

```
agentic-research-assistant/
├── main.py              # FastAPI app with /ask endpoint
├── agent.py             # LangGraph agent (LLM + tools + create_react_agent)
├── streamlit_app.py     # Streamlit UI (input, answer, reasoning trace)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .gitignore           # Excludes .env and Python artifacts
└── README.md            # This file
```

## Setup & Run

### 1. Clone & enter the project

```bash
cd agentic-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and paste your Groq API key
# Get a free key at: https://console.groq.com/keys
```

### 5. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

The API will be live at `http://localhost:8000`. You can test it at `http://localhost:8000/docs` (Swagger UI).

### 6. Start the Streamlit frontend (new terminal)

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`.

## Usage

1. Open the Streamlit UI in your browser
2. Type a question (e.g., "What are the latest breakthroughs in fusion energy?")
3. Click **Ask the Agent**
4. See the agent's answer and expand the reasoning trace to see which web searches it performed

## API Reference

### `POST /ask`

**Request:**
```json
{
  "question": "What is the capital of France?"
}
```

**Response:**
```json
{
  "answer": "The capital of France is Paris...",
  "steps": [
    {
      "tool": "web_search",
      "query": {"query": "capital of France"},
      "result": "[snippet: Paris is the capital...]"
    }
  ]
}
```

If the agent answers without searching, `steps` will be an empty list.

## License

MIT
