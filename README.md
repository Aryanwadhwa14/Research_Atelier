#  Agentic Research Assistant

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

## Conversational Memory

The agent now supports **persistent conversation history per browser session** using LangGraph's `MemorySaver` checkpoint system.

### How It Works

- Each Streamlit browser session automatically gets a unique `thread_id` stored in `st.session_state`
- This `thread_id` is sent with every request to the FastAPI backend
- The LangGraph agent uses this `thread_id` to load and save conversation history
- Follow-up questions can reference previous answers, and the agent understands the context

### Example: Multi-Turn Conversation

**Turn 1 — User asks:**
```
What's the latest Python version?
```

**Agent responds:**
```
As of my latest knowledge update, Python 3.13 is the latest stable release...
```

**Turn 2 — User asks a follow-up (same browser session):**
```
Is it backward compatible with 3.11 code?
```

**Agent understands context:**
```
Python 3.13 maintains backward compatibility with Python 3.11. Most standard
library changes and deprecations are managed carefully to avoid breaking existing code...
```
The agent understood that "it" refers to "Python 3.13" from the previous turn, demonstrating that conversation history is retained.

### Technical Details

- **Backend:** FastAPI accepts an optional `thread_id` in the request. If not provided, a new UUID is generated.
- **Frontend:** Streamlit initializes `st.session_state.thread_id` on first page load and includes it in all requests.
- **Checkpointing:** The LangGraph agent uses `MemorySaver()` to persist the conversation state, automatically loading prior messages for the same `thread_id`.

### Resetting the Conversation

To start a new conversation (clearing history), simply refresh the browser page. A new `thread_id` will be generated automatically.

## Tech Stack

| Component        | Technology                                       |
|-----------------|--------------------------------------------------|
| Backend          | FastAPI + Uvicorn                                |
| Agent Framework  | LangGraph (`create_react_agent`)                 |
| LLM              | Llama 3.3 70B via Groq (ChatGroq)               |
| Tools            | DuckDuckGo Search, GitHub REST API, arXiv API    |
| Frontend         | Static HTML/CSS/JS served by FastAPI; Streamlit fallback |
| Env Management   | python-dotenv                                    |

## Project Structure

```
agentic-research-assistant/
├── main.py              # FastAPI app with /ask endpoint
├── agent.py             # LangGraph agent (LLM + tools + create_react_agent)
├── streamlit_app.py     # Streamlit UI (input, answer, reasoning trace)
├── static/              # Browser UI (index.html, styles.css, app.js)
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

### 5. Start the FastAPI backend and browser UI

```bash
uvicorn main:app --reload
```

The browser UI and API will be live at `http://localhost:8000`. API docs are at `http://localhost:8000/docs` (Swagger UI), and the health check is `http://localhost:8000/health`.

The browser UI includes a research chat (`POST /ask`) and a Literature Review Generator (`POST /literature-review`). The latter ranks arXiv candidates by topic-term overlap in titles and abstracts, then selects up to five records. The generated review separates abstract evidence from interpretation and labels gaps as candidates limited by the small, search-dependent sample. Citation IDs refer to the returned paper records; this checks reference mapping, not whether every sentence is supported by its citation. GitHub results are related topic-search repositories unless a paper-to-repository link is explicitly evidenced. The result can be downloaded as Markdown. If synthesis fails validation, the fallback presents retrieved abstracts and metadata without inventing comparisons or gaps.

Run the offline literature-review tests with:

```bash
python -m unittest discover -s tests
```

### 6. Start the Streamlit frontend (new terminal)

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`.

## Streamlit Fallback

The original Streamlit interface remains available in a separate terminal:

```bash
streamlit run streamlit_app.py
```

## Usage

1. Open `http://localhost:8000` in your browser (or use the Streamlit fallback)
2. Type a question (e.g., "What are the latest breakthroughs in fusion energy?")
3. Click **Ask the Agent**
4. See the agent's answer and expand the reasoning trace to see which web searches it performed
