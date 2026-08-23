"""
agent.py — LangGraph ReAct Agent Setup

This module configures:
1. The LLM (ChatGroq with Llama 3.3 70B)
2. The tools (DuckDuckGo web search, GitHub repo info, arXiv paper search)
3. The agent (LangGraph's create_react_agent, which implements the ReAct pattern:
   Reason → Act → Observe → Repeat until the agent has enough info to answer)
"""

import os
import arxiv
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file (picks up GROQ_API_KEY)
load_dotenv()

# ── LLM Setup ──────────────────────────────────────────────────────────────────
# ChatGroq wraps the Groq API. We use openai/gpt-oss-20b for fast inference,
# strong reasoning, and clean tool-calling output.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Tool Setup ─────────────────────────────────────────────────────────────────
# DuckDuckGoSearchResults returns search snippets from the web.
# No API key needed — it scrapes DuckDuckGo directly.
search_tool = DuckDuckGoSearchResults(
    name="web_search",
    max_results=3,
    description=(
        "Search the web using DuckDuckGo. Use this tool when you need to find "
        "current information, facts, or data that you don't already know. "
        "Input should be a search query string."
    ),
)


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for scientific papers, academic preprints, and research publications across physics, computer science, mathematics, quantitative biology, quantitative finance, statistics, electrical engineering, and economics.
    Use this tool when you need to find academic papers, scientific articles, or research publications.
    Input should be a search query string (e.g. 'retrieval augmented generation' or 'LLM reasoning agents')."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))
        if not results:
            return f"No arXiv papers found for query: '{query}'."

        formatted_papers = []
        for i, paper in enumerate(results, 1):
            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
            published = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown date"
            summary = paper.summary.replace("\n", " ")
            if len(summary) > 400:
                summary = summary[:400] + "..."

            formatted_papers.append(
                f"**Paper {i}: [{paper.title}]({paper.entry_id})**\n"
                f"- **Authors:** {authors}\n"
                f"- **Published:** {published}\n"
                f"- **PDF Link:** {paper.pdf_url}\n"
                f"- **Summary:** {summary}"
            )

        return "\n\n".join(formatted_papers)
    except Exception as e:
        return f"Error fetching arXiv papers: {str(e)}"


@tool
def github_repo_info(repo: str) -> str:
    """Fetch information about a GitHub repository using the public GitHub API (no authentication required).
    Use this tool when you need details about a GitHub repository, such as star count, forks, open issues, description, language, license, or repository URL.
    Input should be the repository identifier in 'owner/repo' format (e.g. 'langchain-ai/langchain' or 'facebook/react')."""
    repo_clean = repo.strip().rstrip("/")
    if "github.com/" in repo_clean:
        repo_clean = repo_clean.split("github.com/")[-1]

    parts = [p for p in repo_clean.split("/") if p]
    if len(parts) < 2:
        return f"Error: Invalid repository format '{repo}'. Please provide the repository in 'owner/repo' format (e.g. 'owner/repo')."

    owner, repo_name = parts[0], parts[1]
    url = f"https://api.github.com/repos/{owner}/{repo_name}"

    try:
        response = requests.get(
            url,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Agentic-Research-Assistant",
            },
            timeout=10,
        )
        if response.status_code == 404:
            return f"Repository '{owner}/{repo_name}' was not found on GitHub."
        elif response.status_code != 200:
            return f"GitHub API returned status {response.status_code}: {response.text}"

        data = response.json()
        name = data.get("full_name", f"{owner}/{repo_name}")
        description = data.get("description") or "No description provided."
        stars = data.get("stargazers_count", 0)
        forks = data.get("forks_count", 0)
        open_issues = data.get("open_issues_count", 0)
        language = data.get("language") or "Not specified"
        html_url = data.get("html_url", f"https://github.com/{owner}/{repo_name}")
        license_info = data.get("license")
        license_name = license_info.get("name") if license_info else "No license"

        return (
            f"**GitHub Repository:** [{name}]({html_url})\n"
            f"- **URL:** {html_url}\n"
            f"- **Description:** {description}\n"
            f"- **Stars:** {stars:,}\n"
            f"- **Forks:** {forks:,}\n"
            f"- **Open Issues:** {open_issues:,}\n"
            f"- **Primary Language:** {language}\n"
            f"- **License:** {license_name}"
        )
    except Exception as e:
        return f"Error fetching GitHub repo info: {str(e)}"


# List of tools the agent can use
tools = [search_tool, github_repo_info, arxiv_tool]

# System prompt to ensure clean, structured answers
SYSTEM_PROMPT = (
    "You are a helpful and thorough research assistant. "
    "When answering questions, use the web search tool if needed for recent or specific information, "
    "use the arxiv_tool when searching for academic papers, scientific articles, or research publications, "
    "or use the github_repo_info tool when asked about GitHub repositories. "
    "Always provide a well-structured, clear, and comprehensive final response directly addressing the user. "
    "Do not output raw internal thinking or scratchpad notes in your final answer."
)

# ── Agent Setup ────────────────────────────────────────────────────────────────
# create_react_agent builds a ReAct agent as a LangGraph StateGraph.
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


def run_agent(question: str) -> dict:
    """
    Run the agent on a user question and return the final answer
    along with a trace of reasoning steps (tool calls made).

    Args:
        question: The user's natural language question.

    Returns:
        dict with keys:
            - "answer": The agent's final text response.
            - "steps": List of dicts describing each tool call made,
              e.g. [{"tool": "web_search", "query": "...", "result": "..."}]
    """
    # Invoke the agent with the user's question as a HumanMessage
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # ── Extract the final answer ───────────────────────────────────────────
    # The last message in the conversation is the agent's final response
    messages = result["messages"]
    last_msg = messages[-1]
    final_answer = last_msg.content if last_msg.content else ""

    # ── Extract reasoning steps (tool calls) ───────────────────────────────
    # Walk through all messages to find tool calls and their results.
    # AIMessage objects may have a `tool_calls` attribute when the agent
    # decided to use a tool. ToolMessage objects contain the tool's response.
    steps = []
    tool_results = {}

    for msg in messages:
        # Collect tool call results first (ToolMessage has tool_call_id)
        if msg.type == "tool":
            tool_results[msg.tool_call_id] = msg.content

    for msg in messages:
        # AIMessages with tool_calls show what the agent decided to do
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                step = {
                    "tool": tool_call["name"],
                    "query": tool_call["args"],
                    "result": tool_results.get(tool_call["id"], ""),
                }
                steps.append(step)

    return {
        "answer": final_answer,
        "steps": steps,
    }
