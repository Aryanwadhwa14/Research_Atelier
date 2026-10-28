"""
agent.py — LangGraph ReAct Agent Setup

This module configures:
1. The LLM (ChatGroq with Llama 3.3 70B)
2. The tools (DuckDuckGo web search, GitHub repo info, arXiv paper search)
3. The agent (LangGraph's create_react_agent, which implements the ReAct pattern:
   Reason → Act → Observe → Repeat until the agent has enough info to answer)
"""

import os
import re
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import arxiv
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ddgs import DDGS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.errors import GraphRecursionError

# Load environment variables from .env file (picks up GROQ_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)
CHAT_MODEL_MAX_TOKENS = 900

# ── LLM Setup ──────────────────────────────────────────────────────────────────
# ChatGroq wraps the Groq API. We use openai/gpt-oss-20b for fast inference,
# strong reasoning, and clean tool-calling output.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30,
    max_retries=1,
    max_tokens=CHAT_MODEL_MAX_TOKENS,
)


class _ModelTimingCallback(BaseCallbackHandler):
    """Log model latency and token counts without logging prompt/response text."""

    def __init__(self):
        self._started = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self._lock:
            self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        with self._lock:
            self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            started = self._started.pop(run_id, None)
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        finish_reasons = []
        if input_tokens is None or output_tokens is None:
            for batch in getattr(response, "generations", []) or []:
                for generation in batch:
                    metadata = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    input_tokens = input_tokens if input_tokens is not None else metadata.get("input_tokens")
                    output_tokens = output_tokens if output_tokens is not None else metadata.get("output_tokens")
        for batch in getattr(response, "generations", []) or []:
            for generation in batch:
                info = getattr(generation, "generation_info", None) or {}
                message_metadata = getattr(getattr(generation, "message", None), "response_metadata", None) or {}
                finish_reason = info.get("finish_reason") or message_metadata.get("finish_reason")
                if finish_reason:
                    finish_reasons.append(str(finish_reason))
        duration_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
        logger.info(
            "groq_call_complete model=%s duration_ms=%s input_tokens=%s output_tokens=%s finish_reason=%s",
            GROQ_MODEL,
            duration_ms,
            input_tokens,
            output_tokens,
            ",".join(finish_reasons) or "unavailable",
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            started = self._started.pop(run_id, None)
        duration_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
        logger.warning(
            "groq_call_failed model=%s duration_ms=%s error_type=%s",
            GROQ_MODEL,
            duration_ms,
            type(error).__name__,
        )


GROQ_MODEL = "openai/gpt-oss-20b"

# ── Tool Setup ─────────────────────────────────────────────────────────────────
WEB_SEARCH_TIMEOUT_SECONDS = 5
# DDGS recommends auto selection so a temporary outage in one engine can fall
# through to another. Keep one bounded DDGS request; it handles engine fallback.
WEB_SEARCH_ATTEMPTS = 1
WEB_SEARCH_BACKENDS = "auto"
TOOL_RESULT_MAX_CHARS = 2000
RETRIEVED_ABSTRACT_MAX_CHARS = 5000


def _limit_tool_result(text: str) -> str:
    text = str(text)
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    return text[:TOOL_RESULT_MAX_CHARS].rstrip() + " … [tool result shortened]"


@tool
def web_search(query: str) -> str:
    """Search current web pages for useful facts and return their titles, URLs, and snippets.
    Use this for current or specific information. If unavailable, use arxiv_tool or
    github_repo_info when relevant and clearly disclose that web search failed."""
    for attempt in range(1, WEB_SEARCH_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            with DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as search:
                results = search.text(
                    query,
                    max_results=3,
                    backend=WEB_SEARCH_BACKENDS,
                )
            if not results:
                logger.info(
                    "external_search_complete provider=ddgs duration_ms=%.1f result_count=0 attempt=%s",
                    (time.perf_counter() - started) * 1000,
                    attempt,
                )
                return "Web search completed but returned no results for this query."

            formatted = "\n\n".join(
                f"Title: {item.get('title') or 'Untitled result'}\n"
                f"URL: {item.get('href') or 'Not provided'}\n"
                f"Snippet: {item.get('body') or 'No snippet provided.'}"
                for item in results
            )
            logger.info(
                "external_search_complete provider=ddgs duration_ms=%.1f result_count=%s attempt=%s",
                (time.perf_counter() - started) * 1000,
                len(results),
                attempt,
            )
            return _limit_tool_result(formatted)
        except Exception as exc:
            logger.warning(
                "web_search_failed provider=ddgs provider_set=%s attempt=%s/%s duration_ms=%.1f error_type=%s",
                WEB_SEARCH_BACKENDS,
                attempt,
                WEB_SEARCH_ATTEMPTS,
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
            )
            if attempt < WEB_SEARCH_ATTEMPTS:
                time.sleep(0.25 * attempt)
            else:
                return (
                    f"Web search is unavailable after {WEB_SEARCH_ATTEMPTS} attempt(s) "
                    f"({type(exc).__name__}). No web results were retrieved. "
                    "Continue with other relevant research tools, and do not present this as a successful search."
                )


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for scientific papers, academic preprints, and research publications across physics, computer science, mathematics, quantitative biology, quantitative finance, statistics, electrical engineering, and economics.
    Use this tool when you need to find academic papers, scientific articles, or research publications.
    Input should be a search query string (e.g. 'retrieval augmented generation' or 'LLM reasoning agents')."""
    started = time.perf_counter()
    try:
        client = arxiv.Client(delay_seconds=3.0, num_retries=1)
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))
        logger.info(
            "external_search_complete provider=arxiv_tool duration_ms=%.1f result_count=%s",
            (time.perf_counter() - started) * 1000,
            len(results),
        )
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

        return _limit_tool_result("\n\n".join(formatted_papers))
    except Exception as e:
        logger.warning(
            "external_search_failed provider=arxiv_tool duration_ms=%.1f error_type=%s",
            (time.perf_counter() - started) * 1000,
            type(e).__name__,
        )
        return _limit_tool_result(f"Error fetching arXiv papers: {type(e).__name__}")


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

    started = time.perf_counter()
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
            logger.info(
                "external_search_complete provider=github_repo duration_ms=%.1f result_count=0 status=404",
                (time.perf_counter() - started) * 1000,
            )
            return f"Repository '{owner}/{repo_name}' was not found on GitHub."
        elif response.status_code != 200:
            logger.warning(
                "external_search_failed provider=github_repo duration_ms=%.1f status=%s",
                (time.perf_counter() - started) * 1000,
                response.status_code,
            )
            return _limit_tool_result(f"GitHub API returned status {response.status_code}.")

        data = response.json()
        name = data.get("full_name", f"{owner}/{repo_name}")
        description = data.get("description") or "No description provided."
        stars = data.get("stargazers_count", 0)
        forks = data.get("forks_count", 0)
        open_issues = data.get("open_issues_count", 0)
        language = data.get("language") or "Not specified"
        html_url = data.get("html_url", f"https://github.com/{owner}/{repo_name}")
        logger.info(
            "external_search_complete provider=github_repo duration_ms=%.1f result_count=1",
            (time.perf_counter() - started) * 1000,
        )
        license_info = data.get("license")
        license_name = license_info.get("name") if license_info else "No license"

        return _limit_tool_result(
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
        logger.warning(
            "external_search_failed provider=github_repo duration_ms=%.1f error_type=%s",
            (time.perf_counter() - started) * 1000,
            type(e).__name__,
        )
        return _limit_tool_result(f"Error fetching GitHub repo info: {type(e).__name__}")


# List of tools the agent can use
tools = [web_search, github_repo_info, arxiv_tool]

# System prompt to ensure clean, structured answers
SYSTEM_PROMPT = (
    "You are a helpful and thorough research assistant. "
    "When answering questions, use the web search tool if needed for recent or specific information, "
    "use the arxiv_tool when searching for academic papers, scientific articles, or research publications, "
    "or use the github_repo_info tool when asked about GitHub repositories. "
    "Keep final answers concise and directly address the user. For broad comparison questions, use this fixed structure: "
    "a brief introduction, a compact Markdown table with at most four comparison rows, and a practical recommendation. "
    "Avoid long exhaustive tables and lists. "
    "If a tool reports an error or no results, do not describe that tool as successful or invent results. "
    "Continue with other relevant tools when possible and disclose unavailable sources. "
    "Do not output raw internal thinking or scratchpad notes in your final answer."
)

# ── Agent Setup ────────────────────────────────────────────────────────────────
# create_react_agent builds a ReAct agent as a LangGraph StateGraph.
# MemorySaver enables persistent conversation history per thread_id (session).
checkpointer = MemorySaver()
CHAT_HISTORY_MAX_CHARS = 6500
CHAT_HISTORY_MAX_TURNS = 6
CHAT_TOOL_RESULT_MAX_CHARS = 600


def _message_size(message) -> int:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    extra = getattr(message, "additional_kwargs", {}) or {}
    reasoning = extra.get("reasoning_content", "")
    calls = json.dumps(getattr(message, "tool_calls", []) or [], ensure_ascii=False, default=str)
    return len(content) + len(str(reasoning)) + len(calls)


def _bound_chat_history(messages: list) -> tuple[list, bool]:
    """Keep recent complete turns within a conservative character budget."""
    turns = []
    for message in messages:
        if getattr(message, "type", None) == "human" and turns:
            turns.append([])
        if not turns:
            turns.append([])
        turns[-1].append(message)

    changed = False
    while len(turns) > CHAT_HISTORY_MAX_TURNS or (
        len(turns) > 1 and sum(_message_size(message) for turn in turns for message in turn) > CHAT_HISTORY_MAX_CHARS
    ):
        turns.pop(0)
        changed = True
    bounded = [message for turn in turns for message in turn]

    if sum(_message_size(message) for message in bounded) > CHAT_HISTORY_MAX_CHARS:
        for index in range(len(bounded) - 1, -1, -1):
            message = bounded[index]
            if getattr(message, "type", None) != "tool":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and len(content) > CHAT_TOOL_RESULT_MAX_CHARS:
                bounded[index] = message.model_copy(update={
                    "content": content[:CHAT_TOOL_RESULT_MAX_CHARS].rstrip() + " … [tool result shortened]"
                })
                changed = True
            if sum(_message_size(item) for item in bounded) <= CHAT_HISTORY_MAX_CHARS:
                break
    return bounded, changed


def _close_incomplete_tool_calls(state: dict) -> dict:
    """Persist an error ToolMessage for every unfinished call before the LLM runs.

    LangGraph can resume a checkpoint at its model node after a tool batch was
    only partly written. This hook runs inside the ReAct loop, including after
    each tool batch, and keeps the conversation while closing only missing IDs.
    """
    messages = list(state.get("messages", []))
    completed_ids = set()
    unique_messages = []
    duplicate_tool_message = False
    for message in messages:
        call_id = getattr(message, "tool_call_id", None)
        if getattr(message, "type", None) == "tool" and call_id:
            if call_id in completed_ids:
                duplicate_tool_message = True
                continue
            completed_ids.add(call_id)
        unique_messages.append(message)
    messages = unique_messages
    pending = []
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id")
            if call_id and call_id not in completed_ids:
                pending.append((call_id, call.get("name") or "tool"))

    repaired = messages + [
        ToolMessage(
            content=(
                f"Tool '{name}' did not finish in the previous execution. "
                "No result is available; retry it only if needed."
            ),
            tool_call_id=call_id,
            name=name,
            status="error",
        )
        for call_id, name in pending
    ]
    bounded, history_trimmed = _bound_chat_history(repaired)

    if not pending and not duplicate_tool_message and not history_trimmed:
        # Input only is sufficient; leave the checkpoint untouched.
        return {"llm_input_messages": bounded}
    logger.warning(
        "normalized_agent_history missing_results=%s duplicate_results_removed=%s history_trimmed=%s retained_chars=%s",
        len(pending),
        duplicate_tool_message,
        history_trimmed,
        sum(_message_size(message) for message in bounded),
    )
    # Overwrite the reducer-backed history and give the exact same repaired
    # sequence to the model. The old user/assistant messages remain intact.
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *bounded],
        "llm_input_messages": bounded,
    }


# Striped locks bound memory while serializing requests that share a thread ID.
_THREAD_LOCKS = [threading.RLock() for _ in range(64)]


def _thread_lock(thread_id: str):
    return _THREAD_LOCKS[hash(thread_id) % len(_THREAD_LOCKS)]


def _tool_call_signature(call: dict) -> tuple[str, str]:
    return (
        str(call.get("name") or ""),
        json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False, default=str),
    )


def _stop_repeated_tool_batch(state: dict) -> dict:
    """End a repeated tool-call batch before the graph executes it again."""
    messages = list(state.get("messages", []))
    if not messages:
        return {}
    last = messages[-1]
    calls = getattr(last, "tool_calls", []) or []
    if getattr(last, "type", None) != "ai" or not calls:
        return {}

    current_turn_start = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if getattr(messages[index], "type", None) == "human"),
        0,
    )
    previous_signatures = {
        _tool_call_signature(call)
        for message in messages[current_turn_start:-1]
        if getattr(message, "type", None) == "ai"
        for call in (getattr(message, "tool_calls", []) or [])
    }
    current_signatures = set()
    repeated = []
    for call in calls:
        signature = _tool_call_signature(call)
        if signature in previous_signatures or signature in current_signatures:
            repeated.append(signature[0])
        current_signatures.add(signature)
    if not repeated:
        return {}

    logger.warning(
        "agent_repeated_tool_batch_detected repeated_count=%s tool_names=%s",
        len(repeated),
        ",".join(sorted(set(repeated))),
    )
    # Replacing the same message ID removes its unexecuted tool calls from the
    # checkpoint, so there is no orphaned call and the graph terminates here.
    return {
        "messages": [AIMessage(
            id=getattr(last, "id", None),
            content="I stopped a repeated source lookup and will answer from the evidence already collected.",
            response_metadata={"researchos_incomplete_reason": "repeated_tool_call"},
        )]
    }


agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    pre_model_hook=_close_incomplete_tool_calls,
    post_model_hook=_stop_repeated_tool_batch,
)


def _repair_incomplete_tool_history(graph, config: dict) -> int:
    """Close interrupted tool-call batches in a checkpoint before model reuse.

    A process interruption can checkpoint an AIMessage containing tool calls
    before the ToolNode has saved its ToolMessages. Add explicit error results
    for only those unanswered calls so later turns keep the rest of the thread.
    """
    snapshot = graph.get_state(config)
    messages = (getattr(snapshot, "values", None) or {}).get("messages", [])
    completed_ids = {
        message.tool_call_id
        for message in messages
        if getattr(message, "type", None) == "tool" and getattr(message, "tool_call_id", None)
    }
    pending = []
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id")
            if call_id and call_id not in completed_ids:
                pending.append((call_id, call.get("name", "tool")))

    if not pending:
        return 0

    graph.update_state(
        config,
        {
            "messages": [
                ToolMessage(
                    content=(
                        f"Tool '{name}' did not finish in the previous request. "
                        "No result is available; retry it only if needed."
                    ),
                    tool_call_id=call_id,
                    name=name,
                    status="error",
                )
                for call_id, name in pending
            ]
        },
        # Mark the recovery messages as tool results so the graph routes back
        # to the model, instead of trying to execute the stale calls again.
        as_node="tools",
    )
    logger.warning("repaired_incomplete_tool_history count=%s", len(pending))
    return len(pending)


CHAT_FINAL_ATTEMPT_MAX_TOKENS = 280
CHAT_FINAL_ATTEMPT_MAX_CHARS = 3200
INCOMPLETE_RESPONSE_PREFIX = "⚠ Incomplete response: "


def _message_finish_reason(message) -> str | None:
    metadata = getattr(message, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason) if reason else None


def _message_was_truncated(message) -> bool:
    reason = (_message_finish_reason(message) or "").lower()
    if reason in {"length", "max_tokens", "token_limit", "max_output_tokens"}:
        return True
    usage = getattr(message, "usage_metadata", None) or {}
    output_tokens = usage.get("output_tokens")
    return isinstance(output_tokens, int) and output_tokens >= CHAT_MODEL_MAX_TOKENS


def _message_text(message) -> str:
    """Extract visible text from string or block-based model content."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(part for part in parts if part.strip()).strip()
    return ""


def _bounded_final_answer(question: str, messages: list, partial_answer: str, reason: str) -> tuple[str, bool] | None:
    """Make one concise no-tools final-answer attempt from collected evidence."""
    evidence = []
    for message in messages:
        if getattr(message, "type", None) != "tool" or getattr(message, "status", None) == "error":
            continue
        content = _message_text(message)
        if content:
            evidence.append(content[:500])
    evidence_text = "\n\n".join(f"Source result {index + 1}: {item}" for index, item in enumerate(evidence[-4:]))
    prompt = (
        "Finish the user's answer in at most 160 words using only the evidence below. Do not call tools. "
        "For a comparison, output one brief introduction, a compact table with no more than three rows, "
        "and one practical recommendation. Do not repeat the partial text verbatim if it is cut off.\n\n"
        f"Question: {question[:500]}\n\n"
        f"Source results collected:\n{evidence_text or 'No successful tool results were collected.'}\n\n"
        f"Partial answer (may end abruptly):\n{(partial_answer or '')[:900]}"
    )[:CHAT_FINAL_ATTEMPT_MAX_CHARS]
    started = time.perf_counter()
    logger.info("agent_request_stage stage=bounded_final_attempt reason=%s", reason)
    try:
        response = llm.invoke(
            prompt,
            max_tokens=CHAT_FINAL_ATTEMPT_MAX_TOKENS,
            config={"callbacks": [_ModelTimingCallback()]},
        )
    except Exception as exc:
        logger.warning(
            "agent_final_attempt_failed duration_ms=%.1f error_type=%s",
            (time.perf_counter() - started) * 1000,
            type(exc).__name__,
        )
        return None

    content = _message_text(response)
    finish_reason = _message_finish_reason(response)
    logger.info(
        "agent_final_attempt_complete duration_ms=%.1f finish_reason=%s output_chars=%s",
        (time.perf_counter() - started) * 1000,
        finish_reason or "unavailable",
        len(content),
    )
    if not content.strip():
        return None
    complete = not _message_was_truncated(response) and not getattr(response, "tool_calls", None)
    return content.strip(), complete


def _mark_incomplete_answer(answer: str, reason: str) -> str:
    warning_by_reason = {
        "output_limit": "The model reached its response limit. The text below may be incomplete.",
        "repeated_tool_call": "The agent stopped after repeating a source lookup. The text below may be incomplete.",
        "recursion_limit": "The agent reached its tool-step limit. The text below may be incomplete.",
    }
    warning = warning_by_reason.get(reason, "The agent could not finish synthesizing a complete answer.")
    body = (answer or "No complete answer was produced, but the request ended safely.").strip()
    return f"{INCOMPLETE_RESPONSE_PREFIX}{warning}\n\n{body}"


def run_agent(question: str, thread_id: str) -> dict:
    """
    Run the agent on a user question and return the final answer
    along with a trace of reasoning steps (tool calls made).

    Args:
        question: The user's natural language question.
        thread_id: A unique identifier for the conversation session.
                   The agent will retain conversation history for this thread_id.

    Returns:
        dict with keys:
            - "answer": The agent's final text response.
            - "steps": List of dicts describing each tool call made,
              e.g. [{"tool": "web_search", "query": "...", "result": "..."}]
    """
    # Invoke the agent with the user's question as a HumanMessage
    # Pass config with thread_id so the agent retains conversation history in this session
    request_started = time.perf_counter()
    logger.info("agent_request_stage stage=graph_invoke_start")
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [_ModelTimingCallback()],
        "recursion_limit": 12,
    }
    # MemorySaver is process-local; prevent overlapping requests from racing a
    # shared checkpoint within this worker. Different sessions can still run.
    graph_terminal = "complete"
    with _thread_lock(thread_id):
        _repair_incomplete_tool_history(agent, config)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
        except GraphRecursionError as exc:
            graph_terminal = "recursion_limit"
            logger.warning(
                "agent_graph_terminated stage=recursion_limit error_type=%s",
                type(exc).__name__,
            )
            _repair_incomplete_tool_history(agent, config)
            snapshot = agent.get_state(config)
            recovered_messages = (getattr(snapshot, "values", None) or {}).get("messages", [])
            useful_evidence = any(
                getattr(message, "type", None) == "tool"
                and getattr(message, "status", None) != "error"
                and bool(getattr(message, "content", ""))
                for message in recovered_messages
            )
            partial_available = any(
                getattr(message, "type", None) == "ai"
                and not (getattr(message, "tool_calls", []) or [])
                and bool(getattr(message, "content", ""))
                for message in recovered_messages
            )
            if not useful_evidence and not partial_available:
                raise
            result = {"messages": recovered_messages}
        except Exception as exc:
            logger.warning("agent_graph_failed stage=invoke error_type=%s", type(exc).__name__)
            # The pre-model hook repairs persisted partial batches on every
            # model step. This boundary repair handles failures that escape
            # before LangGraph gets another chance to run that hook.
            if not _repair_incomplete_tool_history(agent, config):
                raise
            result = agent.invoke({"messages": []}, config=config)

    # ── Extract the final answer ───────────────────────────────────────────
    # The last message may be an output-limited draft, not a completed answer.
    messages = result["messages"]
    last_msg = messages[-1]
    last_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if getattr(messages[index], "type", None) == "human"
         and str(getattr(messages[index], "content", "")).strip() == question.strip()),
        0,
    )
    current_messages = messages[last_user_index:]
    response_reason = (getattr(last_msg, "response_metadata", None) or {}).get(
        "researchos_incomplete_reason"
    )
    truncated_msg = next(
        (message for message in reversed(current_messages)
         if getattr(message, "type", None) == "ai" and _message_was_truncated(message)),
        None,
    )
    if not response_reason and truncated_msg:
        response_reason = "output_limit"
    final_answer = _message_text(last_msg)
    # Some provider limit responses carry finish_reason=length but an empty
    # final content. Preserve the latest earlier assistant draft in that case.
    if not final_answer:
        final_answer = next(
            (_message_text(message) for message in reversed(current_messages)
             if getattr(message, "type", None) == "ai" and _message_text(message)),
            "",
        )
    if graph_terminal == "recursion_limit":
        response_reason = "recursion_limit"

    if response_reason:
        recovered = _bounded_final_answer(question, current_messages, str(final_answer), response_reason)
        if recovered:
            final_answer, completed = recovered
            if completed:
                response_reason = None
            else:
                final_answer = _mark_incomplete_answer(final_answer, "output_limit")
                response_reason = "output_limit"
        else:
            final_answer = _mark_incomplete_answer(str(final_answer), response_reason)

    # ── Extract reasoning steps (tool calls) ───────────────────────────────
    # Walk through all messages to find tool calls and their results.
    # AIMessage objects may have a `tool_calls` attribute when the agent
    # decided to use a tool. ToolMessage objects contain the tool's response.
    steps = []
    tool_results = {}

    for msg in current_messages:
        # Collect tool call results first (ToolMessage has tool_call_id)
        if msg.type == "tool":
            tool_results[msg.tool_call_id] = msg.content

    for msg in current_messages:
        # AIMessages with tool_calls show what the agent decided to do
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                step = {
                    "tool": tool_call["name"],
                    "query": tool_call["args"],
                    "result": tool_results.get(tool_call["id"], ""),
                }
                steps.append(step)

    logger.info(
        "agent_request_terminal graph_state=%s finish_reason=%s incomplete=%s elapsed_ms=%.1f message_count=%s tool_call_count=%s",
        graph_terminal,
        _message_finish_reason(last_msg) or ("repeated_tool_call" if response_reason == "repeated_tool_call" else "unavailable"),
        response_reason is not None,
        (time.perf_counter() - request_started) * 1000,
        len(messages),
        len(steps),
    )

    return {
        "answer": final_answer,
        "steps": steps,
    }


_RANK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "using", "with",
}


def _topic_terms(text: str) -> set[str]:
    """Return normalized topic terms without common connective words."""
    return {
        term for term in re.findall(r"[a-z0-9]+", (text or "").lower())
        if term not in _RANK_STOPWORDS
    }


SYNTHESIS_ABSTRACT_CHARS = 650
SYNTHESIS_TITLE_CHARS = 200
SYNTHESIS_TOPIC_CHARS = 300
SYNTHESIS_MAX_INPUT_CHARS = 7000
SYNTHESIS_MAX_OUTPUT_TOKENS = 900
SYNTHESIS_MAX_ESTIMATED_REQUEST_TOKENS = 6000


def _estimate_request_tokens(prompt: str, completion_ceiling: int) -> int:
    """Conservative rough estimate for English text; never treated as tokenizer output."""
    return (len(prompt) + 1) // 2 + completion_ceiling


def _bound_synthesis_prompt(prompt: str) -> str:
    input_char_budget = min(
        SYNTHESIS_MAX_INPUT_CHARS,
        max(0, 2 * (SYNTHESIS_MAX_ESTIMATED_REQUEST_TOKENS - SYNTHESIS_MAX_OUTPUT_TOKENS)),
    )
    return prompt[:input_char_budget]


def _bounded_abstract_excerpt(abstract: str, topic: str, limit: int = SYNTHESIS_ABSTRACT_CHARS) -> str:
    """Select a topic-relevant, character-bounded excerpt from an abstract."""
    text = " ".join((abstract or "").split())
    if len(text) <= limit:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    terms = _topic_terms(topic)
    scored = sorted(
        enumerate(sentences),
        key=lambda item: (-len(terms.intersection(_topic_terms(item[1]))), item[0]),
    )
    selected = []
    used = 0
    for _, sentence in scored:
        extra = len(sentence) + (1 if selected else 0)
        if used + extra <= limit:
            selected.append(sentence)
            used += extra
    if not selected:
        selected = [text[:limit].rsplit(" ", 1)[0]]
    # Restore the paper's original sentence order after relevance selection.
    selected.sort(key=lambda sentence: sentences.index(sentence))
    excerpt = " ".join(selected)
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0]
    return excerpt.rstrip(" .") + " … [excerpt limited]"


def _friendly_synthesis_error(exc: Exception, *, attempt: int) -> str:
    """Return a concise user-facing error without exposing raw request details."""
    description = str(exc).lower()
    phase = "retry" if attempt > 1 else "generation"
    if any(marker in description for marker in ("rate limit", "too many requests", "429")):
        return f"Model rate limit reached during synthesis {phase}; retrieved evidence is shown below. Please retry later."
    if "tokens per minute" in description or re.search(r"\btpm\b", description):
        return f"Groq's per-request token budget prevented synthesis during {phase}; retrieved evidence is shown below."
    if any(marker in description for marker in (
        "request too large", "context length", "context window", "token limit", "maximum context"
    )) or re.search(r"requested\s+\d+.{0,40}limit\s+\d+", description):
        return f"Model input limit prevented synthesis during {phase}; retrieved evidence is shown below."
    return f"Model synthesis failed during {phase} ({type(exc).__name__}); retrieved evidence is shown below."


def _is_rag_finetuning_comparison(topic: str) -> bool:
    """Recognize queries asking about RAG/retrieval against fine-tuning."""
    terms = _topic_terms(topic)
    has_rag = "rag" in terms or {"retrieval", "augmented", "generation"}.issubset(terms)
    has_finetuning = "finetuning" in terms or {"fine", "tuning"}.issubset(terms)
    asks_comparison = bool(re.search(r"\b(?:vs\.?|versus|compar\w*|trade[\s-]?offs?)\b", topic, re.IGNORECASE))
    return has_rag and has_finetuning and asks_comparison


def build_arxiv_queries(topic: str) -> list[str]:
    """Expand comparison queries to retrieve co-mentions and comparison-focused work."""
    queries = [topic]
    if _is_rag_finetuning_comparison(topic):
        both_methods = (
            '(all:"retrieval-augmented generation" OR all:"retrieval augmented generation" OR all:RAG) '
            'AND (all:"fine-tuning" OR all:"fine tuning")'
        )
        queries.extend([
            both_methods,
            both_methods + " AND (all:comparison OR all:comparative OR all:trade-offs OR all:survey OR all:empirical OR all:evaluation)",
        ])
    return list(dict.fromkeys(queries))


def _comparison_method_coverage(paper: dict) -> tuple[bool, bool]:
    terms = _topic_terms(f"{paper.get('title', '')} {paper.get('abstract', '')}")
    has_rag = "rag" in terms or {"retrieval", "augmented", "generation"}.issubset(terms)
    has_finetuning = "finetuning" in terms or {"fine", "tuning"}.issubset(terms)
    return has_rag, has_finetuning


def _has_comparison_signal(paper: dict) -> bool:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    return bool(re.search(
        r"\b(?:compar\w*|versus|vs\.?|trade[\s-]?offs?)\b",
        text,
        re.IGNORECASE,
    ))


def _has_study_signal(paper: dict) -> bool:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    return _has_comparison_signal(paper) or bool(re.search(
        r"\b(?:survey|empirical|evaluat\w*)\b",
        text,
        re.IGNORECASE,
    ))


def rank_papers(topic: str, papers: list[dict], limit: int = 5) -> list[dict]:
    """Prioritize whole-topic coverage; comparison queries require both methods."""
    terms = _topic_terms(topic)
    comparison_topic = _is_rag_finetuning_comparison(topic)
    indexed = list(enumerate(papers))

    def score(paper: dict) -> tuple[float, float, float, int]:
        title_terms = _topic_terms(paper.get("title", ""))
        all_terms = _topic_terms(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        coverage = len(terms & all_terms) / len(terms) if terms else 0.0
        title_coverage = len(terms & title_terms) / len(terms) if terms else 0.0
        if comparison_topic:
            # Joint method coverage is required; full-topic coverage comes first,
            # with comparison/survey signals used only to break relevance ties.
            return (1.0, coverage, float(_has_study_signal(paper)), title_coverage)
        return (coverage, title_coverage, float(len(terms & title_terms)), len(all_terms & terms))

    if comparison_topic:
        indexed = [
            (index, paper) for index, paper in indexed
            if all(_comparison_method_coverage(paper))
        ]
    indexed.sort(key=lambda item: (tuple(-value for value in score(item[1])), item[0]))
    return [paper for _, paper in indexed[:limit]]


_CITATION_PATTERN = re.compile(r"\[(P[^\]]*)\]")
_BROAD_CLAIM_PATTERN = re.compile(
    r"\b(?:most|all|nearly all|the majority of|generally|typically|widely|universally)\s+"
    r"(?:of\s+)?(?:the\s+)?(?:research|studies|papers|literature|work|approaches|methods)\b"
    r"|\b(?:the field as a whole|across the field|researchers generally|a dominant trend|"
    r"a prevailing approach)\b"
    r"|\b(?:predominantly|predominant|dominant|prevailing|widespread|widely adopted)\b"
    r".{0,100}\b(?:research|studies|papers|literature|field|methods|approaches)\b"
    r"|\b(?:research|studies|papers|literature|field|methods|approaches)\b"
    r".{0,100}\b(?:predominantly|primarily|dominant|prevailing|widespread)\b",
    re.IGNORECASE,
)

# Surface-level guard for unusually absolute wording. This is not fact-checking:
# it can miss equivalent wording and may reject a phrase even when a source supports it.
_ABSOLUTE_CLAIM_PATTERN = re.compile(
    r"\b(?:always|never|guarantees?|guaranteed|completely eliminates?|eliminates? all)\b",
    re.IGNORECASE,
)


def _find_absolute_wording(review: str) -> str | None:
    """Return the exact matched word/phrase so a retry can explain the rejection."""
    match = _ABSOLUTE_CLAIM_PATTERN.search(review or "")
    return match.group(0) if match else None


def validate_review_citations(review: str, papers: list[dict]) -> bool:
    """Accept only known citation IDs, with exactly one returned record per ID."""
    record_ids = [paper.get("citation_id") for paper in papers]
    if any(not citation_id for citation_id in record_ids):
        return False
    if len(record_ids) != len(set(record_ids)):
        return False
    known_ids = set(record_ids)
    citations = _CITATION_PATTERN.findall(review or "")
    if any(not re.fullmatch(r"P\d+", citation_id) for citation_id in citations):
        return False
    return set(citations).issubset(known_ids)


def _has_required_interpretation_sections(review: str) -> bool:
    required = {"cross-paper interpretation", "candidate research gaps"}
    lines = (review or "").splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            title = match.group(2).strip().lower()
            headings.append((index, len(match.group(1)), title))

    for title in required:
        nonempty_section_found = False
        for start, level, heading_title in headings:
            if heading_title != title:
                continue
            end = len(lines)
            for next_index, next_level, _ in headings:
                if next_index > start and next_level <= level:
                    end = next_index
                    break
            body = [
                line.strip()
                for line in lines[start + 1:end]
                if not re.match(r"^\s*#{1,6}\s+", line)
            ]
            if any(body):
                nonempty_section_found = True
                break
        if not nonempty_section_found:
            return False
    return True


def _validate_synthesis_draft(review: str, papers: list[dict]) -> tuple[str | None, str | None]:
    """Return the first validation stage and concise reason, or (None, None)."""
    if not validate_review_citations(review, papers):
        return "citations", "unknown or non-unique citation identifiers"
    if _BROAD_CLAIM_PATTERN.search(review or ""):
        return "broad_claim", "unsupported broad generalization"
    absolute_phrase = _find_absolute_wording(review or "")
    if absolute_phrase:
        return "absolute_wording", f"absolute wording {absolute_phrase!r}"

    if not _has_required_interpretation_sections(review):
        headings = {heading.strip().lower() for heading in re.findall(r"^##\s+(.+?)\s*$", review or "", re.MULTILINE)}
        missing = {"cross-paper interpretation", "candidate research gaps"} - headings
        return "sections", "missing required section headings: " + ", ".join(sorted(missing))
    return None, None


def _log_synthesis_event(level: int, *, stage: str, reason: str, attempt: int) -> None:
    """Log structured diagnostics without request content, credentials, or model output."""
    event = {
        "event": "literature_review_synthesis",
        "stage": stage,
        "reason": reason,
        "attempt": attempt,
    }
    logger.log(level, "literature_review_synthesis %s", json.dumps(event, sort_keys=True))


def _comparison_limitation(topic: str, papers: list[dict]) -> str | None:
    if _is_rag_finetuning_comparison(topic) and not any(_has_comparison_signal(paper) for paper in papers):
        return (
            "No direct-comparison or trade-off signal was identified in the retrieved titles/abstracts. "
            "Treat these records as co-mentions of the methods, not as evidence of a head-to-head result."
        )
    return None


def _compose_review(topic: str, papers: list[dict], interpretation: str) -> str:
    """Build the factual evidence section from records; keep model prose labeled as interpretation."""
    lines = [
        f"## Topic overview\n\nThis review summarizes {len(papers)} arXiv record(s) retrieved for **{topic}**. "
        "This is a small, search-dependent sample based on abstracts and metadata, not a comprehensive review.",
        "## Evidence reported in retrieved abstracts",
    ]
    limitation = _comparison_limitation(topic, papers)
    if limitation:
        lines[0] += f"\n\n**Comparison limitation:** {limitation}"
    for paper in papers:
        lines.append(
            f"### [{paper['citation_id']}] {paper.get('title') or 'Title not provided'}\n\n"
            f"- **Authors:** {', '.join(paper.get('authors') or []) or 'Not listed'}\n"
            f"- **Published:** {paper.get('published') or 'Not listed'}\n"
            f"- **arXiv categories:** {', '.join(paper.get('categories') or []) or 'Not listed'}\n\n"
            f"{_bounded_abstract_excerpt(paper.get('abstract') or '', topic) or 'No abstract was provided in the retrieved record.'}\n\n"
            f"Source: {paper.get('url') or 'URL not provided'}"
        )
    lines.append("## Model-generated interpretation (based on the abstracts only)\n\n" + interpretation.strip())
    return "\n\n".join(lines)


def _fallback_review(topic: str, papers: list[dict]) -> str:
    """Return retrieved evidence and explicit limits without synthesizing conclusions."""
    lines = [
        "## Incomplete review: synthesis unavailable",
        f"## Topic and search scope\n\nThis response covers {len(papers)} arXiv record(s) retrieved for **{topic}**. "
        "It is a small, search-dependent sample and is not a comprehensive literature review.",
        "## Evidence reported in retrieved abstracts",
    ]
    for paper in papers:
        authors = ", ".join(paper.get("authors") or []) or "Authors not listed"
        date = paper.get("published") or "Publication date not listed"
        abstract = _bounded_abstract_excerpt(paper.get("abstract") or "", topic)
        abstract = abstract or "No abstract was provided in the retrieved record."
        lines.append(
            f"### [{paper['citation_id']}] {paper.get('title') or 'Title not provided'}\n\n"
            f"{abstract}\n\nSource: {paper.get('url') or 'URL not provided'} · {authors} · {date}"
        )
    lines.extend([
        "## Comparison and interpretation",
        "No cross-paper comparison or interpretation is asserted because synthesis was unavailable or failed validation. "
        "The records above contain abstracts and metadata, not full-paper analysis.",
        "## Candidate research gaps",
        "No research gaps are inferred from this five-or-fewer-record sample. A broader, systematic review is needed to assess gaps.",
    ])
    limitation = _comparison_limitation(topic, papers)
    if limitation:
        lines.append(f"## Comparison limitation\n\n{limitation}")
    return "\n\n".join(lines)


def _retrieve_arxiv_records(topic: str) -> tuple[list[dict], str | None]:
    """Run the small arXiv query set with bounded retries and per-query timing."""
    paper_by_key = {}
    search_errors = []
    queries = build_arxiv_queries(topic)
    retrieval_started = time.perf_counter()
    try:
        # arXiv asks clients to leave a delay between requests; keep those
        # requests sequential and bound library retries instead of spiking it.
        client = arxiv.Client(delay_seconds=3.0, num_retries=1)
    except Exception as exc:
        client = None
        search_errors.append(f"arXiv client unavailable ({type(exc).__name__}).")
    if client:
        for query_index, query in enumerate(queries, start=1):
            query_started = time.perf_counter()
            result_count = 0
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=12,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    result_count += 1
                    title = " ".join((getattr(result, "title", None) or "").split())
                    abstract = " ".join((getattr(result, "summary", None) or "").split())
                    if len(abstract) > RETRIEVED_ABSTRACT_MAX_CHARS:
                        abstract = abstract[:RETRIEVED_ABSTRACT_MAX_CHARS].rsplit(" ", 1)[0] + " … [abstract shortened]"
                    # A record with neither a title nor abstract cannot provide useful evidence.
                    if not title and not abstract:
                        continue
                    paper = {
                        "title": title or "Untitled paper",
                        "authors": [getattr(author, "name", str(author)) for author in (getattr(result, "authors", None) or [])],
                        "published": getattr(result, "published", None).strftime("%Y-%m-%d")
                        if getattr(result, "published", None) else None,
                        "abstract": abstract,
                        "url": getattr(result, "entry_id", None),
                        "pdf_url": getattr(result, "pdf_url", None),
                        "categories": list(getattr(result, "categories", None) or []),
                    }
                    key = paper["url"] or f"{paper['title'].casefold()}|{abstract.casefold()}"
                    paper_by_key.setdefault(key, paper)
            except Exception as exc:
                search_errors.append(f"arXiv query {query_index} failed ({type(exc).__name__}).")
                logger.warning(
                    "external_search_failed provider=arxiv query_index=%s duration_ms=%.1f error_type=%s",
                    query_index,
                    (time.perf_counter() - query_started) * 1000,
                    type(exc).__name__,
                )
            else:
                logger.info(
                    "external_search_complete provider=arxiv query_index=%s duration_ms=%.1f result_count=%s",
                    query_index,
                    (time.perf_counter() - query_started) * 1000,
                    result_count,
                )

    logger.info(
        "external_retrieval_complete provider=arxiv duration_ms=%.1f unique_count=%s query_count=%s",
        (time.perf_counter() - retrieval_started) * 1000,
        len(paper_by_key),
        len(queries),
    )
    return list(paper_by_key.values()), "; ".join(search_errors) or None


def _retrieve_github_repositories(topic: str) -> tuple[list[dict], str | None]:
    """Retrieve related repositories with a bounded request and safe timing logs."""
    started = time.perf_counter()
    repositories = []
    error = None
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": topic, "sort": "stars", "order": "desc", "per_page": 5},
            headers={"Accept": "application/vnd.github+json", "User-Agent": "Agentic-Research-Assistant"},
            timeout=8,
        )
        if response.status_code == 403 or response.status_code == 429:
            error = "GitHub search rate limit reached."
        else:
            response.raise_for_status()
            for repo in response.json().get("items", []):
                repositories.append({
                    "name": repo.get("full_name"),
                    "description": (repo.get("description") or "")[:500] or None,
                    "url": repo.get("html_url"),
                    "stars": repo.get("stargazers_count"),
                    "language": repo.get("language"),
                })
    except Exception as exc:
        error = f"GitHub search failed ({type(exc).__name__})."
        logger.warning(
            "external_search_failed provider=github duration_ms=%.1f error_type=%s",
            (time.perf_counter() - started) * 1000,
            type(exc).__name__,
        )
    logger.info(
        "external_retrieval_complete provider=github duration_ms=%.1f result_count=%s error=%s",
        (time.perf_counter() - started) * 1000,
        len(repositories),
        bool(error),
    )
    return repositories, error


def generate_literature_review(topic: str) -> dict:
    """Retrieve real arXiv papers and GitHub records concurrently, then synthesize."""
    request_started = time.perf_counter()
    comparison_topic = _is_rag_finetuning_comparison(topic)

    # arXiv queries remain sequential to respect the API's request delay;
    # GitHub is independent and runs at the same time.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-retrieval") as pool:
        arxiv_future = pool.submit(_retrieve_arxiv_records, topic)
        github_future = pool.submit(_retrieve_github_repositories, topic)
        retrieved_records, paper_error = arxiv_future.result()
        repositories, github_error = github_future.result()

    # Rank any records retrieved before a transient iteration failure too.
    papers = rank_papers(topic, retrieved_records, limit=5)
    for index, paper in enumerate(papers, start=1):
        paper["citation_id"] = f"P{index}"

    review = ""
    synthesis_error = None
    if papers:
        allowed_citation_ids = ", ".join(f"[{paper['citation_id']}]" for paper in papers)
        # The model needs only citation ID, title, and a bounded abstract
        # excerpt. Full metadata remains in the API response/paper cards.
        source_notes = "\n\n".join(
            f"[{p['citation_id']}] {(p.get('title') or 'Untitled paper')[:SYNTHESIS_TITLE_CHARS]}\n"
            f"Abstract excerpt: {_bounded_abstract_excerpt(p.get('abstract') or '', topic)}"
            for p in papers
        )
        comparison_note = _comparison_limitation(topic, papers)
        synthesis_prompt = (
            "Write two clearly separated, non-empty interpretation sections. Use exactly these Markdown headings:\n"
            "## Cross-paper interpretation\n## Candidate research gaps\n"
            "Do not rewrite or repeat the factual evidence; it will be displayed separately from the supplied abstracts. "
            f"The only allowed citation IDs for this response are: {allowed_citation_ids}. Use these exact IDs and do not "
            "invent IDs or cite papers absent from the supplied records. Each ID maps to exactly one retrieved paper record. "
            "You may repeat the same citation ID wherever that paper supports a claim; repeated references are allowed and "
            "do not mean there are duplicate paper records. Attach a supplied citation ID to every paper-specific claim. A citation is appropriate only "
            "when the supplied abstract supports the associated claim; otherwise state that the sample does not establish it. "
            "Use qualified, scoped language. Avoid universal claims such as 'always', 'never', 'guarantees', or 'eliminates all'. "
            "The evidence is at most five papers and abstracts, so it cannot establish field-wide prevalence or definitive gaps. "
            "Do not make broad claims such as 'most research' or 'the field generally'. Research-gap statements must be "
            "candidates for further study, not established gaps. Do not claim a direct RAG-versus-fine-tuning comparison "
            "unless a supplied abstract directly compares both approaches.\n\n"
            f"Topic: {topic[:SYNTHESIS_TOPIC_CHARS]}\n\n"
            f"Comparison limitation: {comparison_note or 'Relevant records may discuss the approaches without a direct comparison.'}\n\n"
            f"PAPERS:\n{source_notes}"
        )
        # The 8,000 limit reported for this deployment is the free-tier
        # per-request TPM budget, not the model's 131K context window. Keep
        # prompt plus requested completion below a conservative estimate.
        synthesis_prompt = _bound_synthesis_prompt(synthesis_prompt)
        previous_failure = None
        for attempt in range(2):
            retry_note = ""
            if attempt == 1:
                retry_note = (
                    f"\n\nRevision required: the previous draft failed the {previous_failure['stage']} check because it had "
                    f"{previous_failure['reason']}. Correct this issue and return both non-empty sections using these exact headings:\n"
                    "## Cross-paper interpretation\n## Candidate research gaps\n"
                    f"Use only these exact citation IDs: {allowed_citation_ids}. Repeated citations to one ID are allowed. "
                    "Do not make unsupported claims; use only supplied citation IDs and claims supported by the abstracts."
                )
            try:
                prompt = _bound_synthesis_prompt(synthesis_prompt + retry_note)
                logger.info(
                    "groq_synthesis_prompt chars=%s estimated_request_tokens=%s completion_ceiling=%s",
                    len(prompt),
                    _estimate_request_tokens(prompt, SYNTHESIS_MAX_OUTPUT_TOKENS),
                    SYNTHESIS_MAX_OUTPUT_TOKENS,
                )
                result = llm.invoke(
                    prompt,
                    max_tokens=SYNTHESIS_MAX_OUTPUT_TOKENS,
                    config={"callbacks": [_ModelTimingCallback()]},
                )
            except Exception as exc:
                exception_type = type(exc).__name__
                synthesis_error = _friendly_synthesis_error(exc, attempt=attempt + 1)
                error_lower = synthesis_error.lower()
                stage = (
                    "tpm_limit" if "per-request token budget" in error_lower else
                    "rate_limit" if "rate limit" in error_lower else
                    "input_limit" if "input limit" in error_lower else
                    "model_invocation"
                )
                _log_synthesis_event(
                    logging.WARNING,
                    stage=stage,
                    reason=f"model call raised {exception_type}",
                    attempt=attempt + 1,
                )
                break

            draft = result.content if isinstance(result.content, str) else str(result.content)
            failed_stage, failure_reason = _validate_synthesis_draft(draft, papers)
            if failed_stage:
                previous_failure = {"stage": failed_stage, "reason": failure_reason}
                _log_synthesis_event(
                    logging.WARNING,
                    stage=failed_stage,
                    reason=failure_reason,
                    attempt=attempt + 1,
                )
                if attempt == 0:
                    continue
                synthesis_error = (
                    f"Synthesis retry failed the {failed_stage} check ({failure_reason}); "
                    "showing retrieved evidence instead."
                )
                break

            review = _compose_review(topic, papers, draft)
            synthesis_error = None
            _log_synthesis_event(
                logging.INFO,
                stage="accepted",
                reason="draft passed citation, claim, wording, and section checks",
                attempt=attempt + 1,
            )
            break

    if not papers:
        if comparison_topic:
            if paper_error:
                review = (
                    "No jointly relevant RAG and fine-tuning papers could be retrieved because one or more arXiv searches failed. "
                    "Try again later or reword the topic."
                )
            else:
                review = (
                    "No arXiv papers were found whose title or abstract mentions both RAG/retrieval-augmented generation "
                    "and fine-tuning. The review is left empty rather than populated with one-sided papers. "
                    "Try a broader or reworded topic."
                )
        elif paper_error:
            review = (
                f"No usable arXiv papers could be retrieved for **{topic}** because the arXiv search failed. "
                "Try again later, or try a broader or reworded topic."
            )
        else:
            review = (
                f"No usable arXiv papers were found for **{topic}**. "
                "Try a broader or reworded topic."
            )
    elif not review:
        review = _fallback_review(topic, papers)

    logger.info(
        "literature_review_complete duration_ms=%.1f paper_count=%s repository_count=%s synthesis_failed=%s",
        (time.perf_counter() - request_started) * 1000,
        len(papers),
        len(repositories),
        bool(synthesis_error),
    )
    return {
        "topic": topic,
        "papers": papers,
        "repositories": repositories,
        "review": review,
        "errors": {"arxiv": paper_error, "github": github_error, "synthesis": synthesis_error},
    }
