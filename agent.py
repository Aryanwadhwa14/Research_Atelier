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
import arxiv
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file (picks up GROQ_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

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
# MemorySaver enables persistent conversation history per thread_id (session).
checkpointer = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


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
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

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
            f"{paper.get('abstract') or 'No abstract was provided in the retrieved record.'}\n\n"
            f"Source: {paper.get('url') or 'URL not provided'}"
        )
    lines.append("## Model-generated interpretation (based on the abstracts only)\n\n" + interpretation.strip())
    return "\n\n".join(lines)


def _fallback_review(topic: str, papers: list[dict]) -> str:
    """Return retrieved evidence and explicit limits without synthesizing conclusions."""
    lines = [
        f"## Topic and search scope\n\nThis response covers {len(papers)} arXiv record(s) retrieved for **{topic}**. "
        "It is a small, search-dependent sample and is not a comprehensive literature review.",
        "## Evidence reported in retrieved abstracts",
    ]
    for paper in papers:
        authors = ", ".join(paper.get("authors") or []) or "Authors not listed"
        date = paper.get("published") or "Publication date not listed"
        abstract = paper.get("abstract") or "No abstract was provided in the retrieved record."
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


def generate_literature_review(topic: str) -> dict:
    """Retrieve real arXiv papers and GitHub records, then synthesize a sourced review."""
    paper_by_key = {}
    search_errors = []
    comparison_topic = _is_rag_finetuning_comparison(topic)
    try:
        client = arxiv.Client()
    except Exception as exc:
        client = None
        search_errors.append(str(exc))
    if client:
        for query in build_arxiv_queries(topic):
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=12 if comparison_topic else 20,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    title = " ".join((getattr(result, "title", None) or "").split())
                    abstract = " ".join((getattr(result, "summary", None) or "").split())
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
                search_errors.append(f"{query}: {exc}")

    papers = list(paper_by_key.values())
    paper_error = "; ".join(search_errors) or None

    # Rank any records retrieved before a transient iteration failure too.
    papers = rank_papers(topic, papers, limit=5)
    for index, paper in enumerate(papers, start=1):
        paper["citation_id"] = f"P{index}"

    repositories = []
    github_error = None
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": topic, "sort": "stars", "order": "desc", "per_page": 5},
            headers={"Accept": "application/vnd.github+json", "User-Agent": "Agentic-Research-Assistant"},
            timeout=12,
        )
        if response.status_code == 403 or response.status_code == 429:
            github_error = "GitHub search rate limit reached."
        else:
            response.raise_for_status()
            for repo in response.json().get("items", []):
                repositories.append({
                    "name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "url": repo.get("html_url"),
                    "stars": repo.get("stargazers_count"),
                    "language": repo.get("language"),
                })
    except Exception as exc:
        github_error = str(exc)

    review = ""
    synthesis_error = None
    if papers:
        allowed_citation_ids = ", ".join(f"[{paper['citation_id']}]" for paper in papers)
        source_notes = "\n\n".join(
            f"[{p['citation_id']}] {p['title']}\nAuthors: {', '.join(p['authors']) or 'Not listed'}\n"
            f"Published: {p['published'] or 'Not listed'}\nCategories: {', '.join(p['categories']) or 'Not listed'}\n"
            f"Abstract: {p['abstract'] or 'Not available'}\nURL: {p['url'] or 'Not available'}"
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
            f"Topic: {topic}\n\n"
            f"Comparison limitation: {comparison_note or 'Relevant records may discuss the approaches without a direct comparison.'}\n\n"
            f"PAPERS:\n{source_notes}"
        )
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
                result = llm.invoke(synthesis_prompt + retry_note)
            except Exception as exc:
                exception_type = type(exc).__name__
                synthesis_error = f"Model synthesis failed during {'retry' if attempt else 'generation'}: {exc}"
                _log_synthesis_event(
                    logging.WARNING,
                    stage="model_invocation",
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

    return {
        "topic": topic,
        "papers": papers,
        "repositories": repositories,
        "review": review,
        "errors": {"arxiv": paper_error, "github": github_error, "synthesis": synthesis_error},
    }
