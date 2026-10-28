const $ = (selector) => document.querySelector(selector);
const chatLog = $('#chat-log');
const {removeRepositorySection, reviewToMarkdown} = window.reviewMarkdown;
let threadId = sessionStorage.getItem('research-thread') || crypto.randomUUID();
sessionStorage.setItem('research-thread', threadId);

function addMessage(text, role, meta = '') {
  const node = document.createElement('div');
  node.className = `message ${role}`;
  node.textContent = text;
  if (meta) {
    const detail = document.createElement('small');
    detail.className = 'message-meta';
    detail.textContent = meta;
    node.append(detail);
  }
  chatLog.append(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return node;
}

async function ask(question) {
  if (chatLog.querySelector('.welcome-card')) chatLog.innerHTML = '';
  addMessage(question, 'user');
  const loading = document.createElement('div');
  loading.className = 'typing';
  loading.textContent = 'Searching sources and synthesizing…';
  chatLog.append(loading);
  try {
    const response = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, thread_id: threadId})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    threadId = data.thread_id;
    sessionStorage.setItem('research-thread', threadId);
    loading.remove();
    addMessage(data.answer || 'The assistant returned an empty answer.', 'assistant', `${(data.steps || []).length} tool call(s)`);
  } catch (error) {
    loading.remove();
    addMessage(`${error.message}. Check that the FastAPI service and GROQ_API_KEY are configured.`, 'error');
  }
}

$('#chat-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const input = $('#chat-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  ask(question);
});
$('#chat-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#chat-form').requestSubmit();
  }
});
document.querySelectorAll('.prompt-chip').forEach((button) => button.addEventListener('click', () => {
  $('#chat-input').value = button.dataset.prompt;
  $('#chat-input').focus();
}));
$('#new-chat').addEventListener('click', () => {
  threadId = crypto.randomUUID();
  sessionStorage.setItem('research-thread', threadId);
  chatLog.innerHTML = '<div class="welcome-card"><div class="sparkle">✳</div><h3>What are you exploring?</h3><p>Ask a research question to get started. I can search the web, find academic papers, and look up GitHub projects.</p></div>';
});

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function renderReview(data, markdown) {
  const results = $('#review-results');
  results.innerHTML = '';
  results.hidden = false;
  const header = el('div', 'results-head');
  header.append(el('h3', '', `Review: ${data.topic}`));
  const download = el('button', 'download-button', '↓ Download Markdown');
  download.type = 'button';
  download.addEventListener('click', () => {
    const blob = new Blob([markdown], {type: 'text/markdown;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const topicSlug = (data.topic || 'research').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
    const downloadedAt = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `literature-review-${topicSlug || 'research'}-${downloadedAt}.md`;
    link.click();
    URL.revokeObjectURL(url);
  });
  header.append(download);
  results.append(header);
  const papersTitle = el('h4', '', `Retrieved papers (${data.papers.length})`);
  results.append(papersTitle);
  const grid = el('div', 'paper-grid');
  data.papers.forEach((paper) => {
    const card = el('article', 'paper-card');
    const link = el('a', '', `[${paper.citation_id || `P${data.papers.indexOf(paper) + 1}`}] ${paper.title}`);
    if (paper.url) {
      link.href = paper.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
    card.append(link);
    card.append(el('p', '', paper.abstract || 'No abstract available.'));
    card.append(el('small', '', `${paper.authors?.join(', ') || 'Authors not listed'} · ${paper.published || 'Date not listed'} · ${paper.categories?.join(', ') || 'Categories not listed'}`));
    grid.append(card);
  });
  if (!data.papers.length) grid.append(el('p', 'error-notes', 'No usable arXiv papers were found. Try a broader or reworded topic.'));
  results.append(grid);
  if (data.review) {
    results.append(el('h4', '', 'Structured review'));
    const reviewContent = el('div', 'review-markdown');
    const linkedReview = removeRepositorySection(data.review).replace(/\[(P\d+)\]/g, (citation, id) => {
      const paper = data.papers.find((item) => item.citation_id === id);
      return paper?.url ? `[${id}](${paper.url})` : citation;
    });
    const renderedHtml = marked.parse(linkedReview);
    reviewContent.innerHTML = DOMPurify.sanitize(renderedHtml);
    results.append(reviewContent);
  }
  results.append(el('h4', '', `Related GitHub repositories (${data.repositories.length})`));
  results.append(el('p', 'error-notes', 'Topic-search results; no direct relationship to a listed paper is established.'));
  const repoList = el('div', 'repo-list');
  data.repositories.forEach((repo) => {
    const row = document.createElement('div');
    const link = el('a', '', repo.name);
    link.href = repo.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    row.append(link, document.createTextNode(` — ${repo.description || 'No description'} · ${repo.stars ?? 'Stars unavailable'} stars`));
    repoList.append(row);
  });
  if (!data.repositories.length) repoList.textContent = 'No matching repositories were returned.';
  results.append(repoList);
  const notices = Object.entries(data.errors || {}).filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`);
  if (notices.length) results.append(el('p', 'error-notes', `Some sources were unavailable: ${notices.join(' · ')}`));
}

$('#review-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const topic = $('#topic-input').value.trim();
  const status = $('#review-status');
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  status.textContent = 'Searching arXiv and GitHub, then drafting the review…';
  try {
    const response = await fetch('/literature-review', {
      method: 'POST', cache: 'no-store',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({topic})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    const markdownForResponse = reviewToMarkdown(data);
    renderReview(data, markdownForResponse);
    status.textContent = data.errors?.synthesis
      ? `Review synthesis did not pass validation or service checks. Showing retrieved evidence and metadata. Reason: ${data.errors.synthesis}`
      : data.papers.length
      ? `Review ready · ${data.papers.length} paper(s) retrieved.`
      : data.errors?.arxiv
        ? 'Could not retrieve usable arXiv papers. Retry later, or try a broader or reworded topic.'
        : 'No usable arXiv papers found. Try a broader or reworded topic.';
    $('#review-results').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    status.textContent = `${error.message}. Please try again.`;
  } finally {
    button.disabled = false;
  }
});
