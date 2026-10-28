const $ = (selector) => document.querySelector(selector);
const chatLog = $('#chat-log');
const {normalizeReviewData, removeRepositorySection, reviewToMarkdown} = window.reviewMarkdown;
let revealObserver = null;
let revealDelayIndex = 0;
const observedRevealTargets = new WeakSet();
const CHAT_REQUEST_TIMEOUT_MS = 180000;
let chatRequestEpoch = 0;
const activeChatControllers = new Set();
let threadId = sessionStorage.getItem('research-thread') || crypto.randomUUID();
sessionStorage.setItem('research-thread', threadId);

const themeToggle = $('#theme-toggle');
const themeColorMeta = $('meta[name="theme-color"]');
const colorSchemePreference = window.matchMedia?.('(prefers-color-scheme: dark)');
let hasExplicitTheme = false;
try {
  const savedTheme = localStorage.getItem('research-theme');
  hasExplicitTheme = savedTheme === 'dark' || savedTheme === 'light';
} catch {
  // Theme switching still works for this page when storage is unavailable.
}

function setTheme(theme, persist = false) {
  const isDark = theme === 'dark';
  document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
  themeToggle?.setAttribute('aria-checked', String(isDark));
  if (themeColorMeta) themeColorMeta.content = isDark ? '#101B17' : '#f6f7f5';
  if (persist) {
    hasExplicitTheme = true;
    try {
      localStorage.setItem('research-theme', isDark ? 'dark' : 'light');
    } catch {
      // Keep the selected theme for this page even when persistence is blocked.
    }
  }
}

setTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
themeToggle?.addEventListener('click', () => {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true);
});
colorSchemePreference?.addEventListener?.('change', (event) => {
  if (!hasExplicitTheme) setTheme(event.matches ? 'dark' : 'light');
});

function observeRevealTargets(root) {
  if (!revealObserver || !root) return;
  const targets = [];
  if (root.matches?.('[data-reveal]')) targets.push(root);
  targets.push(...root.querySelectorAll('[data-reveal]'));
  targets.forEach((target) => {
    if (observedRevealTargets.has(target)) return;
    observedRevealTargets.add(target);
    target.style.setProperty('--reveal-delay', `${(revealDelayIndex % 4) * 65}ms`);
    revealDelayIndex += 1;
    try {
      revealObserver.observe(target);
    } catch {
      target.classList.add('is-visible');
    }
  });
}

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
  const requestEpoch = chatRequestEpoch;
  const controller = typeof AbortController === 'undefined' ? null : new AbortController();
  if (controller) activeChatControllers.add(controller);
  let timedOut = false;
  let rejectTimeout;
  const timeoutPromise = new Promise((_, reject) => { rejectTimeout = reject; });
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller?.abort();
    rejectTimeout(new Error('Chat request timed out'));
  }, CHAT_REQUEST_TIMEOUT_MS);
  try {
    const request = fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, thread_id: threadId}),
      ...(controller ? {signal: controller.signal} : {})
    });
    const response = await Promise.race([request, timeoutPromise]);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    threadId = data.thread_id;
    sessionStorage.setItem('research-thread', threadId);
    const answer = data.answer || 'The assistant returned an empty answer.';
    const incomplete = answer.match(/^⚠ Incomplete response: ([^\n]+)\n\n([\s\S]*)$/);
    const message = incomplete ? incomplete[2] : answer;
    const toolCount = (data.steps || []).length;
    const meta = incomplete
      ? `Incomplete answer · ${incomplete[1]} · ${toolCount} tool call(s) attempted`
      : `${toolCount} tool call(s) attempted`;
    addMessage(message, incomplete ? 'assistant incomplete' : 'assistant', meta);
  } catch (error) {
    if (requestEpoch === chatRequestEpoch) {
      const detail = timedOut
        ? 'The assistant request timed out. Please try again.'
        : error?.name === 'AbortError'
          ? 'The assistant request was cancelled.'
          : error?.message === 'Failed to fetch'
            ? 'The assistant could not be reached. Check your connection and try again.'
            : error?.message || 'The assistant request failed. Please try again.';
      addMessage(detail, 'error');
    }
  } finally {
    clearTimeout(timeoutId);
    loading.remove();
    if (controller) activeChatControllers.delete(controller);
  }
}

$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#chat-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  await ask(question);
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
  chatRequestEpoch += 1;
  activeChatControllers.forEach((controller) => controller.abort());
  activeChatControllers.clear();
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

function displayField(value, fallback) {
  if (typeof value === 'string' && value.trim()) return value.trim();
  if (Array.isArray(value)) {
    const items = value.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim());
    if (items.length) return items.join(', ');
  }
  return fallback;
}

function safeExternalUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return '';
  try {
    const parsed = new URL(value, window.location.href);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : '';
  } catch {
    return '';
  }
}

function renderReviewMarkdown(container, review, papers) {
  const source = removeRepositorySection(review);
  if (!source) {
    container.textContent = 'No review text was included in the response. Retrieved paper records are shown above.';
    return;
  }

  const linkedReview = source.replace(/\[(P\d+)\]/g, (citation, id) => {
    const paper = papers.find((item) => item.citation_id === id);
    const url = safeExternalUrl(paper?.url);
    return url ? `[${id}](${url})` : citation;
  });

  try {
    const parseMarkdown = window.marked?.parse;
    const sanitizeHtml = window.DOMPurify?.sanitize;
    if (typeof parseMarkdown !== 'function' || typeof sanitizeHtml !== 'function') {
      container.textContent = linkedReview;
      return;
    }
    // Model output is inserted only after passing through the HTML sanitizer.
    container.innerHTML = sanitizeHtml(parseMarkdown(linkedReview));
  } catch {
    // Keep the report readable if a CDN renderer is unavailable or throws.
    container.textContent = linkedReview;
  }
}

function renderReview(data, markdown) {
  data = normalizeReviewData(data);
  const results = $('#review-results');
  results.innerHTML = '';
  results.hidden = false;
  const header = el('div', 'results-head');
  header.dataset.reveal = '';
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
  observeRevealTargets(header);
  const papersTitle = el('h4', '', `Retrieved papers (${data.papers.length})`);
  papersTitle.dataset.reveal = '';
  results.append(papersTitle);
  observeRevealTargets(papersTitle);
  const grid = el('div', 'paper-grid');
  results.append(grid);
  observeRevealTargets(grid);
  data.papers.forEach((paper, index) => {
    const card = el('article', 'paper-card');
    card.dataset.reveal = '';
    const citationId = displayField(paper.citation_id, `P${index + 1}`);
    const title = displayField(paper.title, 'Title not provided');
    const link = el('a', '', `[${citationId}] ${title}`);
    const sourceUrl = safeExternalUrl(paper.url);
    if (sourceUrl) {
      link.href = sourceUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
    card.append(link);
    card.append(el('p', '', displayField(paper.abstract, 'No abstract available.')));
    card.append(el('small', '', `${displayField(paper.authors, 'Authors not listed')} · ${displayField(paper.published, 'Date not listed')} · ${displayField(paper.categories, 'Categories not listed')}`));
    grid.append(card);
    observeRevealTargets(card);
  });
  if (!data.papers.length) grid.append(el('p', 'error-notes', 'No usable arXiv papers were found. Try a broader or reworded topic.'));
  if (data.review) {
    const structuredTitle = el('h4', '', 'Structured review');
    structuredTitle.dataset.reveal = '';
    results.append(structuredTitle);
    const reviewContent = el('div', 'review-markdown');
    reviewContent.dataset.reveal = '';
    renderReviewMarkdown(reviewContent, data.review, data.papers);
    results.append(reviewContent);
    observeRevealTargets(structuredTitle);
    observeRevealTargets(reviewContent);
  } else {
    const missingReview = el('div', 'review-markdown', 'No review text was returned. Retrieved paper records are shown above.');
    missingReview.dataset.reveal = '';
    results.append(el('h4', '', 'Structured review'));
    results.append(missingReview);
    observeRevealTargets(missingReview);
  }
  const repositoriesTitle = el('h4', '', `Related GitHub repositories (${data.repositories.length})`);
  repositoriesTitle.dataset.reveal = '';
  results.append(repositoriesTitle);
  observeRevealTargets(repositoriesTitle);
  results.append(el('p', 'error-notes', 'Topic-search results; no direct relationship to a listed paper is established.'));
  const repoList = el('div', 'repo-list');
  repoList.dataset.reveal = '';
  data.repositories.forEach((repo) => {
    const row = document.createElement('div');
    const repoName = displayField(repo.name, 'Unnamed repository');
    const link = el('a', '', repoName);
    const repoUrl = safeExternalUrl(repo.url);
    if (repoUrl) {
      link.href = repoUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      row.append(link);
    } else {
      row.append(document.createTextNode(repoName));
    }
    row.append(document.createTextNode(` — ${displayField(repo.description, 'No description')} · ${repo.stars ?? 'Stars unavailable'} stars`));
    repoList.append(row);
  });
  if (!data.repositories.length) repoList.textContent = 'No matching repositories were returned.';
  results.append(repoList);
  observeRevealTargets(repoList);
  const notices = Object.entries(data.errors || {}).filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`);
  if (notices.length) {
    const notice = el('p', 'error-notes', `Some sources were unavailable: ${notices.join(' · ')}`);
    notice.dataset.reveal = '';
    results.append(notice);
  }
  observeRevealTargets(results);
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
    const responseData = await response.json();
    if (!response.ok) throw new Error(responseData?.detail || `Request failed (${response.status})`);
    const data = normalizeReviewData(responseData);
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

(() => {
  const targets = [...document.querySelectorAll('[data-reveal]')];
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!targets.length || reduceMotion || !('IntersectionObserver' in window)) return;

  try {
    revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle('is-visible', entry.isIntersecting);
      });
    }, {rootMargin: '-3% 0px -8% 0px', threshold: 0.08});
    observeRevealTargets(document);
    // Enable the hidden/revealed styles only after every initial target is observed.
    document.documentElement.classList.add('motion-ready');
    window.addEventListener('pagehide', () => {
      revealObserver?.disconnect();
      revealObserver = null;
      document.documentElement.classList.remove('motion-ready');
    }, {once: true});
  } catch {
    revealObserver = null;
    document.documentElement.classList.remove('motion-ready');
  }
})();
