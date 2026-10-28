const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const {normalizeReviewData, reviewToMarkdown} = require('../static/review_markdown.js');

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.style = {setProperty() {}};
    this.attributes = {};
    this.hidden = false;
    this.disabled = false;
    this._text = '';
    this._html = '';
    this.clicked = false;
  }

  set textContent(value) {
    this._text = String(value ?? '');
    this.children = [];
    this._html = '';
  }

  get textContent() {
    return this._text + this.children.map((child) => child.textContent || '').join('');
  }

  set innerHTML(value) {
    this._html = String(value ?? '');
    this._text = '';
    this.children = [];
  }

  get innerHTML() { return this._html; }

  append(...nodes) { this.children.push(...nodes); }
  addEventListener(name, handler) { this.listeners.set(name, handler); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  scrollIntoView() {}
  click() { this.clicked = true; }
  querySelector(selector) {
    if (selector === 'button[type="submit"]') return this.submitButton;
    return null;
  }
}

function findAll(root, predicate) {
  const found = [];
  for (const child of root.children || []) {
    if (predicate(child)) found.push(child);
    found.push(...findAll(child, predicate));
  }
  return found;
}

function createPage(responseData, {withMarkdownLibraries = true} = {}) {
  const selectors = [
    '#chat-log', '#theme-toggle', 'meta[name="theme-color"]', '#chat-form', '#chat-input',
    '#new-chat', '#review-form', '#topic-input', '#review-status', '#review-results',
  ];
  const elements = new Map(selectors.map((selector) => [selector, new FakeElement('div')]));
  const reviewForm = elements.get('#review-form');
  const submitButton = new FakeElement('button');
  reviewForm.submitButton = submitButton;
  const results = elements.get('#review-results');
  const downloads = [];
  const nativeURL = URL;
  class TestURL extends nativeURL {}
  TestURL.createObjectURL = (blob) => { downloads.push(blob); return 'blob:review-test'; };
  TestURL.revokeObjectURL = () => {};

  const document = {
    documentElement: {dataset: {}, classList: {add() {}, remove() {}}},
    querySelector(selector) { return elements.get(selector) || null; },
    querySelectorAll() { return []; },
    createElement(tagName) { return new FakeElement(tagName); },
    createTextNode(text) { const node = new FakeElement('#text'); node.textContent = text; return node; },
  };
  const window = {
    location: {href: 'https://researchos.example/'},
    reviewMarkdown: require('../static/review_markdown.js'),
    matchMedia() { return {matches: false, addEventListener() {}}; },
    addEventListener() {},
  };
  if (withMarkdownLibraries) {
    window.marked = {parse: (markdown) => `<h2 data-source="${markdown.includes('Cross-paper interpretation')}">parsed review</h2>`};
    window.DOMPurify = {sanitize: (html) => `<p class="sanitized">${html}</p>`};
  }
  const context = vm.createContext({
    document,
    window,
    sessionStorage: {getItem() { return null; }, setItem() {}},
    crypto: webcrypto,
    fetch: async () => ({ok: true, json: async () => responseData}),
    URL: TestURL,
    Blob,
    Date,
    console,
  });
  const source = fs.readFileSync(require.resolve('../static/app.js'), 'utf8');
  vm.runInContext(source, context, {filename: 'static/app.js'});
  return {reviewForm, submitButton, results, elements, downloads};
}

async function submitReview(page, topic = 'retrieval augmented generation') {
  page.elements.get('#topic-input').value = topic;
  const handler = page.reviewForm.listeners.get('submit');
  assert.equal(typeof handler, 'function', 'review form submit handler should be connected');
  await handler({preventDefault() {}, currentTarget: page.reviewForm});
}

async function testRendersPaperDetailsSafeMarkdownAndMatchingDownload() {
  const response = {
    topic: 'retrieval augmented generation',
    review: '## Cross-paper interpretation\n\nThe abstract reports a result [P1].',
    papers: [
      {
        citation_id: 'P1', title: 'Grounded generation', authors: ['A. Researcher'],
        published: '2025-02-03', categories: ['cs.CL'], abstract: 'A retrieved abstract.',
        url: 'https://arxiv.org/abs/2502.00001',
      },
      {citation_id: 'P2', title: 'Optional metadata', authors: {unexpected: true}, abstract: null, url: 'javascript:alert(1)'},
    ],
    repositories: [],
    errors: {arxiv: null, github: null, synthesis: null},
  };
  const page = createPage(response);
  await submitReview(page);

  const text = page.results.textContent;
  assert.match(text, /Retrieved papers \(2\)/);
  assert.match(text, /Grounded generation/);
  assert.match(text, /A\. Researcher/);
  assert.match(text, /2025-02-03/);
  assert.match(text, /A retrieved abstract\./);
  assert.match(text, /Optional metadata/);
  assert.match(text, /Authors not listed/);
  assert.match(text, /No abstract available/);
  const cards = findAll(page.results, (node) => node.className === 'paper-card');
  assert.equal(cards.length, 2);
  const links = findAll(page.results, (node) => node.tagName === 'A');
  assert.equal(links.some((link) => link.href === 'https://arxiv.org/abs/2502.00001'), true);
  assert.equal(links.some((link) => String(link.href).startsWith('javascript:')), false);

  const markdownView = findAll(page.results, (node) => node.className === 'review-markdown')[0];
  assert.match(markdownView.innerHTML, /class="sanitized"/);
  assert.match(markdownView.innerHTML, /parsed review/);

  const downloadButton = findAll(page.results, (node) => node.className === 'download-button')[0];
  downloadButton.listeners.get('click')();
  const downloaded = page.downloads[0];
  assert.ok(downloaded, 'download button should create a Markdown file');
  const downloadedMarkdown = await downloaded.text();
  assert.equal(downloadedMarkdown, reviewToMarkdown(normalizeReviewData(response)));
}

async function testMissingFieldsAndUnavailableMarkdownLibrariesStayReadable() {
  const response = {topic: 'sparse metadata', papers: [{}], repositories: null, review: null};
  const page = createPage(response, {withMarkdownLibraries: false});
  await submitReview(page);

  assert.match(page.results.textContent, /Retrieved papers \(1\)/);
  assert.match(page.results.textContent, /Title not provided/);
  assert.match(page.results.textContent, /No review text was returned/);
  assert.match(page.results.textContent, /No matching repositories were returned/);
  const emptyResponse = normalizeReviewData({});
  assert.deepEqual(emptyResponse.papers, []);
  assert.deepEqual(emptyResponse.repositories, []);
}

(async () => {
  await testRendersPaperDetailsSafeMarkdownAndMatchingDownload();
  await testMissingFieldsAndUnavailableMarkdownLibrariesStayReadable();
  console.log('Literature review rendering regression checks passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
