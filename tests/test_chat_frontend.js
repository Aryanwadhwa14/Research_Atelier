/* Offline checks that chatbot requests always settle visibly and clean up. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function makeElement(tag = 'div') {
  const listeners = new Map();
  const node = {
    tagName: tag,
    className: '',
    classList: {add() {}, remove() {}, toggle() {}},
    children: [],
    dataset: {},
    value: '',
    textContent: '',
    style: {setProperty() {}},
    addEventListener(name, callback) { listeners.set(name, callback); },
    getListener(name) { return listeners.get(name); },
    append(...items) { items.forEach((item) => { item.parentNode = this; this.children.push(item); }); },
    remove() {
      if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    },
    querySelector(selector) {
      if (selector === '.welcome-card') return this.children.find((child) => child.className === 'welcome-card') || null;
      return null;
    },
    querySelectorAll() { return []; },
    setAttribute() {},
    focus() {},
    requestSubmit() { return this.getListener('submit')?.({preventDefault() {}}); },
    scrollIntoView() {},
    scrollHeight: 0,
    set innerHTML(value) { this._innerHTML = value; if (value === '') this.children = []; },
    get innerHTML() { return this._innerHTML || ''; },
  };
  return node;
}

async function runChatRequest(fetchImpl, timers = {}) {
  const elements = new Map([
    ['#chat-log', makeElement()], ['#chat-form', makeElement('form')],
    ['#chat-input', makeElement('textarea')], ['#new-chat', makeElement('button')],
    ['#theme-toggle', makeElement('button')], ['meta[name="theme-color"]', makeElement('meta')],
    ['#review-form', makeElement('form')], ['#review-results', makeElement()],
    ['#topic-input', makeElement('textarea')], ['#review-status', makeElement()],
  ]);
  elements.get('#chat-input').value = 'Compare vector databases';
  const documentElement = makeElement('html');
  documentElement.dataset = {theme: 'light'};
  const document = {
    documentElement,
    querySelector(selector) { return elements.get(selector) || null; },
    querySelectorAll() { return []; },
    createElement: makeElement,
    createTextNode(text) { return {textContent: text}; },
  };
  const window = {
    reviewMarkdown: {normalizeReviewData() {}, removeRepositorySection() {}, reviewToMarkdown() {}},
    matchMedia() { return {matches: true, addEventListener() {}}; },
    location: {href: 'https://researchos.test/'},
    addEventListener() {},
  };
  const context = {
    document, window, fetch: fetchImpl, AbortController,
    sessionStorage: {getItem() { return null; }, setItem() {}},
    localStorage: {getItem() { return null; }, setItem() {}},
    crypto: {randomUUID: () => 'thread-test'},
    URL, Blob, Date, console,
    setTimeout: timers.setTimeout || setTimeout,
    clearTimeout: timers.clearTimeout || clearTimeout,
  };
  vm.runInNewContext(fs.readFileSync('static/app.js', 'utf8'), context, {filename: 'static/app.js'});
  const event = {preventDefault() {}};
  await elements.get('#chat-form').getListener('submit')(event);
  return elements.get('#chat-log').children;
}

(async () => {
  const completed = await runChatRequest(async () => ({
    ok: true,
    async json() { return {answer: '⚠ Incomplete response: The text may be incomplete.\n\nMilvus needs ops effort, complex', steps: [{}, {}], thread_id: 'thread-test'}; },
  }));
  assert.equal(completed.some((node) => node.className === 'typing'), false);
  const partial = completed.find((node) => node.className === 'message assistant incomplete');
  assert.ok(partial);
  assert.match(partial.textContent, /Milvus needs ops effort, complex/);
  assert.match(partial.children[0].textContent, /2 tool call\(s\) attempted/);

  const networkError = await runChatRequest(async () => { throw new TypeError('Failed to fetch'); });
  assert.equal(networkError.some((node) => node.className === 'typing'), false);
  assert.match(networkError.find((node) => node.className === 'message error').textContent, /could not be reached/);

  const timedOut = await runChatRequest(() => new Promise(() => {}), {
    setTimeout(callback) { queueMicrotask(callback); return 1; },
    clearTimeout() {},
  });
  assert.equal(timedOut.some((node) => node.className === 'typing'), false);
  assert.match(timedOut.find((node) => node.className === 'message error').textContent, /timed out/);
  console.log('Chat frontend request tests passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
