/* Offline checks that chatbot requests always settle visibly and clean up. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function makeElement(tag = 'div') {
  const listeners = new Map();
  const attributes = new Map();
  const node = {
    tagName: tag,
    nodeType: tag === 'fragment' ? 11 : 1,
    className: '',
    classList: {add() {}, remove() {}, toggle() {}},
    children: [],
    dataset: {},
    value: '',
    _textContent: '',
    get textContent() { return this._textContent + this.children.map((child) => child.textContent || '').join(''); },
    set textContent(value) { this._textContent = String(value); this.children = []; },
    style: {setProperty() {}},
    addEventListener(name, callback) { listeners.set(name, callback); },
    getListener(name) { return listeners.get(name); },
    append(...items) { items.forEach((item) => {
      if (item.nodeType === 11) this.children.push(...item.children);
      else { item.parentNode = this; this.children.push(item); }
    }); },
    replaceChildren(...items) { this.children = []; this.append(...items); },
    remove() {
      if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    },
    querySelector(selector) {
      if (selector === '.welcome-card') return this.children.find((child) => child.className === 'welcome-card') || null;
      return null;
    },
    querySelectorAll() { return []; },
    setAttribute(name, value) { attributes.set(name, value); },
    getAttribute(name) { return attributes.get(name) || null; },
    focus() {},
    requestSubmit() { return this.getListener('submit')?.({preventDefault() {}}); },
    scrollIntoView() {},
    scrollHeight: 0,
    set innerHTML(value) { this._innerHTML = value; if (value === '') this.children = []; },
    get innerHTML() { return this._innerHTML || ''; },
  };
  return node;
}

function sourceNode(tagName, attrs = {}, children = []) {
  return {
    nodeType: 1,
    tagName: tagName.toUpperCase(),
    childNodes: children,
    getAttribute(name) { return attrs[name] || null; },
  };
}

function textNode(text) { return {nodeType: 3, nodeValue: text}; }

async function runChatRequest(fetchImpl, timers = {}, markdownEnabled = true) {
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
    createTextNode(text) { return {nodeType: 3, textContent: text, children: []}; },
    createDocumentFragment() { return makeElement('fragment'); },
  };
  const window = {
    reviewMarkdown: {normalizeReviewData() {}, removeRepositorySection() {}, reviewToMarkdown() {}},
    ...(markdownEnabled ? {marked: {parse: (value) => value}} : {}),
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
    ...(markdownEnabled ? {DOMParser: class {
      parseFromString(markdown) {
        const link = String(markdown).match(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/);
        return {body: {childNodes: [sourceNode('p', {}, [
          sourceNode('strong', {}, [textNode('Recent approaches')]),
          textNode(' Milvus needs ops effort, complex '),
          ...(link ? [sourceNode('a', {href: link[2]}, [textNode(link[1])])] : []),
          sourceNode('script', {}, [textNode('alert(1)')]),
          sourceNode('a', {href: 'javascript:alert(1)'}, [textNode('unsafe link')]),
        ])]}};
      }
    }} : {}),
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
    async json() { return {answer: '⚠ Incomplete response: The text may be incomplete.\n\n**Recent approaches**\n\n[Paper source](https://example.org/paper)', steps: [{}, {}], thread_id: 'thread-test'}; },
  }));
  assert.equal(completed.some((node) => node.className === 'typing'), false);
  const partial = completed.find((node) => node.className === 'message assistant incomplete');
  assert.ok(partial);
  assert.match(partial.textContent, /Milvus needs ops effort, complex/);
  const renderedParagraph = partial.children[0].children[0];
  assert.ok(renderedParagraph.children.some((node) => node.tagName === 'strong'));
  assert.equal(renderedParagraph.children.some((node) => node.tagName === 'script'), false);
  const safeLink = renderedParagraph.children.find((node) => node.tagName === 'a');
  assert.equal(safeLink.getAttribute('href'), 'https://example.org/paper');
  const unsafeLink = renderedParagraph.children.filter((node) => node.tagName === 'a')[1];
  assert.equal(unsafeLink.getAttribute('href'), null);
  const incompleteMeta = partial.children.find((node) => node.className === 'message-meta');
  assert.match(incompleteMeta.textContent, /2 tool call\(s\) attempted/);

  const plainFallback = await runChatRequest(async () => ({
    ok: true,
    async json() { return {answer: '**Plain Markdown fallback**', steps: [], thread_id: 'thread-test'}; },
  }), {}, false);
  const plainAssistant = plainFallback.find((node) => node.className === 'message assistant');
  assert.match(plainAssistant.textContent, /\*\*Plain Markdown fallback\*\*/);
  assert.equal(plainAssistant.children[0].children.length, 0);

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
