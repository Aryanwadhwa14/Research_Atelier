const assert = require('node:assert/strict');
const {removeRepositorySection, reviewToMarkdown} = require('../static/review_markdown.js');

const data = {
  topic: 'RAG versus fine-tuning',
  review: [
    '## Cross-paper interpretation',
    '',
    'The abstract discusses both methods [P1].',
    '',
    '## Related GitHub repositories',
    '',
    '- [old-copy](https://github.com/example/old-copy)',
    '',
    '## Closing note',
    '',
    'This report note should remain.',
  ].join('\n'),
  papers: [{
    citation_id: 'P1',
    title: 'A comparison',
    authors: ['A. Author'],
    published: '2025-01-01',
    categories: ['cs.CL'],
    url: 'https://arxiv.org/abs/1234',
    abstract: 'Abstract text.',
  }],
  repositories: [{
    name: 'example/current',
    url: 'https://github.com/example/current',
    description: 'Current repository result',
    stars: 7,
  }],
};

const body = removeRepositorySection(data.review);
assert.equal(body.includes('## Related GitHub repositories'), false);
assert.equal(body.includes('## Closing note'), true);
const markdown = reviewToMarkdown(data);
assert.equal((markdown.match(/^## Related GitHub repositories$/gm) || []).length, 1);
assert.equal(markdown.includes('example/current'), true);
assert.equal(markdown.includes('old-copy'), false);
assert.equal(markdown.includes('[P1]'), true);
console.log('Markdown repository de-duplication checks passed.');
