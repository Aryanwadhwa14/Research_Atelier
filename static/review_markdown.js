(function exposeReviewMarkdown(root) {
  function removeRepositorySection(review) {
    return String(review || '')
      .replace(/(?:^|\n)## Related GitHub repositories\s*\n[\s\S]*?(?=\n##\s|$)/gi, '')
      .trim();
  }

  function escapeMarkdown(text) {
    return String(text || '').replace(/[\[\]()*_`]/g, '\\$&');
  }

  function reviewToMarkdown(data) {
    const report = removeRepositorySection(data.review);
    const lines = [
      `# Literature review: ${data.topic}`,
      '',
      '## Structured review',
      '',
      report || 'No review could be generated because no papers were retrieved.',
      '',
      '## Retrieved papers',
      '',
    ];
    if (!data.papers.length) lines.push('_No papers were returned by arXiv._', '');
    data.papers.forEach((paper, index) => {
      lines.push(
        `### [${paper.citation_id || `P${index + 1}`}] ${paper.title}`,
        '',
        `- **Authors:** ${paper.authors?.length ? paper.authors.map(escapeMarkdown).join(', ') : 'Not listed'}`,
        `- **Published:** ${paper.published || 'Not listed'}`,
        `- **Categories:** ${paper.categories?.length ? paper.categories.join(', ') : 'Not listed'}`,
        `- **Source:** ${paper.url ? `[arXiv](${paper.url})` : 'URL not available'}`,
        '',
        paper.abstract || '_No abstract available._',
        '',
      );
    });
    lines.push(
      '## Related GitHub repositories',
      '',
      '_Topic-search results; no direct relationship to a listed paper is established._',
      '',
    );
    if (!data.repositories.length) lines.push('_No repositories were returned._', '');
    data.repositories.forEach((repo) => {
      lines.push(`- [${repo.name}](${repo.url}) — ${repo.description || 'No description'} (${repo.stars ?? 'Stars unavailable'} stars)`);
    });
    return lines.join('\n');
  }

  const api = {removeRepositorySection, reviewToMarkdown};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.reviewMarkdown = api;
})(globalThis);
