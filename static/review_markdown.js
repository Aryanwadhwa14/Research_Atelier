(function exposeReviewMarkdown(root) {
  function normalizeReviewData(data) {
    const source = data && typeof data === 'object' ? data : {};
    const asRecords = (value) => Array.isArray(value)
      ? value.filter((item) => item && typeof item === 'object' && !Array.isArray(item))
      : [];
    return {
      ...source,
      topic: typeof source.topic === 'string' && source.topic.trim() ? source.topic.trim() : 'Research topic',
      review: typeof source.review === 'string' ? source.review : '',
      papers: asRecords(source.papers),
      repositories: asRecords(source.repositories),
      errors: source.errors && typeof source.errors === 'object' && !Array.isArray(source.errors)
        ? source.errors
        : {},
    };
  }

  function removeRepositorySection(review) {
    return String(review || '')
      .replace(/(?:^|\n)## Related GitHub repositories\s*\n[\s\S]*?(?=\n##\s|$)/gi, '')
      .trim();
  }

  function escapeMarkdown(text) {
    return String(text || '').replace(/[\[\]()*_`]/g, '\\$&');
  }

  function reviewToMarkdown(data) {
    data = normalizeReviewData(data);
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
        `### [${paper.citation_id || `P${index + 1}`}] ${paper.title || 'Title not provided'}`,
        '',
        `- **Authors:** ${Array.isArray(paper.authors) && paper.authors.length
          ? paper.authors.map(escapeMarkdown).join(', ')
          : typeof paper.authors === 'string' && paper.authors.trim() ? escapeMarkdown(paper.authors) : 'Not listed'}`,
        `- **Published:** ${paper.published || 'Not listed'}`,
        `- **Categories:** ${Array.isArray(paper.categories) && paper.categories.length
          ? paper.categories.map(escapeMarkdown).join(', ')
          : typeof paper.categories === 'string' && paper.categories.trim() ? escapeMarkdown(paper.categories) : 'Not listed'}`,
        `- **Source:** ${paper.url ? `[arXiv](${paper.url})` : 'URL not available'}`,
        '',
        typeof paper.abstract === 'string' && paper.abstract.trim() ? paper.abstract : '_No abstract available._',
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
      const name = typeof repo.name === 'string' && repo.name.trim() ? repo.name : 'Unnamed repository';
      const description = typeof repo.description === 'string' && repo.description.trim()
        ? repo.description
        : 'No description';
      lines.push(repo.url
        ? `- [${name}](${repo.url}) — ${description} (${repo.stars ?? 'Stars unavailable'} stars)`
        : `- ${name} — ${description} (${repo.stars ?? 'Stars unavailable'} stars)`);
    });
    return lines.join('\n');
  }

  const api = {normalizeReviewData, removeRepositorySection, reviewToMarkdown};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.reviewMarkdown = api;
})(globalThis);
