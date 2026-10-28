"""Offline behavior tests for literature-review ranking and synthesis safeguards."""

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import agent
import main


def arxiv_record(title, abstract, entry_id, **metadata):
    return SimpleNamespace(
        title=title,
        summary=abstract,
        entry_id=entry_id,
        authors=metadata.get("authors", []),
        published=metadata.get("published"),
        pdf_url=metadata.get("pdf_url"),
        categories=metadata.get("categories", []),
    )


def run_with_mocks(
    records,
    model_text="",
    model_exception=None,
    model_responses=None,
    topic="retrieval augmented generation for large language models",
    github_items=None,
    capture_client=False,
):
    client = MagicMock()
    client.results.side_effect = lambda search: iter(records)
    github_response = MagicMock()
    github_response.status_code = 200
    github_response.json.return_value = {"items": github_items or []}
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content=model_text)
    if model_exception is not None:
        model.invoke.side_effect = model_exception
    elif model_responses is not None:
        model.invoke.side_effect = [SimpleNamespace(content=text) for text in model_responses]
    with (
        patch("agent.arxiv.Client", return_value=client),
        patch("agent.requests.get", return_value=github_response),
        patch("agent.llm", model),
    ):
        result = agent.generate_literature_review(topic)
    if capture_client:
        return result, model.invoke, client
    return result, model.invoke


class PaperRankingTests(unittest.TestCase):
    def test_relevant_interdisciplinary_candidate_ranks_above_off_topic_record(self):
        interdisciplinary = {
            "title": "Retrieval-augmented generation for biomedical language models",
            "abstract": "We study retrieval and generation for clinical question answering.",
        }
        off_topic = {
            "title": "Vision transformers for satellite image segmentation",
            "abstract": "This paper studies remote sensing image classification.",
        }

        ranked = agent.rank_papers(
            "retrieval augmented generation for large language models",
            [off_topic, interdisciplinary],
        )

        self.assertIs(ranked[0], interdisciplinary)
        self.assertIs(ranked[1], off_topic)  # low score ranks last; it is not hard-filtered

    def test_comparison_ranking_requires_both_methods_and_prioritizes_comparative_evidence(self):
        topic = "RAG versus fine-tuning for large language models"
        fine_tuning_only = {
            "title": "Fine-tuning large language models for instruction following",
            "abstract": "We study fine-tuning approaches for language models.",
        }
        rag_only = {
            "title": "Retrieval-augmented generation for question answering",
            "abstract": "We evaluate retrieval-augmented generation systems.",
        }
        survey = {
            "title": "A survey of trade-offs between RAG and fine-tuning for LLMs",
            "abstract": "This survey reviews both methods and their trade-offs.",
        }
        direct = {
            "title": "RAG versus fine-tuning for large language models: an empirical comparison",
            "abstract": "We compare both approaches on the same language-model tasks.",
        }

        ranked = agent.rank_papers(topic, [fine_tuning_only, rag_only, survey, direct])

        self.assertEqual(ranked, [direct, survey])

    def test_whole_topic_coverage_outranks_an_isolated_survey_signal(self):
        topic = "RAG versus fine-tuning for large language models"
        whole_topic_match = {
            "title": "RAG and fine-tuning for large language models",
            "abstract": "We discuss both approaches for large language models.",
        }
        partial_survey = {
            "title": "A survey of trade-offs between RAG and fine-tuning for LLMs",
            "abstract": "This survey reviews both methods.",
        }

        ranked = agent.rank_papers(topic, [partial_survey, whole_topic_match])

        self.assertIs(ranked[0], whole_topic_match)

    def test_comparison_topic_builds_targeted_arxiv_queries(self):
        topic = "RAG versus fine-tuning for large language models"

        queries = agent.build_arxiv_queries(topic)

        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0], topic)
        self.assertIn('"retrieval-augmented generation"', queries[1])
        self.assertIn('"fine-tuning"', queries[1])
        self.assertIn("survey", queries[2])
        self.assertIn("empirical", queries[2])


class LiteratureReviewTests(unittest.TestCase):
    def test_no_joint_method_papers_returns_limitation_without_calling_model(self):
        topic = "RAG versus fine-tuning for large language models"
        unrelated_to_comparison = arxiv_record(
            "Fine-tuning large language models for instruction following",
            "We study fine-tuning methods for language models.",
            "https://arxiv.org/abs/1",
        )

        result, model_call = run_with_mocks([unrelated_to_comparison], topic=topic)

        self.assertEqual(result["papers"], [])
        self.assertIn("mentions both RAG/retrieval-augmented generation and fine-tuning", result["review"])
        self.assertIn("one-sided papers", result["review"])
        model_call.assert_not_called()

    def test_targeted_retrieval_merges_and_deduplicates_comparison_papers(self):
        topic = "RAG versus fine-tuning for large language models"
        direct = arxiv_record(
            "RAG versus fine-tuning for large language models: an empirical comparison",
            "We compare both approaches on the same language-model tasks.",
            "https://arxiv.org/abs/direct",
        )
        single_method = arxiv_record(
            "Fine-tuning large language models for instruction following",
            "We study fine-tuning methods for language models.",
            "https://arxiv.org/abs/single",
        )
        valid_draft = (
            "## Cross-paper interpretation\n\nThe abstract describes a direct comparison [P1].\n\n"
            "## Candidate research gaps\n\nThe sample is too small to establish a field-wide gap."
        )

        result, model_call, client = run_with_mocks(
            [single_method, direct],
            model_text=valid_draft,
            topic=topic,
            capture_client=True,
        )

        self.assertEqual(client.results.call_count, 3)
        self.assertEqual([paper["url"] for paper in result["papers"]], ["https://arxiv.org/abs/direct"])
        self.assertEqual(result["papers"][0]["citation_id"], "P1")
        self.assertEqual(result["papers"][0]["url"], "https://arxiv.org/abs/direct")
        self.assertIsNone(result["errors"]["synthesis"])
        self.assertIn("Model-generated interpretation", result["review"])
        self.assertTrue(model_call.called)

    def test_github_section_is_only_in_dedicated_results_not_review_text(self):
        record = arxiv_record("Retrieval for language models", "Retrieval is evaluated.", "https://arxiv.org/abs/1")
        repositories = [{
            "full_name": "example/rag-demo",
            "description": "A sample repository",
            "html_url": "https://github.com/example/rag-demo",
            "stargazers_count": 7,
            "language": "Python",
        }]
        valid_draft = (
            "## Cross-paper interpretation\n\nThe abstract reports an evaluation [P1].\n\n"
            "## Candidate research gaps\n\nNo field-wide gap can be inferred."
        )

        result, _ = run_with_mocks([record], model_text=valid_draft, github_items=repositories)

        self.assertEqual(len(result["repositories"]), 1)
        self.assertEqual(result["repositories"][0]["name"], "example/rag-demo")
        self.assertNotIn("## Related GitHub repositories", result["review"])

        fallback, _ = run_with_mocks([record], model_text="Invalid synthesis", github_items=repositories)
        self.assertEqual(len(fallback["repositories"]), 1)
        self.assertNotIn("## Related GitHub repositories", fallback["review"])

    def test_qualified_always_phrase_triggers_one_retry_then_succeeds(self):
        record = arxiv_record(
            "Retrieval for language models",
            "We evaluate retrieval augmentation but do not measure hallucination outcomes or compare fine-tuning.",
            "https://arxiv.org/abs/1",
        )
        first_draft = (
            "## Cross-paper interpretation\n\nRAG does not always eliminate hallucinations [P1].\n\n"
            "## Candidate research gaps\n\nThe abstract leaves this outcome open."
        )
        revised_draft = (
            "## Cross-paper interpretation\n\nThe abstract does not report hallucination outcomes [P1].\n\n"
            "## Candidate research gaps\n\nA direct comparison with fine-tuning is not established by this record [P1]."
        )
        result, model_call = run_with_mocks([record], model_responses=[first_draft, revised_draft])

        self.assertEqual(model_call.call_count, 2)
        self.assertIn("'always'", model_call.call_args_list[1].args[0])
        self.assertIsNone(result["errors"]["synthesis"])
        self.assertIn("Model-generated interpretation", result["review"])
        self.assertIn("does not report hallucination outcomes [P1]", result["review"])

    def test_valid_first_attempt_is_accepted_without_retry(self):
        record = arxiv_record("Evidence retrieval for language models", "The abstract reports retrieval evaluation.", "https://arxiv.org/abs/1")
        valid_draft = (
            "## Cross-paper interpretation\n\nThe abstract reports retrieval evaluation [P1].\n\n"
            "## Candidate research gaps\n\nA larger sample could be reviewed."
        )

        result, model_call = run_with_mocks([record], model_text=valid_draft)

        self.assertEqual(model_call.call_count, 1)
        self.assertIsNone(result["errors"]["synthesis"])
        self.assertIn("Model-generated interpretation", result["review"])

    def test_citation_validation_failure_retries_and_validates_the_retry(self):
        record = arxiv_record("Evidence retrieval for language models", "The abstract reports retrieval evaluation.", "https://arxiv.org/abs/1")
        invalid_first = (
            "## Cross-paper interpretation\n\nThe abstract reports evaluation [P9].\n\n"
            "## Candidate research gaps\n\nA larger sample could be reviewed."
        )
        valid_retry = (
            "  ### Cross-paper interpretation ###  \n\nThe abstract reports evaluation [P1] and [P1].\n\n"
            " # Candidate research gaps # \n\nA larger sample could be reviewed."
        )

        result, model_call = run_with_mocks([record], model_responses=[invalid_first, valid_retry])

        self.assertEqual(model_call.call_count, 2)
        self.assertIn("failed the citations check", model_call.call_args_list[1].args[0])
        self.assertIn("only allowed citation IDs for this response are: [P1]", model_call.call_args_list[0].args[0])
        self.assertIn("repeated references are allowed", model_call.call_args_list[0].args[0])
        self.assertIn("## Cross-paper interpretation\n## Candidate research gaps", model_call.call_args_list[1].args[0])
        self.assertIsNone(result["errors"]["synthesis"])
        self.assertIn("The abstract reports evaluation [P1]", result["review"])

    def test_retry_with_valid_citation_but_missing_heading_still_falls_back(self):
        record = arxiv_record("Evidence retrieval for language models", "The abstract reports retrieval evaluation.", "https://arxiv.org/abs/1")
        invalid_first = (
            "## Cross-paper interpretation\n\nThe abstract reports evaluation [P9].\n\n"
            "## Candidate research gaps\n\nA larger sample could be reviewed."
        )
        missing_heading_retry = (
            "## Cross-paper interpretation\n\nThe abstract reports evaluation [P1].\n\n"
            "## Other observations\n\nA larger sample could be reviewed."
        )

        with self.assertLogs("agent", level="WARNING") as captured:
            result, model_call = run_with_mocks([record], model_responses=[invalid_first, missing_heading_retry])

        self.assertEqual(model_call.call_count, 2)
        self.assertIn('"stage": "citations"', captured.output[0])
        self.assertIn('"stage": "sections"', captured.output[1])
        self.assertIn("missing required section headings: candidate research gaps", result["errors"]["synthesis"])
        self.assertIn("The abstract reports retrieval evaluation.", result["review"])
        self.assertNotIn("Model-generated interpretation", result["review"])

    def test_section_validation_accepts_heading_levels_and_whitespace_but_requires_content(self):
        formatted = (
            "   ### Cross-paper interpretation ###   \n\nAn interpretation.\n\n"
            " # Candidate research gaps # \n\nA candidate for further study."
        )
        empty = "## Cross-paper interpretation\n\n## Candidate research gaps\n\nOnly one section has content."

        self.assertTrue(agent._has_required_interpretation_sections(formatted))
        self.assertFalse(agent._has_required_interpretation_sections(empty))

    def test_retry_validation_failure_is_logged_and_uses_safe_fallback(self):
        abstract = "The abstract evaluates retrieval for language models."
        record = arxiv_record("Evidence retrieval for language models", abstract, "https://arxiv.org/abs/1")
        absolute_first = (
            "## Cross-paper interpretation\n\nRAG always eliminates hallucinations [P1].\n\n"
            "## Candidate research gaps\n\nA larger sample could be reviewed."
        )
        invalid_retry = (
            "## Cross-paper interpretation\n\nThe abstract reports evaluation [P9].\n\n"
            "## Candidate research gaps\n\nA larger sample could be reviewed."
        )

        with self.assertLogs("agent", level="WARNING") as captured:
            result, model_call = run_with_mocks([record], model_responses=[absolute_first, invalid_retry])

        self.assertEqual(model_call.call_count, 2)
        self.assertIn('"stage": "absolute_wording"', captured.output[0])
        self.assertIn('"attempt": 1', captured.output[0])
        self.assertIn('"stage": "citations"', captured.output[1])
        self.assertIn("Synthesis retry failed the citations check", result["errors"]["synthesis"])
        self.assertIn(abstract, result["review"])
        self.assertNotIn("Model-generated interpretation", result["review"])

    def test_zero_usable_papers_returns_message_without_calling_model(self):
        unusable_record = SimpleNamespace(title="   ", summary="   ")
        for records in ([], [unusable_record]):
            with self.subTest(record_count=len(records)):
                result, model_call = run_with_mocks(records)

                self.assertEqual(result["papers"], [])
                self.assertIn("No usable arXiv papers", result["review"])
                self.assertIn("broader or reworded topic", result["review"])
                model_call.assert_not_called()

    def test_model_exception_returns_metadata_and_abstract_in_safe_fallback(self):
        abstract = "The paper studies retrieval for evidence-grounded question answering."
        record = arxiv_record("Evidence retrieval for language models", abstract, "https://arxiv.org/abs/1")
        result, model_call = run_with_mocks([record], model_exception=RuntimeError("model unavailable"))

        self.assertTrue(model_call.called)
        self.assertIn("model unavailable", result["errors"]["synthesis"])
        self.assertEqual(result["papers"][0]["title"], "Evidence retrieval for language models")
        self.assertEqual(result["papers"][0]["abstract"], abstract)
        self.assertIn(abstract, result["review"])
        self.assertIn("No cross-paper comparison or interpretation is asserted", result["review"])
        self.assertNotIn("Model-generated interpretation", result["review"])

    def test_absolute_claim_with_valid_citation_uses_fallback(self):
        record = arxiv_record("RAG for language models", "An approach is described.", "https://arxiv.org/abs/1")
        text = (
            "## Cross-paper interpretation\n\nRAG always eliminates hallucinations [P1].\n\n"
            "## Candidate research gaps\n\nNo conclusion can be drawn."
        )
        result, _ = run_with_mocks([record], text)

        self.assertIn("absolute wording", result["errors"]["synthesis"])
        self.assertIn("'always'", result["errors"]["synthesis"])
        self.assertNotIn("RAG always eliminates hallucinations", result["review"])
        self.assertIn("An approach is described.", result["review"])

    def test_absolute_claim_without_citation_uses_fallback(self):
        record = arxiv_record("RAG for language models", "An approach is described.", "https://arxiv.org/abs/1")
        text = (
            "## Cross-paper interpretation\n\nRAG always eliminates hallucinations.\n\n"
            "## Candidate research gaps\n\nNo conclusion can be drawn."
        )
        result, _ = run_with_mocks([record], text)

        self.assertIn("absolute wording", result["errors"]["synthesis"])
        self.assertIn("'always'", result["errors"]["synthesis"])
        self.assertNotIn("RAG always eliminates hallucinations", result["review"])
        self.assertIn("An approach is described.", result["review"])

    def test_missing_optional_metadata_is_preserved_as_missing(self):
        record = SimpleNamespace(title="Retrieval in scientific discovery", summary=None)
        result, model_call = run_with_mocks(
            [record],
            "## Cross-paper interpretation\n\nNo comparison can be drawn from this record [P1].\n\n"
            "## Candidate research gaps\n\nA broader review is needed before proposing candidate gaps.",
        )

        self.assertEqual(set(result), {"topic", "papers", "repositories", "review", "errors"})
        self.assertEqual(len(result["papers"]), 1)
        paper = result["papers"][0]
        self.assertTrue({"title", "authors", "published", "abstract", "url", "pdf_url"}.issubset(paper))
        self.assertEqual(paper["citation_id"], "P1")
        self.assertEqual(paper["title"], "Retrieval in scientific discovery")
        self.assertEqual(paper["authors"], [])
        self.assertIsNone(paper["published"])
        self.assertEqual(paper["abstract"], "")
        self.assertIsNone(paper["url"])
        self.assertIsNone(paper["pdf_url"])
        self.assertEqual(paper["categories"], [])
        self.assertTrue(model_call.called)
        self.assertIn("Evidence reported in retrieved abstracts", result["review"])
        self.assertIn("Model-generated interpretation", result["review"])

    def test_unknown_model_citation_uses_safe_fallback(self):
        record = arxiv_record("RAG for language models", "An approach is described.", "https://arxiv.org/abs/1")
        result, _ = run_with_mocks([record], "The result is clear [P9].")

        self.assertIn("unknown", result["errors"]["synthesis"])
        self.assertNotIn("[P9]", result["review"])
        self.assertIn("[P1]", result["review"])

    def test_unsupported_broad_claim_is_rejected_without_invented_conclusions(self):
        record = arxiv_record("RAG for language models", "An approach is described.", "https://arxiv.org/abs/1")
        result, _ = run_with_mocks(
            [record],
            "## Cross-paper interpretation\n\nMost research uses this method [P1].\n\n"
            "## Candidate research gaps\n\nA possible gap needs further study.",
        )

        self.assertIn("broad generalization", result["errors"]["synthesis"])
        self.assertNotIn("Most research uses", result["review"])
        self.assertNotIn("A possible gap needs further study", result["review"])
        self.assertIn("An approach is described.", result["review"])
        self.assertIn("Candidate research gaps", result["review"])
        self.assertIn("five-or-fewer-record sample", result["review"])

    def test_citation_ids_must_be_unique_and_refer_to_returned_records(self):
        papers = [{"citation_id": "P1"}, {"citation_id": "P1"}]
        self.assertFalse(agent.validate_review_citations("Claim [P1].", papers))
        self.assertFalse(agent.validate_review_citations("Claim [P2].", [{"citation_id": "P1"}]))
        self.assertTrue(agent.validate_review_citations("Claim [P1].", [{"citation_id": "P1"}]))

    def test_generator_returns_at_most_five_records_with_unique_citation_ids(self):
        records = [
            arxiv_record(
                f"Retrieval augmented generation paper {i}", "Large language models use retrieval.",
                f"https://arxiv.org/abs/{i}", authors=[SimpleNamespace(name="A. Author")],
                published=datetime(2024, 1, 2), categories=["cs.CL"],
            )
            for i in range(7)
        ]
        result, _ = run_with_mocks(
            records,
            "## Cross-paper interpretation\n\nThe records discuss related questions [P1].\n\n"
            "## Candidate research gaps\n\nNo candidate gap can be established from abstracts alone.",
        )
        self.assertEqual(len(result["papers"]), 5)
        self.assertEqual([paper["citation_id"] for paper in result["papers"]], [f"P{i}" for i in range(1, 6)])


class ApiCompatibilityTests(unittest.TestCase):
    def test_review_route_returns_composed_review_without_caching_and_ask_is_preserved(self):
        topic = "RAG versus fine-tuning for large language models"
        composed_review = "## Cross-paper interpretation\n\nCurrent review [P1]."
        review_data = {
            "topic": topic,
            "papers": [{"citation_id": "P1", "title": "Example", "abstract": "Abstract", "url": "https://arxiv.org/abs/1"}],
            "repositories": [],
            "review": composed_review,
            "errors": {"arxiv": None, "github": None, "synthesis": None},
        }
        client = TestClient(main.app)
        homepage = client.get("/")
        self.assertEqual(homepage.headers["cache-control"], "no-store, max-age=0")
        self.assertIn("/static/app.js?v=2", homepage.text)

        with patch("main.generate_literature_review", return_value=review_data) as generate:
            response = client.post("/literature-review", json={"topic": topic})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), review_data)
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        generate.assert_called_once_with(topic)

        with patch("main.run_agent", return_value={"answer": "Chat still works.", "steps": []}):
            chat_response = client.post("/ask", json={"question": "Explain retrieval augmented generation"})
        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(set(chat_response.json()), {"answer", "steps", "thread_id"})


if __name__ == "__main__":
    unittest.main()
