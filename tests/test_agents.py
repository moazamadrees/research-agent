import pytest
from unittest.mock import MagicMock, patch

from app.models import SubTask, ReviewStatus
from app.agents.reporter import _compute_confidence, _parse_report
from app.agents.state import ResearchState


class TestComputeConfidence:
    def test_full_coverage_high_retrieval(self):
        subtasks = [
            SubTask(id=1, query="q1", retrieved_chunks=["c1"], retrieval_score=0.9),
            SubTask(id=2, query="q2", retrieved_chunks=["c2"], retrieval_score=0.85),
        ]
        score = _compute_confidence(subtasks, llm_score=0.9)
        assert score > 0.7

    def test_empty_corpus_low_score(self):
        subtasks = [
            SubTask(id=1, query="q1", retrieved_chunks=[], retrieval_score=0.0),
            SubTask(id=2, query="q2", retrieved_chunks=[], retrieval_score=0.0),
        ]
        score = _compute_confidence(subtasks, llm_score=0.3)
        assert score < 0.5

    def test_no_subtasks_returns_zero(self):
        assert _compute_confidence([], llm_score=0.9) == 0.0

    def test_score_clamped_to_one(self):
        subtasks = [SubTask(id=1, query="q", retrieved_chunks=["c"], retrieval_score=1.0)]
        score = _compute_confidence(subtasks, llm_score=1.0)
        assert score <= 1.0

    def test_score_clamped_to_zero(self):
        subtasks = [SubTask(id=1, query="q", retrieved_chunks=[], retrieval_score=0.0)]
        score = _compute_confidence(subtasks, llm_score=0.0)
        assert score >= 0.0


class TestParseReport:
    SAMPLE_REPORT = """## Executive Summary
This is the executive summary of the research findings.

## Section: RAG Architecture Overview
RAG combines dense retrieval with generative models to improve factual grounding.

## Section: LLM Reasoning Capabilities
Large language models demonstrate emergent reasoning via chain-of-thought prompting.

## Confidence Assessment
Score: 0.82
Reasoning: Strong evidence from multiple high-quality sources."""

    def _make_subtasks(self):
        return [
            SubTask(id=1, query="RAG architecture", source_docs=["paper1.pdf"]),
            SubTask(id=2, query="LLM reasoning", source_docs=["paper2.pdf"]),
        ]

    def test_parses_executive_summary(self):
        summary, _, _ = _parse_report(self.SAMPLE_REPORT, self._make_subtasks())
        assert "executive summary" in summary.lower()

    def test_parses_sections(self):
        _, sections, _ = _parse_report(self.SAMPLE_REPORT, self._make_subtasks())
        assert len(sections) == 2
        assert "RAG" in sections[0].title

    def test_parses_confidence_score(self):
        _, _, score = _parse_report(self.SAMPLE_REPORT, self._make_subtasks())
        assert abs(score - 0.82) < 0.01

    def test_fallback_on_empty_input(self):
        subtasks = self._make_subtasks()
        subtasks[0] = subtasks[0].model_copy(update={"summary": "fallback summary"})
        summary, sections, score = _parse_report("", subtasks)
        assert len(sections) == 2
        assert score == 0.5

    def test_confidence_clamped(self):
        bad_report = self.SAMPLE_REPORT.replace("Score: 0.82", "Score: 1.99")
        _, _, score = _parse_report(bad_report, self._make_subtasks())
        assert score <= 1.0


class TestReviewFlagging:
    def test_flags_low_confidence(self):
        from app.config import get_settings
        settings = get_settings()
        threshold = settings.confidence_threshold
        subtasks = [SubTask(id=1, query="q", retrieved_chunks=[], retrieval_score=0.0)]
        score = _compute_confidence(subtasks, llm_score=0.1)
        assert score < threshold

    def test_approves_high_confidence(self):
        from app.config import get_settings
        settings = get_settings()
        threshold = settings.confidence_threshold
        subtasks = [
            SubTask(id=1, query="q", retrieved_chunks=["c1", "c2"], retrieval_score=0.92)
        ]
        score = _compute_confidence(subtasks, llm_score=0.88)
        assert score >= threshold
