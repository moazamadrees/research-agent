import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.models import (
    ResearchReport,
    ReviewStatus,
    SubTask,
    ReportSection,
    IngestResponse,
    HealthResponse,
)


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def client(mock_settings):
    with patch("app.ingestion.get_collection_count", return_value=42):
        with patch("app.ingestion.ingest_directory", return_value=(0, 0)):
            from app.main import app
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["chroma_documents"] == 42


class TestResearchEndpoint:
    def _make_mock_report(self):
        return ResearchReport(
            question="What is RAG?",
            subtasks=[
                SubTask(
                    id=1,
                    query="RAG architecture",
                    retrieved_chunks=["chunk1"],
                    source_docs=["paper.pdf"],
                    summary="RAG combines retrieval and generation.",
                    retrieval_score=0.85,
                )
            ],
            sections=[
                ReportSection(
                    title="RAG Architecture",
                    content="Detailed findings...",
                    sources=["paper.pdf"],
                )
            ],
            executive_summary="RAG improves LLM factual accuracy.",
            confidence_score=0.82,
            review_status=ReviewStatus.AUTO_APPROVED,
            review_reason=None,
            model_used="claude-sonnet-4-5-20250929",
        )

    def test_research_returns_report(self, client):
        mock_report = self._make_mock_report()
        with patch(
            "app.main.run_research_pipeline",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            response = client.post("/research", json={"question": "What is RAG and how does it work?"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is RAG?"
        assert data["confidence_score"] == 0.82
        assert data["review_status"] == "auto_approved"

    def test_research_question_too_short(self, client):
        response = client.post("/research", json={"question": "RAG?"})
        assert response.status_code == 422

    def test_research_pipeline_error_returns_500(self, client):
        with patch(
            "app.main.run_research_pipeline",
            new_callable=AsyncMock,
            side_effect=Exception("LLM error"),
        ):
            response = client.post("/research", json={"question": "What is retrieval augmented generation?"})
        assert response.status_code == 500

    def test_flagged_report_has_review_reason(self, client):
        mock_report = self._make_mock_report()
        mock_report.confidence_score = 0.40
        mock_report.review_status = ReviewStatus.FLAGGED_FOR_REVIEW
        mock_report.review_reason = "Low confidence due to sparse retrieval."

        with patch(
            "app.main.run_research_pipeline",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            response = client.post("/research", json={"question": "What is retrieval augmented generation?"})
        assert response.status_code == 200
        data = response.json()
        assert data["review_status"] == "flagged_for_review"
        assert data["review_reason"] is not None
