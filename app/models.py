from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ReviewStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class SubTask(BaseModel):
    id: int
    query: str
    retrieved_chunks: list[str] = Field(default_factory=list)
    source_docs: list[str] = Field(default_factory=list)
    summary: str = ""
    retrieval_score: float = 0.0


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=10, description="The research question to investigate")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class ReportSection(BaseModel):
    title: str
    content: str
    sources: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    question: str
    subtasks: list[SubTask]
    sections: list[ReportSection]
    executive_summary: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    review_status: ReviewStatus
    review_reason: Optional[str] = None
    model_used: str


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_stored: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    chroma_documents: int
    model: str
