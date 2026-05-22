from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator

from app.models import SubTask, ReportSection


class ResearchState(TypedDict):
    question: str
    top_k: int
    subtasks: Annotated[list[SubTask], operator.add]
    sections: Annotated[list[ReportSection], operator.add]
    executive_summary: str
    confidence_score: float
    review_status: str
    review_reason: Optional[str]
    error: Optional[str]
