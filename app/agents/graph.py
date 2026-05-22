import logging
from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.state import ResearchState
from app.agents.router import router_agent
from app.agents.retriever import retriever_agent
from app.agents.summarizer import summarizer_agent
from app.agents.reporter import reporter_agent
from app.config import get_settings
from app.models import ResearchReport, ReviewStatus

logger = logging.getLogger(__name__)


def _check_for_error(state: ResearchState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


def _build_graph() -> Any:
    graph = StateGraph(ResearchState)

    graph.add_node("router", router_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("reporter", reporter_agent)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _check_for_error,
        {"continue": "retriever", "error": END},
    )
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_research_pipeline(question: str, top_k: int = 5) -> ResearchReport:
    settings = get_settings()
    graph = get_graph()

    initial_state: ResearchState = {
        "question": question,
        "top_k": top_k,
        "subtasks": [],
        "sections": [],
        "executive_summary": "",
        "confidence_score": 0.0,
        "review_status": ReviewStatus.AUTO_APPROVED,
        "review_reason": None,
        "error": None,
    }

    logger.info(f"[Pipeline] Starting research pipeline for: {question}")
    final_state = await graph.ainvoke(initial_state)

    subtasks = final_state.get("subtasks", [])
    sections = final_state.get("sections", [])

    unique_subtasks = list({t.id: t for t in subtasks}.values())
    unique_sections = []
    seen_titles = set()
    for s in sections:
        if s.title not in seen_titles:
            unique_sections.append(s)
            seen_titles.add(s.title)

    report = ResearchReport(
        question=question,
        subtasks=unique_subtasks,
        sections=unique_sections,
        executive_summary=final_state.get("executive_summary", ""),
        confidence_score=final_state.get("confidence_score", 0.0),
        review_status=final_state.get("review_status", ReviewStatus.AUTO_APPROVED),
        review_reason=final_state.get("review_reason"),
        model_used=settings.llm_model,
    )

    logger.info(f"[Pipeline] Complete. Status: {report.review_status}")
    return report
