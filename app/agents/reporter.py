import logging

from langchain_anthropic import ChatAnthropic

from app.agents.state import ResearchState
from app.config import get_settings
from app.models import ReportSection, ReviewStatus, SubTask

logger = logging.getLogger(__name__)

REPORTER_SYSTEM_PROMPT = """You are a research report writer. Given multiple research subtask summaries, generate a structured final report.

You must return your response in exactly this format:

## Executive Summary
[2-3 sentence high-level summary of all findings]

## Section: [Title for subtask 1]
[Detailed findings for subtask 1]

## Section: [Title for subtask 2]
[Detailed findings for subtask 2]

(Repeat for all subtasks)

## Confidence Assessment
Score: [a decimal between 0.0 and 1.0]
Reasoning: [1-2 sentences explaining the score based on source quality and coverage]

Rules:
- Confidence score should reflect how well the evidence supports the conclusions
- High confidence (>0.75): Strong, consistent evidence from multiple sources
- Medium confidence (0.5-0.75): Partial evidence or some gaps
- Low confidence (<0.5): Sparse or contradictory evidence"""


def _parse_report(raw: str, subtasks: list[SubTask]) -> tuple[str, list[ReportSection], float]:
    import re

    executive_summary = ""
    sections = []
    confidence_score = 0.5

    exec_match = re.search(r"## Executive Summary\s*\n(.*?)(?=\n##|\Z)", raw, re.DOTALL)
    if exec_match:
        executive_summary = exec_match.group(1).strip()

    section_matches = re.finditer(
        r"## Section:\s*(.+?)\s*\n(.*?)(?=\n## |\Z)", raw, re.DOTALL
    )
    for i, match in enumerate(section_matches):
        title = match.group(1).strip()
        content = match.group(2).strip()
        sources = subtasks[i].source_docs if i < len(subtasks) else []
        sections.append(ReportSection(title=title, content=content, sources=sources))

    conf_match = re.search(r"Score:\s*([0-9.]+)", raw)
    if conf_match:
        try:
            confidence_score = float(conf_match.group(1))
            confidence_score = max(0.0, min(1.0, confidence_score))
        except ValueError:
            confidence_score = 0.5

    if not executive_summary:
        executive_summary = "Research findings compiled from retrieved documents."

    if not sections:
        for subtask in subtasks:
            sections.append(
                ReportSection(
                    title=f"Subtask {subtask.id}: {subtask.query[:60]}",
                    content=subtask.summary or "No summary available.",
                    sources=subtask.source_docs,
                )
            )

    return executive_summary, sections, confidence_score


def _compute_confidence(subtasks: list[SubTask], llm_score: float) -> float:
    if not subtasks:
        return 0.0

    retrieval_scores = [t.retrieval_score for t in subtasks]
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)

    coverage = sum(1 for t in subtasks if t.retrieved_chunks) / len(subtasks)

    blended = (llm_score * 0.5) + (avg_retrieval * 0.3) + (coverage * 0.2)
    return round(min(1.0, max(0.0, blended)), 4)


def reporter_agent(state: ResearchState) -> dict:
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        max_tokens=2048,
    )

    subtasks: list[SubTask] = state["subtasks"]
    question = state["question"]

    subtask_summaries = "\n\n".join(
        f"=== Subtask {t.id}: {t.query} ===\n{t.summary or 'No summary available.'}"
        for t in subtasks
    )

    prompt = (
        f"Original Research Question: {question}\n\n"
        f"Subtask Summaries:\n{subtask_summaries}\n\n"
        f"Generate a structured research report following the specified format."
    )

    logger.info("[Reporter] Generating final report")
    try:
        response = llm.invoke(
            [{"role": "user", "content": prompt}],
            system=REPORTER_SYSTEM_PROMPT,
        )
        raw = response.content.strip()
    except Exception as e:
        logger.error(f"[Reporter] Error: {e}")
        raw = ""

    executive_summary, sections, llm_confidence = _parse_report(raw, subtasks)
    final_confidence = _compute_confidence(subtasks, llm_confidence)

    if final_confidence < settings.confidence_threshold:
        review_status = ReviewStatus.FLAGGED_FOR_REVIEW
        review_reason = (
            f"Confidence score {final_confidence:.2f} is below threshold "
            f"{settings.confidence_threshold}. Possible causes: limited source coverage "
            f"(avg retrieval score: {sum(t.retrieval_score for t in subtasks)/max(len(subtasks),1):.2f}), "
            f"or insufficient evidence in the document corpus."
        )
    else:
        review_status = ReviewStatus.AUTO_APPROVED
        review_reason = None

    logger.info(
        f"[Reporter] Report complete. Confidence: {final_confidence:.2f}, Status: {review_status}"
    )

    return {
        "sections": sections,
        "executive_summary": executive_summary,
        "confidence_score": final_confidence,
        "review_status": review_status,
        "review_reason": review_reason,
    }
