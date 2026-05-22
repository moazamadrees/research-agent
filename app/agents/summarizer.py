import logging

from langchain_anthropic import ChatAnthropic

from app.agents.state import ResearchState
from app.config import get_settings
from app.models import SubTask

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are a research summarizer. Given a research subtask and retrieved document chunks, synthesize the information into a coherent, accurate summary.

Rules:
- Summarize only what is supported by the provided context
- Be specific and technical where appropriate
- Note any gaps or contradictions in the retrieved information
- Keep the summary focused on the subtask query
- Length: 2-4 paragraphs"""


def _build_context_block(subtask: SubTask) -> str:
    if not subtask.retrieved_chunks:
        return "No relevant documents were retrieved for this subtask."
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1} from {subtask.source_docs[i % len(subtask.source_docs)] if subtask.source_docs else 'unknown'}]:\n{chunk}"
        for i, chunk in enumerate(subtask.retrieved_chunks)
    )
    return chunks_text


def summarizer_agent(state: ResearchState) -> dict:
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
        max_tokens=1024,
    )

    subtasks: list[SubTask] = state["subtasks"]
    updated_subtasks = []

    for subtask in subtasks:
        logger.info(f"[Summarizer] Summarizing subtask {subtask.id}: {subtask.query}")
        context = _build_context_block(subtask)

        prompt = (
            f"Research Subtask: {subtask.query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Write a coherent summary of findings for this subtask based on the retrieved context above."
        )

        try:
            response = llm.invoke(
                [{"role": "user", "content": prompt}],
                system=SUMMARIZER_SYSTEM_PROMPT,
            )
            summary = response.content.strip()
        except Exception as e:
            logger.error(f"[Summarizer] Error on subtask {subtask.id}: {e}")
            summary = f"Summary unavailable due to error: {e}"

        updated = subtask.model_copy(update={"summary": summary})
        updated_subtasks.append(updated)

    return {"subtasks": updated_subtasks}
