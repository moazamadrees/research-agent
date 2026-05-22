import logging

from app.agents.state import ResearchState
from app.ingestion import retrieve_relevant_chunks
from app.models import SubTask

logger = logging.getLogger(__name__)


def retriever_agent(state: ResearchState) -> dict:
    subtasks: list[SubTask] = state["subtasks"]
    top_k: int = state.get("top_k", 5)
    updated_subtasks = []

    for subtask in subtasks:
        logger.info(f"[Retriever] Querying for subtask {subtask.id}: {subtask.query}")
        chunks = retrieve_relevant_chunks(query=subtask.query, top_k=top_k)

        if chunks:
            retrieved_texts = [c["text"] for c in chunks]
            source_docs = list({c["source"] for c in chunks})
            avg_similarity = sum(c["similarity"] for c in chunks) / len(chunks)

            updated = subtask.model_copy(
                update={
                    "retrieved_chunks": retrieved_texts,
                    "source_docs": source_docs,
                    "retrieval_score": round(avg_similarity, 4),
                }
            )
        else:
            logger.warning(f"[Retriever] No chunks found for subtask {subtask.id}")
            updated = subtask.model_copy(
                update={
                    "retrieved_chunks": [],
                    "source_docs": [],
                    "retrieval_score": 0.0,
                }
            )

        updated_subtasks.append(updated)

    return {"subtasks": updated_subtasks}
