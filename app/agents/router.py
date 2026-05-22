import json
import logging
import re

from langchain_anthropic import ChatAnthropic

from app.agents.state import ResearchState
from app.config import get_settings
from app.models import SubTask

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a research planning agent. Your job is to decompose a complex research question into 2-3 specific, focused subtasks that together answer the original question.

Rules:
- Generate exactly 2 or 3 subtasks
- Each subtask must be a concrete search query that can be answered with retrieved document chunks
- Subtasks must be distinct and non-overlapping
- Return ONLY a valid JSON array, no explanation

Format:
[
  {"id": 1, "query": "specific search query 1"},
  {"id": 2, "query": "specific search query 2"},
  {"id": 3, "query": "specific search query 3"}
]"""


def router_agent(state: ResearchState) -> dict:
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        max_tokens=512,
    )

    question = state["question"]
    logger.info(f"[Router] Decomposing question: {question}")

    messages = [
        {"role": "user", "content": f"Decompose this research question into subtasks:\n\n{question}"},
    ]

    try:
        response = llm.invoke(
            messages,
            system=ROUTER_SYSTEM_PROMPT,
        )
        raw = response.content.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON array found in router response: {raw}")

        subtask_data = json.loads(json_match.group())
        subtasks = [SubTask(id=item["id"], query=item["query"]) for item in subtask_data]
        logger.info(f"[Router] Created {len(subtasks)} subtasks")
        return {"subtasks": subtasks}
    except Exception as e:
        logger.error(f"[Router] Error: {e}")
        fallback = [
            SubTask(id=1, query=question),
            SubTask(id=2, query=f"key concepts and methods related to: {question}"),
        ]
        return {"subtasks": fallback}
