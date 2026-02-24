from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agent.tools import evaluate_retrieval, game_web_search, retrieve_game
from src.rag.vector_store_manager import VectorStoreManager


class AgentState(str, Enum):
    START = "start"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    WEB_SEARCH = "web_search"
    ANSWER = "answer"
    DONE = "done"


@dataclass
class AgentTurn:
    query: str
    state_trace: list[str] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    citations: list[str] = field(default_factory=list)


class GameRAGAgent:
    """Stateful agent that prefers internal retrieval then falls back to web."""

    def __init__(self, vector_store: VectorStoreManager, allow_web_fallback: bool = True) -> None:
        self.vector_store = vector_store
        self.allow_web_fallback = allow_web_fallback
        self.history: list[AgentTurn] = []

    def _build_internal_answer(self, query: str, retrieval_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not retrieval_results:
            return "I could not find enough internal data to answer this query.", []

        top = retrieval_results[0]
        title = top.get("metadata", {}).get("title", "Unknown")
        text = top.get("text", "")
        source = top.get("metadata", {}).get("source_file", "local_dataset")

        answer = (
            f"Based on the local game dataset, the strongest match for '{query}' is '{title}'. "
            f"Relevant context: {text}"
        )
        citations = [f"local:{source}"]
        return answer, citations

    def _build_web_answer(self, query: str, web_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not web_results:
            return (
                "I could not verify this from internal data, and web fallback returned no results.",
                [],
            )

        top = web_results[0]
        answer = (
            f"I used web fallback for '{query}'. Top source suggests: {top.get('content', 'No content')}"
        )
        citations = [result.get("url", "") for result in web_results if result.get("url")]
        return answer, citations

    def _build_no_web_answer(self, query: str, retrieval_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not retrieval_results:
            return (
                "Internal retrieval quality was low and web fallback is disabled, so I cannot answer confidently.",
                [],
            )

        top = retrieval_results[0]
        title = top.get("metadata", {}).get("title", "Unknown")
        text = top.get("text", "")
        source = top.get("metadata", {}).get("source_file", "local_dataset")
        answer = (
            f"Web fallback is disabled. Best low-confidence internal match for '{query}' is '{title}'. "
            f"Context: {text}"
        )
        return answer, [f"local:{source}"]

    def run(self, query: str) -> AgentTurn:
        state = AgentState.START
        turn = AgentTurn(query=query)

        retrieval_payload: dict[str, Any] = {}
        eval_payload: dict[str, Any] = {}
        web_payload: dict[str, Any] = {}

        while state != AgentState.DONE:
            turn.state_trace.append(state.value)

            if state == AgentState.START:
                state = AgentState.RETRIEVE
                continue

            if state == AgentState.RETRIEVE:
                retrieval_payload = retrieve_game(query=query, vector_store=self.vector_store)
                turn.tool_outputs.append(retrieval_payload)
                state = AgentState.EVALUATE
                continue

            if state == AgentState.EVALUATE:
                eval_payload = evaluate_retrieval(query=query, retrieval_payload=retrieval_payload)
                turn.tool_outputs.append(eval_payload)
                if eval_payload.get("decision") == "use_internal":
                    state = AgentState.ANSWER
                elif self.allow_web_fallback:
                    state = AgentState.WEB_SEARCH
                else:
                    turn.tool_outputs.append(
                        {
                            "tool": "game_web_search",
                            "query": query,
                            "results": [],
                            "error": "Web fallback disabled by agent configuration.",
                        }
                    )
                    state = AgentState.ANSWER
                continue

            if state == AgentState.WEB_SEARCH:
                web_payload = game_web_search(query=query)
                turn.tool_outputs.append(web_payload)
                state = AgentState.ANSWER
                continue

            if state == AgentState.ANSWER:
                if eval_payload.get("decision") == "use_internal":
                    turn.final_answer, turn.citations = self._build_internal_answer(
                        query=query,
                        retrieval_results=retrieval_payload.get("results", []),
                    )
                elif not self.allow_web_fallback:
                    turn.final_answer, turn.citations = self._build_no_web_answer(
                        query=query,
                        retrieval_results=retrieval_payload.get("results", []),
                    )
                else:
                    turn.final_answer, turn.citations = self._build_web_answer(
                        query=query,
                        web_results=web_payload.get("results", []),
                    )
                state = AgentState.DONE
                continue

        turn.state_trace.append(AgentState.DONE.value)
        self.history.append(turn)
        return turn
