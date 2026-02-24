from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from src.rag.vector_store_manager import VectorStoreManager


def retrieve_game(
    query: str,
    vector_store: VectorStoreManager,
    top_k: int = 5,
) -> dict[str, Any]:
    """Tool 1: Retrieve candidate context from local vector DB."""

    results = vector_store.semantic_search(query=query, top_k=top_k)
    return {
        "query": query,
        "tool": "retrieve_game",
        "results": results,
    }


def evaluate_retrieval(
    query: str,
    retrieval_payload: dict[str, Any],
    min_similarity: float = 0.45,
    min_results: int = 1,
) -> dict[str, Any]:
    """Tool 2: Decide whether retrieval quality is sufficient for answering."""

    results = retrieval_payload.get("results", [])
    if not results:
        return {
            "tool": "evaluate_retrieval",
            "decision": "fallback",
            "reason": "No vector search results returned.",
            "avg_similarity": 0.0,
        }

    avg_similarity = sum(item.get("similarity", 0.0) for item in results) / len(results)
    passes_threshold = avg_similarity >= min_similarity and len(results) >= min_results

    return {
        "tool": "evaluate_retrieval",
        "query": query,
        "decision": "use_internal" if passes_threshold else "fallback",
        "reason": (
            "Retrieved context quality is acceptable."
            if passes_threshold
            else "Retrieved context quality is below threshold."
        ),
        "avg_similarity": round(avg_similarity, 4),
        "num_results": len(results),
    }


def game_web_search(query: str, max_results: int = 3) -> dict[str, Any]:
    """Tool 3: Fallback web search using Tavily."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "tool": "game_web_search",
            "query": query,
            "results": [],
            "error": "Missing TAVILY_API_KEY environment variable.",
        }

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"video game information: {query}",
        max_results=max_results,
        search_depth="advanced",
    )

    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "content": item.get("content"),
                "url": item.get("url"),
            }
        )

    return {
        "tool": "game_web_search",
        "query": query,
        "results": results,
    }
