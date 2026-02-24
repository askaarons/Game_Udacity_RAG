from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.agent.reporting import build_agent_report
from src.agent.state_machine import GameRAGAgent
from src.rag.data_processor import load_game_records, to_documents
from src.rag.vector_store_manager import VectorStoreManager


def ingest_local_data(project_root: Path, collection_name: str) -> VectorStoreManager:
    data_dir = project_root / "data" / "raw"
    records = load_game_records(data_dir)
    documents = to_documents(records)

    vector_store = VectorStoreManager(
        persist_directory=project_root / "chroma_db",
        collection_name=collection_name,
    )
    inserted = vector_store.add_documents(documents)

    print(f"Loaded records: {len(records)}")
    print(f"Inserted/updated documents: {inserted}")
    print(f"Collection count: {vector_store.count()}")
    return vector_store


def run_queries(agent: GameRAGAgent, queries: list[str]) -> None:
    for index, query in enumerate(queries, start=1):
        turn = agent.run(query)
        report = build_agent_report(asdict(turn))
        print("=" * 100)
        print(f"Query {index}: {query}")
        print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Game Udacity RAG + Agent demo")
    parser.add_argument(
        "--collection",
        default="games",
        help="Chroma collection name (default: games)",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Provide one or more custom queries. Repeat --query for multiple values.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web fallback and answer only from internal retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    default_queries = [
        "When was Elden Ring released and on which platforms?",
        "Who published Stardew Valley?",
        "What can you tell me about GTA VI release window?",
    ]

    queries = args.queries if args.queries else default_queries

    vector_store = ingest_local_data(project_root=project_root, collection_name=args.collection)
    agent = GameRAGAgent(vector_store=vector_store, allow_web_fallback=not args.no_web)
    run_queries(agent=agent, queries=queries)


if __name__ == "__main__":
    main()
