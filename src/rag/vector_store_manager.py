from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .data_processor import GameDocument


class VectorStoreManager:
    """Reusable manager for a persistent ChromaDB game collection."""

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "games",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Video game knowledge base"},
        )

    def add_documents(self, documents: list[GameDocument]) -> int:
        """Upsert documents into Chroma and return inserted count."""

        if not documents:
            return 0

        ids = [document.doc_id for document in documents]
        texts = [document.text for document in documents]
        metadatas = [document.metadata for document in documents]
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(documents)

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Run semantic search and return normalized result rows."""

        response = self.collection.query(query_texts=[query], n_results=top_k)

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        rows: list[dict[str, Any]] = []
        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            rows.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                    "similarity": round(similarity, 4),
                }
            )
        return rows

    def count(self) -> int:
        return self.collection.count()
