from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GameDocument:
    """A normalized game document ready for vector storage."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def build_game_text(game: dict[str, Any]) -> str:
    """Build a high-signal text block for embedding from raw game JSON."""

    title = _stringify(game.get("title") or game.get("name"))
    description = _stringify(game.get("description") or game.get("summary"))
    genres = _stringify(game.get("genres") or game.get("genre"))
    platforms = _stringify(game.get("platforms") or game.get("platform"))
    publisher = _stringify(game.get("publisher") or game.get("publishers"))
    developer = _stringify(game.get("developer") or game.get("developers"))
    release_date = _stringify(game.get("release_date") or game.get("released"))

    lines = [
        f"Title: {title}",
        f"Description: {description}",
        f"Genres: {genres}",
        f"Platforms: {platforms}",
        f"Publisher: {publisher}",
        f"Developer: {developer}",
        f"Release Date: {release_date}",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[1])


def load_game_records(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load all JSON records from a local directory.

    Supports either:
    - one JSON object per file, or
    - list of JSON objects per file.
    """

    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for file_path in sorted(data_path.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("_source_file", file_path.name)
                    item.setdefault("_source_index", index)
                    records.append(item)
        elif isinstance(payload, dict):
            item = dict(payload)
            item.setdefault("_source_file", file_path.name)
            item.setdefault("_source_index", 0)
            records.append(item)

    return records


def to_documents(records: list[dict[str, Any]]) -> list[GameDocument]:
    """Convert raw records into normalized `GameDocument` objects."""

    documents: list[GameDocument] = []
    for index, game in enumerate(records):
        title = _stringify(game.get("title") or game.get("name") or f"game-{index}")
        source_file = _stringify(game.get("_source_file") or "unknown")
        source_index = _stringify(game.get("_source_index") or index)
        doc_id = f"{source_file}:{source_index}:{title}".lower().replace(" ", "-")
        text = build_game_text(game)
        metadata = {
            "title": title,
            "platforms": _stringify(game.get("platforms") or game.get("platform")),
            "publisher": _stringify(game.get("publisher") or game.get("publishers")),
            "release_date": _stringify(game.get("release_date") or game.get("released")),
            "source_file": source_file,
            "source_index": int(source_index) if str(source_index).isdigit() else source_index,
        }
        documents.append(GameDocument(doc_id=doc_id, text=text, metadata=metadata))

    return documents
