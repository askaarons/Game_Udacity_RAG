# Game_Udacity_RAG

Udacity-style capstone implementation for:

- Part 1: RAG pipeline with ChromaDB
- Part 2: Stateful agent with retrieval evaluation and web-search fallback

## Project Structure

- `Udaplay_01_solution_project.ipynb` — Dataset processing, embeddings, and semantic search demo
- `Udaplay_02_solution_project.ipynb` — Agent workflow demo with three example queries
- `data/raw/*.json` — Local video game dataset files
- `src/rag/data_processor.py` — JSON loading and document preparation
- `src/rag/vector_store_manager.py` — Reusable ChromaDB vector store manager
- `src/agent/tools.py` — `retrieve_game`, `evaluate_retrieval`, `game_web_search`
- `src/agent/state_machine.py` — Stateful workflow/state machine agent
- `src/agent/reporting.py` — Structured output/report formatter

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional for web fallback) set Tavily API key:

```bash
export TAVILY_API_KEY="your_api_key"
```

## Run Order

1. Run `Udaplay_01_solution_project.ipynb` to ingest local JSON into persistent ChromaDB (`chroma_db/`).
2. Run `Udaplay_02_solution_project.ipynb` to execute the stateful agent on at least three queries.

## Quick CLI Demo

Run the full flow (ingest + agent queries) from terminal:

```bash
python run_demo.py
```

Optional custom queries:

```bash
python run_demo.py --query "Which game is from Nintendo?" --query "Who developed Hades?"
```

Disable web fallback (internal retrieval only):

```bash
python run_demo.py --no-web
```

## Rubric Mapping

### RAG

- Local game JSON files are loaded and normalized in `src/rag/data_processor.py`.
- Processed records are embedded and persisted in ChromaDB via `VectorStoreManager`.
- Semantic search is demonstrated in `Udaplay_01_solution_project.ipynb`.

### Agent Development

- Three tools are implemented and integrated:
	- `retrieve_game`
	- `evaluate_retrieval`
	- `game_web_search`
- Stateful workflow is implemented as explicit states in `GameRAGAgent`.
- Multi-query memory is maintained with `agent.history`.
- Structured report output includes:
	- state trace
	- tool usage
	- final answer
	- citations

