# ✿ 𝓐𝓷𝓰𝓮𝓵𝓲𝓬𝓪 ✿: Library for Agent-based Labeling

This project provides a **library and CLI** for building **agentic labeling systems** with:

- Multiple independent angelicas
- Optional adjudication on disagreement
- Persistent storage of documents and labels
- Retrieval-augmented prompting via embeddings
- Support static analysis using CLDK
- Agreement metrics (rolling Cohen's kappa)
- **🆕 Enhanced vector matching with confidence scores**
- **🆕 Automatic pattern learning for unmatched cases**
- **🆕 Pattern evolution based on accumulated examples**

The system is **fully configurable** and **analysis-tool-agnostic**.
You can label entire files, individual methods, classes, or any custom “unit” you define.

---

## Key features

- Plug-in Pydantic output schema
- Plug-in pattern / taxonomy text
- Custom prompt templates (angelicas + adjudicator)
- Optional custom agreement logic
- Optional retrieval example formatting
- SQLite persistence (documents, per-agent labels, final labels)
- FAISS vector index for similarity retrieval
- Rolling Cohen's kappa over any JSON field path
- File-based and unit-based labeling (methods, classes, IDs, etc.)
- Tool-agnostic analysis support via shared context
- **🆕 Enhanced vector matching with configurable confidence thresholds**
- **🆕 Automatic pattern storage and learning**
- **🆕 Pattern evolution and statistics tracking**

---

## Installation

```bash
uv sync
source .venv/bin/activate
```

---

## Environment variables

Copy the example file:

```bash
cp .env.example .env
```

Set at minimum:

- API_KEY – LLM provider API key
- BASE_URL – provider base URL

Optional overrides:

- LLM_MODEL – default: openai/gpt-4o
- EMBEDDINGS_MODEL – default: text-embedding-3-small
- angelica_A_MODEL, angelica_B_MODEL, ADJUDICATOR_MODEL

---

## Quickstart (CLI)

### CLI overview

```bash
angelica --help
```

Commands:

- label-dir
- label-units
- plot-kappa

---