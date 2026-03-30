# ✿ Tangent & Angelica ✿ — Testing Analysis and Agentic Labeling for Agent Systems

This repository contains two complementary tools for analyzing and labeling test code in agentic systems:

1. **Tangent** — Static analysis tool for detecting agents, frameworks, tools, and tests in Python repositories
2. **Angelica** — Agentic labeling system for automated code annotation with LLM-powered agents

---

## 📁 Repository Structure

```
.
├── tangent/                    # Static analysis tool
│   ├── agent_analysis/        # Agent, framework, tool, and test detection
│   ├── code_analysis/         # Code complexity and test method analysis
│   └── utils/                 # Utilities and logging
│
├── agentic_labeler/           # Agentic labeling system
│   ├── angelica/              # Core labeling library
│   │   ├── agents/           # Multi-agent labeling system
│   │   ├── llm_client/       # LLM integration and token counting
│   │   ├── metrics/          # Agreement metrics (Cohen's kappa)
│   │   ├── parallel/         # Ray-based parallel processing
│   │   ├── post_labeling/    # Pattern extraction and label refinement
│   │   ├── prompts/          # Prompt templates
│   │   └── storage/          # SQLite + FAISS vector storage
│   ├── tangent_label/        # Tangent-specific labeling configuration
│   └── scripts/              # Example scripts
│
├── Dataset/                   # Research datasets
│   ├── Phase_1/              # Initial data collection
│   ├── Phase_2/              # Static analysis results
│   └── Phase_3/              # Final labeled results
│
├── Label/                     # Labeling artifacts
├── Figures/                   # Research figures and visualizations
└── Supplementary_Documents/   # Additional documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (Angelica) or Python 3.13+ (Tangent)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

Each tool is managed independently with `uv`:

#### Tangent
```bash
cd tangent
uv sync
source .venv/bin/activate
```

#### Angelica
```bash
cd agentic_labeler
uv sync
source .venv/bin/activate
```

---

## 🔍 Tangent — Static Analysis for Agentic Systems

Tangent performs static analysis on Python repositories to detect:
- **Agents**: Classes and functions that implement agentic behavior
- **Frameworks**: LangChain, LangGraph, CrewAI, AutoGen, etc.
- **Tools**: Functions and methods used by agents
- **Tests**: Test methods that exercise agent functionality

### Features

- Multiple analysis backends (CLDK/Scalpel, CodeQL)
- Call graph analysis with configurable depth
- Test complexity metrics
- Assertion type detection
- Framework-specific pattern detection

### Usage

Generate `analysis.json` for a target repository:

```bash
tangent analyze --repo /path/to/python/repo --out analysis.json
```


---

## 🤖 Angelica — Agentic Labeling System

Angelica is a library and CLI for building agentic labeling systems with multiple independent agents, adjudication, and retrieval-augmented prompting.

### Key Features

- **Multi-agent labeling** with configurable agents (A, B, adjudicator)
- **Adjudication on disagreement** with customizable logic
- **Persistent storage** (SQLite + FAISS vector index)
- **Retrieval-augmented prompting** via embeddings
- **Agreement metrics** (rolling Cohen's kappa)
- **Pattern learning** from unmatched cases
- **Parallel processing** with Ray
- **Tool-agnostic** analysis support

### Installation & Setup

1. **Install dependencies:**
   ```bash
   cd agentic_labeler
   uv sync
   source .venv/bin/activate
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   
   Set required variables:
   - `API_KEY` — LLM provider API key
   - `BASE_URL` — Provider base URL
   
   Optional overrides:
   - `LLM_MODEL` — Default: `openai/gpt-4o`
   - `EMBEDDINGS_MODEL` — Default: `text-embedding-3-small`
   - `LABELER_A_MODEL`, `LABELER_B_MODEL`, `ADJUDICATOR_MODEL`

### Usage

#### 1. File-based Labeling

Label entire files in a directory:

```bash
angelica label-dir \
  --config path/to/config.py \
  --path /path/to/code \
  --suffix .py \
  --db labels.db \
  --index-dir vector_index \
  --out results.json
```

**Options:**
- `--fresh-build` — Rebuild database and index from scratch
- `--parallel` — Enable parallel processing with Ray
- `--num-workers 8` — Number of parallel workers
- `--rate-limit-rpm 1000` — Rate limit in requests per minute



### Configuration

Create a configuration file defining your labeling schema:

```python
from angelica.models.config import AgenticConfig
from pydantic import BaseModel

class MyLabel(BaseModel):
    pattern_name: str
    confidence: float
    reasoning: str

CONFIG = AgenticConfig(
    schema=MyLabel,
    patterns="Your taxonomy/pattern descriptions here...",
    # Optional: custom prompt templates, agreement logic, etc.
)
```


---

## 📊 Research Artifacts

### Dataset

The `Dataset/` directory contains research data organized in phases.

### Labels

The `Label/` directory contains:
- `applications.zip`: Labeled application code
- `final_labels.xlsx`: Consolidated labeling results

### Figures

The `Figures/` directory contains research visualizations:
- Assertion type distributions
- Framework testing patterns
- Structural complexity metrics
- Model architecture diagrams
- RQ2 consolidated results

### Supplementary Documents

Additional research documentation in `Supplementary_Documents/supplementary.pdf`

---

## 📦 Dataset Access

The complete dataset used in this research is available on Figshare:

**DOI:** [10.6084/m9.figshare.31883329](https://figshare.com/articles/dataset/Tangent/31883329?file=63288853)
