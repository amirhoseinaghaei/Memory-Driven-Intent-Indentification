# Memory-Driven Intent Identification for Medical Diagnosis

> Official implementation for the NeurIPS paper:
> **"Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference"**
> [[Paper](#)] 

---

## Abstract

While scaling large language models has driven substantial gains in linguistic fluency, building reliable autonomous agents requires more than larger parameter counts. Current agentic systems often produce semantically plausible yet structurally ungrounded reasoning, leading to unreliable multi-turn inference under partial and noisy observations. We introduce \textbf{Structure-Grounded Uncertainty Reduction (SGUR)}, a training-free framework that reformulates sequential reasoning as uncertainty reduction over a structured latent hypothesis space. SGUR organizes domain knowledge into a hierarchical four-layer graph and aligns fragmented evidence to candidate hypothesis graphs through a Partial Fused Gromov--Wasserstein discrepancy, which jointly captures semantic and relational correspondence while tolerating incomplete observations through an overlap-aware mass budget. This geometric alignment is coupled with an information-gain-driven acquisition strategy that actively selects the most discriminative next observation, and with conformal risk control for calibrated, uncertainty-aware predictions.Evaluated on a demanding sequential clinical diagnosis benchmark derived from PrimeKG, SGUR achieves \textbf{92.5\% Hit@3} under sparse two-symptom queries, a $9.4$-point absolute gain over the strongest graph-RAG baseline, while consuming roughly $5\times$ fewer tokens and converging in fewer interaction rounds (MTTC $1.62$ vs.\ $\geq 1.74$). Component-wise ablations confirm that latent routing, partial transport, and information-theoretic node weighting each contribute substantially, with their effect most pronounced in data-sparse regimes. Notably, SGUR with a small backbone (Haiku~3, GPT-5.4-nano) matches or exceeds the accuracy of every baseline run on substantially larger models, retaining over $99\%$ of its Hit@3 across model scales. These results suggest that reliable agent performance depends not only on model scale, but on the structural quality of the reasoning framework organizing inference.
---

## Method Overview

This repository implements a memory-driven, multi-round conversational agent for rare disease identification from natural language symptom descriptions. The system models patient-reported symptoms as a directed graph and compares it against a disease knowledge graph using a  **Partial Fused Gromov-Wasserstein (PFLGW)** distance metric.

Key components:

- **Knowledge Graph**: Neo4j-backed graph encoding diseases, phenotypes, anatomical locations, genes, and drugs with typed edges.
- **Graph-Based Retrieval**: Retrieval is framed as an optimal transport problem. The PFLGW distance compares the patient symptom graph (partial, noisy) against each disease reference graph (complete).
- **Memory-Driven Agent**: A LangGraph agent maintains a session memory of previously seen symptom clusters and candidate diseases across dialogue turns.
- **Multi-Round Evaluation**: Three successive queries simulate a realistic clinical dialogue, with cumulative top-k accuracy measured at each round.

---

## Repository Structure

```
.
├── test_cases/                        # Evaluation benchmarks
│   ├── complex_scenario_questions.xlsx   # Multi-round complex queries
│   ├── simple_scenario_questions.xlsx    # Single-round simple queries
├── src/
│   ├── config/                     # Configuration loader and template
│   ├── data_generation/            # Scripts for generating evaluation questions
│   ├── data_models/                # Pydantic data models for graph schema
│   ├── db_managers/                # Neo4j graph client
│   ├── gen_ai_gateway/             # LLM and embedding API wrappers
│   ├── graph_comparison/           # PFLGW distance implementation (key contribution)
│   │   └── fpgw_dis.py             # Partial Fused Gromov-Wasserstein distance
│   ├── indexing_pipeline/          # Knowledge graph construction pipeline
│   ├── medical_agent/              # LangGraph agent, tools, and evaluation harness
│   ├── preprocessing/              # Disease/gene/anatomy data preprocessing
│   ├── retrieval/                  # Retriever combining embeddings + graph distance
│   └── utils/
├── requirements.txt
└── README.md
```

---

## Requirements

**System requirements:**
- Python 3.10+
- Neo4j 5.x (Community Edition is sufficient)
- 8 GB+ RAM (16 GB recommended for embedding indexing)

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Neo4j

Download Neo4j from [neo4j.com/download](https://neo4j.com/download/) and start a local instance:

```bash
neo4j start
```

Default bolt URI: `neo4j://localhost:7687`. Set a password when prompted.

### 2. Configuration

Copy the template and fill in your credentials:

```bash
cp src/config/config.json.example src/config/config.json
```

Edit `src/config/config.json`:

```json
{
  "API_KEY": "your-openai-api-key",
  "API_BASE": "https://api.openai.com/v1",
  "MODEL_NAME": "gpt-4o",
  "NO_THINK_MODEL_NAME": "gpt-4o",
  "TEMPERATURE": 0,
  "MAX_OUTPUT_TOKEN": 4096,
  "EMBEDDING_API_BASE": "https://api.openai.com/v1",
  "EMBEDDING_MODEL": "text-embedding-3-large",
  "NEO4J_URI": "neo4j://localhost:7687",
  "NEO4J_USER": "neo4j",
  "NEO4J_PASSWORD": "your-neo4j-password"
}
```

> `src/config/config.json` is excluded from version control via `.gitignore`.

---

## Building the Knowledge Graph

### Step 1 — Preprocess source data

```bash
python -m src.preprocessing.build_disease
python -m src.preprocessing.build_disease_rel
python -m src.preprocessing.build_gene_anatomy
```

### Step 2 — Extract entities and ingest into Neo4j

```bash
python -m src.indexing_pipeline.pipeline \
  --input_dir input/ \
  --schema_path src/indexing_pipeline/schema.json \
  --output_dir output/ \
  --embed_nodes
```

See [src/indexing_pipeline/README.md](src/indexing_pipeline/README.md) for full pipeline documentation.

---

## Reproducing Evaluation Results

### Prerequisites

Before running evaluation, the Neo4j knowledge graph must be built and the retriever clusters must be initialized (done automatically on first run).

### Configure the evaluation script

Open `src/medical_agent/run_evaluation.py` and set the absolute path to the evaluation dataset:

```python
excel_path="dataset/complex_scenario_questions.xlsx",
```

> **Note:** The current file contains a hardcoded absolute path that must be updated to match your local installation before running.

### Run evaluation

```bash
python -m src.medical_agent.run_evaluation
```

This evaluates the agent on the complex scenario benchmark over three rounds and prints cumulative top-3 accuracy per round.

### Expected output format

```
=== SUMMARY ===
{
  "N": 202,
  "topk": 3,
  "exclusive": {
    "round1_accuracy": X.XX,
    "round2_accuracy": X.XX,
    "round3_accuracy": X.XX
  },
  "cumulative": {
    "solved_by_round2_accuracy": X.XX,
    "solved_by_round3_accuracy": X.XX
  }
}
```

---

## Results

Main results on the complex scenario benchmark (top-3 accuracy):

| Method             | Round 1 | By Round 2 | By Round 3 |
|--------------------|---------|------------|------------|
| Semantic only      |         |            |            |
| + Graph structure  |         |            |            |
| + PFLGW (ours)     |         |            |            |

> Results vary by LLM backend. See the paper for full ablation tables.

---

## Dataset

The evaluation datasets are in `dataset/`:

| File | Description |
|------|-------------|
| `complex_scenario_questions.xlsx` | 202 multi-round diagnostic questions with 3 progressive symptom queries per disease |
| `simple_scenario_questions.xlsx` | Single-round queries with direct symptom lists |

Each row in the question files contains:
- `disease_id` — target disease identifier
- `question1`, `question2`, `question3` — progressive natural-language symptom queries
- `symptoms_used_q1/2/3` — comma-separated canonical symptom terms used per round

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{[CITATION_KEY],
  title     = {[PAPER TITLE]},
  author    = {[AUTHORS]},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

This tool is intended for research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
