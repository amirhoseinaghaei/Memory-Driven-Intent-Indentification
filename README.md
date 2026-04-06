# 🩺 Medical Graph Agent

A sophisticated AI-powered medical diagnosis assistant that leverages graph databases and advanced retrieval techniques to help identify potential diseases based on user-described symptoms. Built with Neo4j, LangGraph, and OpenAI embeddings for accurate and explainable medical querying.

## ✨ Features

- **Graph-Based Retrieval**: Utilizes Neo4j graph database to model complex relationships between diseases, symptoms, genes, and drugs
- **Intelligent Clustering**: Employs UMAP and HDBSCAN for phenotype clustering to improve retrieval accuracy
- **Interactive Agent**: LangGraph-powered conversational agent that can clarify symptoms and provide ranked disease candidates
- **Embedding-Powered Search**: Sentence Transformers for semantic similarity matching
- **Modular Architecture**: Clean separation of concerns with dedicated modules for indexing, retrieval, and agent logic

## 📋 Prerequisites

- Python 3.8+
- Neo4j 5.x (Community or Enterprise)
- OpenAI API access (or compatible API)
- Sufficient RAM for embeddings processing (8GB+ recommended)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd memory_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Neo4j**:
   - Download and install Neo4j from [neo4j.com](https://neo4j.com/download/)
   - Start Neo4j with default credentials (neo4j/neo4j) and change password
   - Note the bolt URI (default: `neo4j://localhost:7687`)

## ⚙️ Configuration

1. **Configure API credentials**:
   Edit `src/config/config.json` with your API endpoints and keys:
   ```json
   {
     "API_KEY": "your-openai-api-key",
     "API_BASE": "https://api.openai.com/v1",
     "MODEL_NAME": "gpt-4",
     "EMBEDDING_API_BASE": "https://api.openai.com/v1",
     "EMBEDDING_MODEL": "text-embedding-3-large",
     "NEO4J_URI": "neo4j://localhost:7687",
     "NEO4J_USER": "neo4j",
     "NEO4J_PASSWORD": "your-neo4j-password"
   }
   ```

2. **Verify data files**:
   Ensure the following data files are present in the project root:
   - `phenotype_disease.csv`
   - `gene_disease.csv`
   - `drug_disease.csv`
   - `dataset/phenotype_catalog.json`

## 🔄 Indexing the Data

The indexing process builds the knowledge graph and prepares embeddings for efficient retrieval.

### Step 1: Preprocess Disease Data
Run the preprocessing scripts to build disease mappings:
```bash
python -m src.preprocessing.build_disease
python -m src.preprocessing.build_disease_rel
python -m src.preprocessing.build_gene_anatomy
```

### Step 2: Ingest Data into Neo4j
Use the indexing pipeline to populate the graph database:
```bash
# Note: Adjust paths and parameters as needed
python -c "
from src.indexing_pipeline.pipeline import run_extraction
from pathlib import Path
run_extraction(
    input_dir=Path('data_directory'),
    schema_path=Path('src/indexing_pipeline/schema.json'),
    output_dir=Path('output'),
    api_key='your-api-key',
    model='gpt-4',
    api_base='https://api.openai.com/v1'
)
"
```

### Step 3: Build Phenotype Clusters
Initialize the retriever and build clusters for improved symptom matching:
```python
from src.config.config import settings
from src.retrieval.retriever import Retriever
from src.gen_ai_gateway.chat_completion import ChatCompletion

chat = ChatCompletion(settings)
retriever = Retriever(settings, chat)
retriever.build_clusters()
```

## 🏃 Running the Agent

### Interactive Mode
Run the medical agent in interactive mode for symptom-based disease querying:
```bash
python -m src.medical_agent.agent
```

The agent will:
1. Accept symptom descriptions
2. Retrieve and rank potential diseases
3. Ask clarifying questions if needed
4. Provide top disease candidates with confidence scores

### Example Query
```
My child has been peeing a lot and seems smaller than other kids the same age. Could polyuria from the kidney be related to short stature?
```

### Programmatic Usage
```python
from src.config.config import settings
from src.medical_agent.agent import build_graph_agent
from src.retrieval.retriever import Retriever
from src.gen_ai_gateway.chat_completion import ChatCompletion

# Initialize components
chat = ChatCompletion(settings)
retriever = Retriever(settings, chat)
retriever.build_clusters()
agent = build_graph_agent(retriever)

# Query the agent
result = agent.invoke({
    "user_query": "fever, cough, fatigue",
    "previous_groups": [],
    "previous_diseases": []
})
print(result["final"])
```

## 📊 Understanding Results

The agent returns:
- **Top-3 Candidates**: Most likely diseases with similarity scores
- **Confidence Threshold**: Diseases with score ≥ 0.99 are considered confident matches
- **Clarification Requests**: Additional symptoms needed for better disambiguation
- **Target Tracking**: If testing with known diseases, shows target disease rank

## 🛠️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   LangGraph      │───▶│   Neo4j Graph   │
│                 │    │   Agent          │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Retrieval      │
                       │   Engine         │
                       │   (Embeddings +  │
                       │    Clustering)   │
                       └──────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.