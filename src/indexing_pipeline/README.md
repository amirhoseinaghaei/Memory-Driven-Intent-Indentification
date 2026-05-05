# 📊 Indexing Pipeline

The indexing pipeline is the core component that **extracts medical entities** (diseases, phenotypes, anatomy, drugs) from medical documents using AI and **ingests them into a Neo4j graph database**. This pipeline transforms unstructured medical text into a structured knowledge graph for efficient querying and analysis.

## 🎯 Purpose

The indexing pipeline:
1. **Extracts entities** from medical documents using LLM-powered extraction
2. **Parses and normalizes** extracted data
3. **Validates relationships** between entities
4. **Ingests into Neo4j** for graph-based storage and querying
5. **Generates embeddings** for semantic search (optional)

## 📋 Pipeline Architecture

```
Medical Documents (TXT files)
    ↓
Entity Extraction (LLM)
    ↓
Parsing & Normalization
    ↓
Relationship Validation
    ↓
Graph Ingestion (Neo4j)
    ↓
Embedding Generation (Optional)
    ↓
Knowledge Graph Ready for Querying
```

## 📁 Directory Structure

```
src/indexing_pipeline/
├── pipeline.py              # Main pipeline orchestrator
├── entity_extractor.py      # LLM-based entity extraction (optimized for async)
├── parser.py                # Document parsing and normalization
├── graph_ingestor.py        # Neo4j ingestion logic
├── schema.json              # Entity schema definition
└── README.md                # This file
```

## 🔧 Configuration

### Schema Definition (`schema.json`)

The schema defines the layers (entity types) and relationships allowed in your knowledge graph:

```json
{
  "layers": [
    {"index": 1, "name": "phenotype"},
    {"index": 2, "name": "anatomy"},
    {"index": 3, "name": "disease"},
    {"index": 4, "name": "drug"}
  ],
  "relations": [
    {"from": "phenotype", "to": "anatomy", "type": "LOCATED_IN"},
    {"from": "anatomy", "to": "disease", "type": "AFFECTS"},
    {"from": "disease", "to": "drug", "type": "CURED_BY"}
  ],
  "root_layer": "disease"
}
```

To customize, edit `src/indexing_pipeline/schema.json` and modify:
- **layers**: Entity types to extract
- **relations**: Allowed relationships between layers
- **root_layer**: The primary entity type

### API Configuration (`src/config/config.json`)

Update your configuration file with API credentials and Neo4j connection details:

```json
{
  "API_KEY": "your-openai-api-key",
  "MODEL_NAME": "gpt-5.4-nano",
  "API_BASE": "https://api.openai.com/v1",
  "EMBEDDING_MODEL": "text-embedding-3-large",
  "NEO4J_URI": "neo4j://localhost:7687",
  "NEO4J_USER": "neo4j",
  "NEO4J_PASSWORD": "your-neo4j-password"
}
```

## 🚀 Quick Start

### 1. Prepare Input Documents

Create a folder with `.txt` files containing medical information:

```bash
input/
├── 001_disease1.txt
├── 002_disease2.txt
└── 003_disease3.txt
```

Each file should contain structured or semi-structured medical information describing:
- Disease name and ID
- Associated phenotypes
- Affected anatomy
- Related drugs
- Symptoms and clinical findings

### 2. Run the Pipeline

From the project root directory:

```powershell
python -m src.indexing_pipeline.pipeline `
  --input_dir "input" `
  --schema_path "src/indexing_pipeline/schema.json" `
  --output_dir "output" `
  --config_path "src/config/config.json" `
  --chunk_size 5000 `
  --embed_nodes
```

### 3. Monitor Progress

The pipeline generates real-time logs in both console and file:

```
🚀 ================================================================================
🚀  INDEXING PIPELINE STARTED - 2026-04-15 14:30:45
🚀 ================================================================================

================================================================================
STEP 1: ENTITY EXTRACTION
================================================================================
🔍 Loading schema from: src/indexing_pipeline/schema.json
✓ Schema loaded successfully
📁 Input directory: input
📄 Found 3 files to process
⚙️  Processing files...
Processing files: 100%|████████████| 3/3 [01:30<00:00, 30s/file]
✓ Extraction completed successfully

================================================================================
STEP 2: GRAPH INGESTION
================================================================================
🗄️  Initializing Neo4j ingestor
📤 Ingesting payload into Neo4j...
✓ Ingestion completed successfully

✅ PIPELINE COMPLETED SUCCESSFULLY - 2026-04-15 14:31:45
```

Detailed logs are saved to `indexing_pipeline.log`.

## 📊 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | Path | Required | Directory with `.txt` files to extract |
| `--schema_path` | Path | Required | Path to schema.json |
| `--output_dir` | Path | Required | Output directory for JSON payload |
| `--config_path` | Path | `src/config/config.json` | Configuration file path |
| `--api_key` | String | From config | OpenAI API key (overrides config) |
| `--model` | String | From config | LLM model name (overrides config) |
| `--api_base` | String | From config | API base URL (overrides config) |
| `--neo4j_database` | String | `neo4j` | Neo4j database name |
| `--chunk_size` | Integer | `5000` | Data chunk size for ingestion |
| `--embed_nodes` | Flag | False | Enable node embeddings |

## 📝 Input File Format

Input text files should contain medical information in a clear format. Example:

```
Disease: Nephropathic Cystinosis
Disease ID: 100151_9066_18467

Phenotypes (observed features):
- Renal tubular dysfunction
- Corneal crystals
- Cognitive impairment
- Hypothyroidism

Anatomy involved:
- Kidney cortex
- Cornea
- Brain
- Thyroid

Treatments:
- Cysteamine
- Supportive care
```

The LLM will intelligently extract and structure this information according to your schema.

## 📤 Output Format

The pipeline generates `combined_graph_payload.json` containing:

```json
{
  "schema": {
    "layers": [...],
    "relations": [...],
    "root_layer": "disease"
  },
  "records": [
    {
      "disease": {
        "id": "disease_id",
        "name": "Disease Name"
      },
      "phenotypes": [
        {"id": "phenotype_1", "name": "Symptom", "anatomies": [...]}
      ],
      "drugs": [
        {"id": "drug_1", "name": "Treatment"}
      ]
    }
  ],
  "failed_files": [
    {"file": "filename.txt", "error": "error message"}
  ]
}
```

## ⚡ Performance Optimization

### Parallel Processing

The entity extractor supports concurrent processing. Adjust `max_parallel` parameter:

```python
from src.indexing_pipeline.entity_extractor import SchemaDrivenExtractor

extractor = SchemaDrivenExtractor(
    api_key="your-key",
    max_parallel=5  # Process 5 files concurrently
)
results = extractor.extract_files_parallel_sync(schema, file_list)
```

**Expected speedup**: 5x faster with `max_parallel=5` (1 min for 10 files vs 5 min sequentially)

### Caching

System prompts are automatically cached after first use, reducing prompt generation overhead.

### Model Selection

Use faster models for quicker processing:

```bash
--model "gpt-5.4-nano"  # Faster, ~30-50s per file
--model "gpt-5.2"       # Standard
--model "gpt-4-turbo"   # More accurate but slower
```

## 🐛 Troubleshooting

### Issue: "No .txt files found"
**Solution**: Ensure input directory path is correct and contains `.txt` files.

### Issue: "Failed to connect to Neo4j"
**Solution**: 
- Verify Neo4j is running: `neo4j console`
- Check credentials in `src/config/config.json`
- Confirm URI format: `neo4j://localhost:7687`

### Issue: "API rate limit exceeded"
**Solution**: Reduce `max_parallel` parameter or increase `timeout`.

### Issue: "Empty anatomies in output"
**Solution**: 
- Verify schema defines `anatomy` layer
- Ensure input text mentions anatomical structures
- Check LLM extraction quality with debug logging

### Issue: "Anatomy extraction not working"
**Diagnostic**:
1. Check `indexing_pipeline.log` for detailed errors
2. Verify schema relations include `phenotype → anatomy` mapping
3. Test with a single file first:
   ```bash
   python -m src.indexing_pipeline.pipeline \
     --input_dir "test_single_file_dir" \
     --schema_path "src/indexing_pipeline/schema.json" \
     --output_dir "output_test"
   ```

## 🔍 Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs at:
- **Console**: Real-time output with progress bars
- **File**: `indexing_pipeline.log` (persistent record)

## 📊 Monitoring

After pipeline completion, validate the knowledge graph:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    # Count entities by type
    result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
    for record in result:
        print(f"{record['label']}: {record['count']} nodes")
    
    # Count relationships
    result = session.run("MATCH (a)-[r]-(b) RETURN type(r) as rel_type, count(*) as count")
    for record in result:
        print(f"{record['rel_type']}: {record['count']} relations")

driver.close()
```

## 📚 Examples

### Example 1: Process single disease document

```bash
python -m src.indexing_pipeline.pipeline \
  --input_dir "input" \
  --schema_path "src/indexing_pipeline/schema.json" \
  --output_dir "output"
```

### Example 2: Process with custom model and higher parallelism

```python
from pathlib import Path
from src.indexing_pipeline.pipeline import run_extraction, run_graph_ingestion
from src.config.config import load_settings_from_json, Settings
from src.gen_ai_gateway.embedder import Embed

schema_path = Path("src/indexing_pipeline/schema.json")
input_dir = Path("input")
output_dir = Path("output")

# Extract entities
output_file = run_extraction(
    input_dir=input_dir,
    schema_path=schema_path,
    output_dir=output_dir,
    api_key="your-key",
    model="gpt-5.2",
    api_base="https://api.openai.com/v1"
)

# Ingest into Neo4j
settings = load_settings_from_json(Path("src/config/config.json"))
embedder = Embed(settings)

run_graph_ingestion(
    payload_path=output_file,
    uri=settings.NEO4J_URI,
    user=settings.NEO4J_USER,
    password=settings.NEO4J_PASSWORD,
    database="neo4j",
    embedder=embedder,
    chunk_size=5000,
    embed_nodes=True
)
```

### Example 3: Custom schema for different use cases

Create `custom_schema.json` for your domain:

```json
{
  "layers": [
    {"index": 1, "name": "symptom"},
    {"index": 2, "name": "organ"},
    {"index": 3, "name": "condition"},
    {"index": 4, "name": "treatment"},
    {"index": 5, "name": "risk_factor"}
  ],
  "relations": [
    {"from": "symptom", "to": "organ", "type": "PRESENTS_IN"},
    {"from": "organ", "to": "condition", "type": "CAUSES"},
    {"from": "condition", "to": "treatment", "type": "TREATED_BY"},
    {"from": "risk_factor", "to": "condition", "type": "INCREASES_RISK"}
  ],
  "root_layer": "condition"
}
```

Then run:

```bash
python -m src.indexing_pipeline.pipeline \
  --input_dir "input" \
  --schema_path "custom_schema.json" \
  --output_dir "output"
```

## 📖 Related Documentation

- [Entity Extractor Optimization Guide](../../../archive/OPTIMIZATION_GUIDE.md)
- [Graph Database Schema](./schema.json)
- [Config Reference](../config/config.json)

## ❓ FAQ

**Q: How long does extraction take?**
A: ~30-60 seconds per document with LLM baseline. Use parallelization for faster processing.

**Q: Can I extract from unstructured text?**
A: Yes! The LLM is designed to extract from both structured and semi-structured text.

**Q: What's the maximum file size?**
A: LLM context window dependent (typically 4k-128k tokens). Split very large documents.

**Q: Can I reuse extractions?**
A: Yes! The JSON output can be processed independently for graph ingestion.

**Q: How do I scale to thousands of documents?**
A: Use multiprocessing or parallel extraction with `max_parallel=10+` (with appropriate API rate limit management).

---

**Last Updated**: April 2026
