# Wikipedia Vector Search Knowledge Base

A demonstration of [MongoDB 8.2 Community Edition's new search and vector search capabilities](https://www.mongodb.com/company/blog/product-release-announcements/supercharge-self-managed-apps-search-vector-search-capabilities) (public preview). This project showcases how to build a comprehensive vector search-powered knowledge base using Wikipedia articles with MongoDB's native search features. Supports vector similarity search, full-text search, and hybrid search methods - all running locally without MongoDB Atlas.

## 🎯 What This Demonstrates

This project showcases the **public preview** of search and vector search in **MongoDB 8.2 Community Edition** - powerful capabilities previously only available in MongoDB Atlas, now accessible for local development and self-managed deployments.

### Key Capabilities Demonstrated

- **Native Vector Search** (`$vectorSearch`)
  - Semantic search using 768-dimensional embeddings
  - No external vector database required
  - Full functional parity with MongoDB Atlas

- **Full-Text Search** (`$search`)
  - Keyword-based search with fuzzy matching
  - Autocomplete and relevance scoring
  - Text analysis and faceting

- **Hybrid Search**
  - Combines vector and text search using RRF or weighted scores
  - Best of both semantic understanding and keyword matching

### Complete Pipeline Features

- Wikipedia XML parsing with streaming support
- Intelligent text chunking (semantic, fixed, hybrid strategies)
- Local embedding generation with LM Studio
- MongoDB Community Edition 8.2 storage with native search indexes
- Resumable ingestion with checkpointing
- Interactive CLI for testing all search methods

## 📋 Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) with Python 3.13+
- Docker and Docker Compose
- **MongoDB Community Server 8.2+** (included in docker-compose.yml)
- **mongot binary** (MongoDB Search Server - included in docker-compose.yml)
- [LM Studio](https://lmstudio.ai/download) with `text-embedding-nomic-embed-text-v1.5@q8_0` model
- Wikipedia XML dump (optional, for data ingestion)

## 🚀 Quick Start

### 1. Setup MongoDB Community Edition 8.2 with Search

```bash
# Start MongoDB Community Server 8.2 and mongot (search server)
docker compose up -d

# Verify containers are running
docker compose ps
```

This starts:
- **MongoDB Community Server 8.2.0** (port 27017) - The core database
- **mongot 0.53.1** (ports 27028, 9946) - MongoDB's search server for Community Edition
- Automatic replica set initialization

**Note:** This uses MongoDB Community Edition, NOT MongoDB Atlas. The search and vector search capabilities are now available locally in the 8.2 public preview.

### 2. Setup LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. In LM Studio, download the model: `text-embedding-nomic-embed-text-v1.5@q8_0`
3. Start the local server (default: `http://localhost:1234`)
4. Load the embedding model in LM Studio

### 3. Install Python Dependencies

```bash
# Install dependencies with uv
uv sync
```

### 4. Download Wikipedia Data

Download the Wikipedia XML dump and place it in the project directory:

```bash
# Create data directory
mkdir -p data

# Download Wikipedia dump (warning: large file ~20GB)
curl -o data/enwiki-latest-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2

# Or use wget
wget -P data/ \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
```

**Note:** The file is compressed (~20GB) and doesn't need to be extracted - the parser handles `.bz2` files directly.

### 5. Configuration

Edit `config/config.yaml` to point to your Wikipedia dump:

```yaml
wikipedia:
  xml_path: "./data/enwiki-latest-pages-articles-multistream.xml.bz2"

mongodb:
  uri: "mongodb://localhost:27017/?directConnection=true"
  database: "wikipedia_kb"

embedding:
  model: "text-embedding-nomic-embed-text-v1.5@q8_0"
  lmstudio_url: "http://localhost:1234"
  dimension: 768
```

### 6. Ingest Wikipedia Data

```bash
# Test with 100 articles first (recommended)
./scripts/ingest.py --max-articles 100

# Process more articles
./scripts/ingest.py --max-articles 1000

# Resume from checkpoint
./scripts/ingest.py --resume

# Clean database before ingesting
./scripts/ingest.py --clean --max-articles 500

# Specify custom XML file location
./scripts/ingest.py \
  --xml-file ./data/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --max-articles 100
```

### 7. Run Interactive Search Demo

```bash
./scripts/search.py
```

## Scripts Usage

### `scripts/search.py`

Ingests Wikipedia articles into MongoDB with embeddings.

**Usage:**
```bash
./scripts/ingest.py [OPTIONS]
```

**Options:**
- `--xml-file PATH` - Path to Wikipedia XML dump (default: `./data/enwiki-latest-pages-articles-multistream.xml.bz2`)
- `--max-articles N` - Maximum number of articles to process
- `--resume` - Resume from last checkpoint
- `--clean` - Clean database before ingesting (WARNING: deletes all data)
- `--checkpoint-interval N` - Save checkpoint every N articles (default: 50)
- `--no-cache` - Disable embedding cache
- `--verbose`, `-v` - Enable verbose logging

**Examples:**
```bash
# Start fresh with 100 articles
./scripts/ingest.py --clean --max-articles 100

# Resume previous ingestion
./scripts/ingest.py --resume

# Custom checkpoint interval
./scripts/ingest.py --max-articles 500 --checkpoint-interval 100

# Specify custom XML file location
./scripts/ingest.py \
  --xml-file ./data/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --max-articles 100
```

### `scripts/search.py`

Interactive CLI for searching the Wikipedia knowledge base.

**Usage:**
```bash
./scripts/search.py [OPTIONS]
```

**Options:**
- `--verbose`, `-v` - Enable verbose logging

**Available Commands:**
```
/vector <query>     - Semantic search using embeddings
/text <query>       - Keyword search using MongoDB text search
/hybrid <query>     - Combined search (RRF algorithm)
/weighted <query>   - Combined search (weighted scores)
/compare <query>    - Compare all search methods
/stats              - Show database statistics
/help               - Show help message
/quit or /exit      - Exit the program
```

**Example Queries:**
```bash
# Semantic search
/vector How does photosynthesis convert light energy into chemical energy?

# Keyword search
/text Albert Einstein special relativity theory

# Hybrid search
/hybrid machine learning neural networks deep learning

# Compare all methods
/compare quantum computing applications
```

## 📊 Architecture

This project demonstrates a complete local AI-powered search stack without cloud dependencies:

```
Wikipedia XML Dump (.xml.bz2)
    ↓
Parser (extract articles)
    ↓
Text Processor (clean, chunk)
    ↓
Embedding Generator (LM Studio - Local)
    ↓
MongoDB Community 8.2 + mongot (Local Search Server)
├── Vector Search ($vectorSearch)
├── Full-Text Search ($search)
└── Hybrid Search (Combined)
    ↓
Interactive Demo CLI
```

**All running locally** - no MongoDB Atlas or cloud services required!

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov=scripts --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_search_service.py -v

# Linting
uv run ruff check src/ tests/ scripts/
```

## 🔧 Components

### 1. WikiXMLParser (`wiki_parser.py`)
- Streams Wikipedia XML dumps
- Filters redirects and disambiguation pages
- Cleans wiki markup
- Handles compressed .bz2 files

### 2. TextProcessor (`text_processor.py`)
- Multiple chunking strategies:
  - **Semantic**: Preserves paragraphs/sections
  - **Fixed**: Fixed token count with overlap
  - **Hybrid**: Combines both approaches
- Token counting with tiktoken
- Maintains context (title, section headers)

### 3. EmbeddingGenerator (`embedding_service.py`)
- LM Studio API integration via `lmstudio` Python package
- Uses `text-embedding-nomic-embed-text-v1.5@q8_0` model (768 dimensions)
- Batch processing with progress tracking
- File-based embedding cache for efficiency
- Automatic retry with exponential backoff
- Dimension validation

### 4. MongoDBManager (`mongodb_manager.py`)
- Connection management
- Collection and index setup
- Bulk insert operations
- Vector and text search indexes
- Statistics and monitoring

### 5. IngestionPipeline (`ingest_pipeline.py`)
- Orchestrates: parse → chunk → embed → store
- Batch processing for efficiency
- Checkpointing for resumability
- Progress tracking and statistics
- Error handling and recovery

### 6. SearchService (`search_service.py`)
- Vector search using MongoDB's native `$vectorSearch` aggregation stage
- Text search using MongoDB's native `$search` with fuzzy matching
- Hybrid search (RRF and weighted algorithms)
- Score normalization
- Result deduplication

## 🎓 Search Algorithms

### Vector Search
Uses MongoDB Community Edition 8.2's native `$vectorSearch` aggregation stage to find semantically similar content based on embedding cosine similarity. This is the same capability previously only available in MongoDB Atlas.

### Text Search
Uses MongoDB Community Edition 8.2's native `$search` aggregation stage with text indexes and optional fuzzy matching for keyword-based retrieval.

### Hybrid Search

**RRF (Reciprocal Rank Fusion)**:
```
score = 1 / (k + rank)
```
Combines ranked lists fairly without needing score normalization.

**Weighted Score**:
```
final_score = α × normalized_vector_score + β × normalized_text_score
```
Where α + β = 1.0 (default: 0.6 vector + 0.4 text)

## 🔬 Development

### Running Tests
```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test class
uv run pytest tests/test_search_service.py::TestSearchService -v

# Run with debugging
uv run pytest tests/test_search_service.py -v -s
```

## 📝 Configuration

Configuration is managed via `config/config.yaml`:

