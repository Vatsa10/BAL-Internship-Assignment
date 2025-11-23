# Multi-Modal Financial RAG Pipelines

A high-performance, local-first Retrieval Augmented Generation (RAG) system designed for complex Financial and Policy documents (e.g., IMF Article IV Reports).

This repository contains two distinct pipeline implementations optimized for different needs:

- **app.py (Standard Pipeline)**: Optimized for speed, custom chunking, and re-ranking. Best for text-heavy documents.
- **main.py (Docling Pipeline)**: Optimized for complex layout parsing, cross-modal retrieval, and high-fidelity table extraction. Best for documents with complex layouts and tables.
Demo video -  https://www.loom.com/share/7c81872491694ec7a4d8d640bd070c79
## Pipeline Comparison

| Feature | Standard Pipeline (`app.py`) | Docling Pipeline (`main.py`) |
| :--- | :--- | :--- |
| **Parsing Engine** | PyMuPDF + PaddleOCR | IBM Docling (SOTA Layout Analysis) |
| **Chunking Strategy** | Context-Aware Sentence Windowing | Hybrid Semantic Chunking |
| **Table Handling** | Text-based heuristics + Markdown conversion | Structure-aware TableFormer models |
| **Re-Ranking** | FlashRank (Text-only) | FlashRank (Cross-modal) |
| **Retrieval** | Dense vectors (BAAI/bge-small-en-v1.5) | Hybrid: Dense + Sparse (RRF Fusion) |
| **Vision Integration** | Chart analysis via Gemini VLM | Chart analysis + Caption extraction |
| **Speed** | Very Fast | Moderate (Heavy ML models for layout) |
| **Vector Collection** | `imf_policy_reports_final_v4_3` | `docling_financial_context_v4` |
| **Best For** | Text-heavy policy docs, huge archives | Docs with complex columns, headers, tables |
| **Evaluation** | Basic metrics | Comprehensive multi-modality evaluation |

## Tech Stack

- **Frontend**: Gradio (Async Web UI)
- **Vector Database**: Qdrant (Cloud or Local Docker)
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5 for dense, Qdrant/bm25 for sparse)
- **LLM / VLM**: Google Gemini 2.0 Flash (Free Tier) - Handles Text & Image reasoning
- **Re-Ranking**: FlashRank (ms-marco-MiniLM-L-12-v2) - Cross-modal relevance scoring
- **Layout Analysis**: IBM Docling (main.py only) - SOTA document understanding
- **Infrastructure**: Fully Async (`asyncio` + `ThreadPoolExecutor`)

## Advanced Features

### Docling Pipeline (main.py)

**Cross-Modal Retrieval & Reranking:**
- Hybrid search combining dense embeddings and sparse BM25 vectors
- Reciprocal Rank Fusion (RRF) for optimal result fusion
- FlashRank cross-modal reranking for improved relevance
- Vision-text integration: Charts analyzed and indexed alongside text

**Retrieval Fine-Tuning:**
- Adaptive retrieval limits based on query type (text/table/chart)
- Context-aware chunking with semantic boundaries
- Sticky headers to preserve document structure
- Caption extraction for figures and tables

**Summarization & Briefing:**
- Executive briefing generation from top-ranked documents
- Structured output: Outlook, Risks, Policy Recommendations
- Query-specific response formatting (table/chart/text)

**Multi-Modality Support:**
- Text extraction with layout preservation
- Table structure recognition via TableFormer
- Chart/figure analysis with caption context
- Image-text fusion for comprehensive understanding

### Standard Pipeline (app.py)

**Fast Re-Ranking:**
- FlashRank for text-only relevance scoring
- Sentence window chunking with overlap
- Parallel OCR and vision processing

**Efficient Processing:**
- ProcessPoolExecutor for CPU-bound tasks
- Concurrent image analysis and text extraction
- Query result caching for repeated queries

## Prerequisites

- Python 3.10+
- Google Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))
- Qdrant Cluster (Free tier at [cloud.qdrant.io](https://cloud.qdrant.io) or Local Docker)
- **Windows Users**: Enable "Developer Mode" in Windows Settings (required for Docling model downloads) OR run terminal as Administrator.

## Installation

1. **Clone the repository**

2. **Install System Dependencies (Linux only):**
   ```bash
   sudo apt-get install libgl1-mesa-glx
   ```

3. **Install Python Dependencies:**
   ```bash
   pip install pymupdf qdrant-client google-generativeai fastembed paddleocr paddlepaddle opencv-python-headless gradio requests python-dotenv docling flashrank
   ```

## Configuration

Create a `.env` file in the root directory:

```env
# Generative AI Key
GEMINI_API_KEY=your_gemini_key_here

# Qdrant Vector Database
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### Option 1: Standard Pipeline (Fast & Robust)

Use this for quick analysis of long reports where text flow is standard.

```bash
python app.py
```

**Key Features:**
- Uses **FlashRank** to re-order search results for higher accuracy
- Uses **PaddleOCR** for scanned pages
- Uses a **Sliding Window** approach to keep context
- Optimized for speed and throughput

### Option 2: Docling Pipeline (Layout Master)

Use this for documents with complex layouts, multi-column text, or financial tables that standard parsers break.

```bash
python main.py
```

**Key Features:**
- Uses **IBM's Docling models** to understand the visual layout of the PDF
- Extracts tables structurally (not just as text)
- **Hybrid retrieval**: Dense + Sparse vectors with RRF fusion
- **Cross-modal reranking**: FlashRank scores text-image relevance
- **Adaptive prompts**: Tailored responses for tables, charts, and text queries

## Evaluation & Benchmarking

The Docling pipeline includes a comprehensive evaluation suite for benchmarking performance across multiple modalities.

### Running Evaluations

**1. Ingest a test document:**
```bash
python main.py
# Upload qatar_test_doc.pdf in the Ingest tab
```

**2. Run evaluation suite:**
```bash
python evaluate_rag.py
```

This runs 15 evaluation queries across 3 modalities:
- **Text queries (5)**: Economic challenges, outlook, risks, banking, reforms
- **Table queries (5)**: Macroeconomic indicators, production, finance, external sector, trends
- **Chart queries (5)**: Economic performance, fiscal position, external trends, inflation, LNG sector

**3. Diagnose text query performance:**
```bash
python diagnose_text_queries.py
```

Shows keyword matching rates, response quality, and recommendations for improvement.

### Evaluation Metrics

**Response Time:**
- Average, Median, Min, Max response times
- Per-modality breakdown

**Quality Scores (1-5 scale):**
- **Relevance**: Keyword presence and semantic alignment
- **Completeness**: Response depth and source count
- **Accuracy**: Response structure and factual correctness

**Retrieval Metrics:**
- Average sources retrieved
- Average response length
- Per-modality performance

### Output Files

- `benchmark_results.json` - Raw evaluation data with detailed metrics
- Modality-specific performance breakdown
- Recommendations for optimization

### Performance Targets

| Metric | Target | Acceptable |
| :--- | :--- | :--- |
| Response Time | < 2s | < 5s |
| Relevance Score | > 4.2/5 | > 3.5/5 |
| Completeness Score | > 4.0/5 | > 3.5/5 |
| Accuracy Score | > 4.1/5 | > 3.5/5 |
| Avg Sources | > 4 | > 3 |
