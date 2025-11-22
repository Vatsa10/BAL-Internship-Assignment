# Multi-Modal Financial RAG Pipelines

A high-performance, local-first Retrieval Augmented Generation (RAG) system designed for complex Financial and Policy documents (e.g., IMF Article IV Reports).

This repository contains two distinct pipeline implementations optimized for different needs:

- **app.py (Standard Pipeline)**: Optimized for speed, custom chunking, and re-ranking.
- **main.py (Docling Pipeline)**: Optimized for complex layout parsing and high-fidelity table extraction.

## Pipeline Comparison

| Feature | Standard Pipeline (`app.py`) | Docling Pipeline (`main.py`) |
| :--- | :--- | :--- |
| **Parsing Engine** | PyMuPDF + PaddleOCR | IBM Docling (SOTA Layout Analysis) |
| **Chunking Strategy** | Context-Aware Sentence Windowing | Hybrid Semantic Chunking |
| **Table Handling** | Text-based heuristics + Markdown conversion | Structure-aware TableFormer models |
| **Re-Ranking** | Yes (FlashRank) | No (Relies on Docling's high-quality chunks) |
| **Speed** | Very Fast | Slower (Heavy ML models for layout) |
| **Vector Collection** | `imf_policy_reports_final_v4_3` | `docling_financial` |
| **Best For** | Text-heavy policy docs, huge archives. | Docs with complex columns, headers, & tables. |

## Tech Stack

- **Frontend**: Gradio (Async Web UI)
- **Vector Database**: Qdrant (Cloud or Local Docker)
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5) - Runs locally on CPU.
- **LLM / VLM**: Google Gemini 2.0 Flash (Free Tier) - Handles Text & Image reasoning.
- **Infrastructure**: Fully Async (`asyncio` + `ProcessPoolExecutor`/`ThreadPoolExecutor`).

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
- Uses **FlashRank** to re-order search results for higher accuracy.
- Uses **PaddleOCR** for scanned pages.
- Uses a **Sliding Window** approach (Page 1 Policy Text...) to keep context.

### Option 2: Docling Pipeline (Layout Master)

Use this for documents with complex layouts, multi-column text, or financial tables that standard parsers break.

```bash
python main.py
```

**Key Features:**
- Uses **IBM's Docling models** to understand the visual layout of the PDF.
- Extracts tables structurally (not just as text).
- **Hybrid pipeline**: Uses Docling for structure + PyMuPDF for fast image extraction.

## Architecture Overview

```mermaid
graph TD
    User[User Upload] --> Router{Select Pipeline}
    
    subgraph "Standard Pipeline (app.py)"
        P1[PyMuPDF] --> Text[Sentence Window Chunking]
        P1 --> OCR[PaddleOCR (Scans)]
        P1 --> Img1[Gemini VLM (Charts)]
        Text & OCR & Img1 --> Embed1[FastEmbed]
        Embed1 --> Q1[(Qdrant Collection: imf_policy_reports_final_v4_3)]
        Search1[Vector Search] --> Rank[FlashRank Re-Ranking]
    end
    
    subgraph "Docling Pipeline (main.py)"
        D1[Docling Converter] --> Layout[Layout & Table Analysis]
        Layout --> Hybrid[Hybrid Chunking]
        D1 --> Img2[Gemini VLM (Charts)]
        Hybrid & Img2 --> Embed2[FastEmbed]
        Embed2 --> Q2[(Qdrant Collection: docling_financial)]
    end
    
    Router --> P1
    Router --> D1
    
    Rank --> LLM[Gemini Flash Analyst]
    Q2 --> LLM
    LLM --> Answer[Final Response]
```

## Troubleshooting

1. **`OSError: [WinError 1314] A required privilege is not held by the client`**
   - **Cause**: Docling needs to create symbolic links to cache AI models.
   - **Fix**: Run your terminal/VS Code as **Administrator** or enable **Developer Mode** in Windows Settings. The code attempts to mitigate this using environment variables (`HF_HUB_DISABLE_SYMLINKS`), but permissions may still be required on some systems.

2. **`SSL: CERTIFICATE_VERIFY_FAILED`**
   - **Cause**: Corporate firewalls or missing certificates when downloading models from Hugging Face.
   - **Fix**: Both scripts include automatic patches (`HF_HUB_DISABLE_SSL_VERIFY`) to bypass this. If it persists, check your VPN/Proxy settings.

3. **`No module named 'paddle'`**
   - **Cause**: You installed `paddleocr` but missed the engine.
   - **Fix**: Run `pip install paddlepaddle`.

4. **Ingestion hangs or is slow**
   - **Solution**:
     - `app.py` uses massive parallelism (OCR + Vision + Text all run concurrently).
     - `main.py` is heavier. Be patient on the first run as it downloads models (~500MB).
