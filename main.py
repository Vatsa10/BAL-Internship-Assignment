import os
import sys

# --- FIX: WINDOWS COMPATIBILITY ---
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1" 

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- IMPORTS ---
import asyncio
import uuid  
import fitz  # PyMuPDF
import io
import shutil
import requests
import re
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import gradio as gr

# --- LIBRARIES ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker 

from qdrant_client import AsyncQdrantClient
# FIX: Added SparseIndexParams to imports
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, SparseIndexParams, FusionQuery, Fusion, Prefetch
from fastembed import TextEmbedding, SparseTextEmbedding # Hybrid Search
import google.generativeai as genai

# FlashRank for Reranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️ Warning: 'flashrank' not found. Reranking disabled.")

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# FIX: New Collection Name to enforce Multi-Vector Schema
COLLECTION_NAME = "docling_financial_hybrid_v3"

if not GEMINI_API_KEY: raise ValueError("❌ GEMINI_API_KEY missing")

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. DOCLING SETUP ---
def get_docling_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

# --- 2. ADVANCED PIPELINE CLASS ---
class AsyncDoclingRAG:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Client with increased timeout
        self.client = AsyncQdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY, 
            timeout=60
        )
        
        # HYBRID SEARCH MODELS
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") 
        
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.sem = asyncio.Semaphore(10) 

        # RERANKER
        if FLASHRANK_AVAILABLE:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        else:
            self.reranker = None

        print("🔄 Initializing Docling...")
        # get_docling_converter() # Lazy load
        print("✅ Advanced RAG System Initialized.")

    async def check_database(self) -> str:
        try:
            if await self.client.collection_exists(COLLECTION_NAME):
                cnt = (await self.client.count(COLLECTION_NAME)).count
                if cnt > 0:
                    return f"### ✅ Ready\n**Collection:** `{COLLECTION_NAME}`\n**Vectors:** {cnt}\n*Hybrid Search Enabled*"
                else:
                    return f"### ⚠️ Empty\nCollection `{COLLECTION_NAME}` exists but is empty."
            return f"### ❌ Not Found\nCollection `{COLLECTION_NAME}` missing. Please ingest."
        except Exception as e:
            return f"### ❌ Error\n{str(e)}"

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            print(f"⚙️ Creating Multi-Vector Collection: {COLLECTION_NAME}")
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    # FIX: Correct syntax for Sparse Vectors
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                }
            )

    # --- IMAGE ANALYSIS ---
    async def analyze_image(self, image_bytes: bytes, page_num: int) -> str:
        async with self.sem:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                prompt = "Analyze this chart. Extract key trends, axes, and values. Be concise."
                response = await self.gemini_model.generate_content_async([prompt, img])
                return f"Chart on Page {page_num}: {response.text}"
            except: return ""

    # --- PARSING HELPERS ---
    def _run_docling(self, pdf_path: str):
        converter = get_docling_converter()
        result = converter.convert(pdf_path)
        chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5") 
        chunks = []
        for chunk in chunker.chunk(result.document):
            chunks.append({"text": chunk.text, "metadata": chunk.meta.export_json_dict()})
        return chunks

    def _extract_images_fast(self, pdf_path: str):
        doc = fitz.open(pdf_path)
        images = []
        for i, page in enumerate(doc):
            for img in page.get_images(full=True):
                try:
                    base = doc.extract_image(img[0])
                    if len(base["image"]) > 5000:
                        images.append({"bytes": base["image"], "page": i + 1})
                except: pass
        return images

    # --- HYBRID INGESTION ---
    async def ingest(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found"
        await self.initialize_db()
        loop = asyncio.get_running_loop()
        
        progress(0.1, desc="Parsing (Docling + Images)...")
        docling_fut = loop.run_in_executor(self.executor, self._run_docling, pdf_path)
        images_fut = loop.run_in_executor(self.executor, self._extract_images_fast, pdf_path)
        text_chunks, raw_images = await asyncio.gather(docling_fut, images_fut)
        
        progress(0.4, desc=f"Analyzing {len(raw_images)} Charts...")
        img_tasks = [self.analyze_image(img["bytes"], img["page"]) for img in raw_images]
        img_descs = await asyncio.gather(*img_tasks)
        
        final_points = []
        # Add Text
        for c in text_chunks:
            final_points.append({"text": c["text"], "type": "text", "source": "docling", "metadata": c["metadata"]})
        # Add Charts
        for desc in img_descs:
            if desc: final_points.append({"text": desc, "type": "chart", "source": "vision", "metadata": {}})

        if not final_points: return "⚠️ No content."

        progress(0.7, desc="Generating Hybrid Embeddings...")
        texts = [p["text"] for p in final_points]
        
        # Generate embeddings
        dense_vecs = list(self.dense_model.embed(texts))
        sparse_vecs = list(self.sparse_model.embed(texts))
        
        points = []
        for i in range(len(final_points)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": sparse_vecs[i].as_object()
                },
                payload=final_points[i]
            ))
        
        batch_size = 50
        for i in range(0, len(points), batch_size):
            await self.client.upsert(COLLECTION_NAME, points[i:i+batch_size])
            
        return f"✅ Ingested {len(points)} hybrid vectors!"

    # --- HYBRID SEARCH + RRF + RERANKING ---
    async def query(self, text: str):
        # 1. Generate Queries
        dense_vec = list(self.dense_model.embed([text]))[0].tolist()
        sparse_vec = list(self.sparse_model.embed([text]))[0].as_object()
        
        # 2. Hybrid Retrieval (Qdrant Prefetch)
        response = await self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=20),
                Prefetch(query=sparse_vec, using="sparse", limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=20 
        )
        
        hits = response.points
        if not hits: return "No data found.", []

        # 3. Cross-Encoder Reranking
        if FLASHRANK_AVAILABLE and self.reranker:
            passages = [{"id": h.id, "text": h.payload["text"], "meta": h.payload} for h in hits]
            ranked = self.reranker.rerank(RerankRequest(query=text, passages=passages))
            top_hits = ranked[:5] 
        else:
            top_hits = hits[:5]

        # 4. Formatting
        ctx_parts = []
        citations = []
        for h in top_hits:
            payload = h["meta"] if isinstance(h, dict) else h.payload
            text_content = h["text"] if isinstance(h, dict) else h.payload["text"]
            
            # Metadata extraction
            meta = payload.get("metadata", {})
            page = meta.get("page_no", "?") 
            if page == "?":
                # Regex fallback for images if page isn't in metadata
                m = re.search(r"Page (\d+)", text_content)
                if m: page = m.group(1)

            citations.append(f"Page {page}")
            ctx_parts.append(f"[Page {page} | {payload.get('type','?')}]\n{text_content}")

        ctx = "\n\n".join(ctx_parts)
        
        prompt = f"""
        Role: Financial Analyst. 
        Task: Answer using context. Cite pages (e.g. [Page 5]).
        Context: {ctx}
        Question: {text}
        """
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text, list(set(citations))

    # --- SUMMARIZATION ---
    async def generate_briefing(self):
        # 1. Broad Retrieval
        dense_vec = list(self.dense_model.embed(["Overview of financial performance and risks"]))[0].tolist()
        
        # FIX: Use query_points with "using='dense'" to avoid schema mismatch
        response = await self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="dense",
            limit=30
        )
        hits = response.points
        
        if not hits: return "⚠️ No data available to summarize."

        summary_ctx = "\n".join([h.payload["text"] for h in hits])
        
        prompt = f"""
        Role: Senior Economic Advisor.
        Task: Write an Executive Briefing based on these excerpts.
        Structure:
        1. Executive Summary
        2. Key Indicators (Table)
        3. Risks & Outlook
        
        Excerpts:
        {summary_ctx[:30000]}
        """
        
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text

# --- UI & SINGLETON ---

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AsyncDoclingRAG()
    return _rag_instance

async def update_status_on_load():
    rag = get_rag()
    return await rag.check_database()

async def run_ingest(file, url):
    rag = get_rag()
    if url:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open("download.pdf", "wb") as f: f.write(r.content)
                path = "download.pdf"
            else: return f"Error: Status {r.status_code}"
        except Exception as e: return f"Download Error: {e}"
    elif file:
        path = file.name
    else:
        return "Please provide file or URL."
    
    result = await rag.ingest(path)
    status = await rag.check_database()
    return result, status

async def chat_wrapper(message, history):
    rag = get_rag()
    response_text, _ = await rag.query(message)
    return response_text

async def briefing_wrapper():
    rag = get_rag()
    return await rag.generate_briefing()

# --- LAYOUT ---
with gr.Blocks(title="Advanced Docling RAG") as demo:
    gr.Markdown("# 🚀 Advanced Financial RAG (Hybrid + RRF + Summarization)")
    
    # STATUS BOX
    status_box = gr.Markdown("🔄 Checking Knowledge Base...")
    demo.load(update_status_on_load, outputs=[status_box])
    
    with gr.Tab("💬 Chat & Search"):
        with gr.Accordion("📂 Data Ingestion", open=False):
            with gr.Row():
                f_in = gr.File(label="PDF")
                u_in = gr.Textbox(label="URL")
            btn = gr.Button("Ingest")
            out = gr.Textbox(label="Result")
            
            btn.click(run_ingest, [f_in, u_in], [out, status_box])
        
        gr.ChatInterface(
            fn=chat_wrapper,
            type="messages"
        )

    with gr.Tab("📝 Executive Briefing"):
        gr.Markdown("Generate a comprehensive summary of the ingested document.")
        brief_btn = gr.Button("Generate Briefing", variant="primary")
        brief_out = gr.Markdown()
        brief_btn.click(briefing_wrapper, outputs=[brief_out])

if __name__ == "__main__":
    demo.launch()