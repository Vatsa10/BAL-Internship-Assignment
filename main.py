import os
# Disable SSL Verify for Model Downloads (Fixes your SSL error)
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import asyncio
import fitz  # PyMuPDF for blazing fast image extraction
import io
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import gradio as gr
import uuid

# --- LIBRARIES ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker  # Docling's native smart chunker

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "docling_financial_v1"

if not GEMINI_API_KEY: raise ValueError("❌ GEMINI_API_KEY missing")

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. DOCLING CONVERTER SETUP (The Engine) ---
def get_docling_converter():
    # Configure for Speed & Table Accuracy
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for scanned pages
    pipeline_options.do_table_structure = True # Critical for Finance
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST # Fast mode
    
    # Thread-safe converter initialization
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

# --- 2. ASYNC PIPELINE CLASS ---
class AsyncDoclingRAG:
    def __init__(self):
        # Use ThreadPool for Docling (CPU bound but releases GIL often)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.sem = asyncio.Semaphore(10) # Rate limit for VLM

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    # --- A. IMAGE ANALYSIS (VLM) ---
    async def analyze_image(self, image_bytes: bytes, page_num: int) -> str:
        async with self.sem:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                prompt = "Analyze this chart/figure from a financial report. Identify trends, axes, and key values. Be concise."
                response = await self.gemini_model.generate_content_async([prompt, img])
                return f"Chart on Page {page_num}: {response.text}"
            except Exception:
                return ""

    # --- B. HYBRID INGESTION ---
    def _run_docling(self, pdf_path: str):
        """Runs Docling to get high-quality Text & Tables."""
        print("   📄 Docling: Parsing Structure & Tables...")
        converter = get_docling_converter()
        result = converter.convert(pdf_path)
        
        # Use Docling's smart chunker (handles layout awareness automatically)
        chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5") 
        chunk_iter = chunker.chunk(result.document)
        
        chunks = []
        for chunk in chunk_iter:
            # Docling chunks contain metadata about headings and captions
            chunks.append({
                "text": chunk.text,
                "metadata": chunk.meta.export_json_dict()
            })
        return chunks

    def _extract_images_fast(self, pdf_path: str):
        """Uses PyMuPDF for instant image byte extraction (Faster than Docling for this specific task)."""
        print("   🖼️ PyMuPDF: extracting images...")
        doc = fitz.open(pdf_path)
        images = []
        for i, page in enumerate(doc):
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    if len(base["image"]) > 5000: # Filter tiny icons
                        images.append({"bytes": base["image"], "page": i + 1})
                except: pass
        return images

    async def ingest(self, pdf_path: str, progress=gr.Progress()):
        await self.initialize_db()
        
        loop = asyncio.get_running_loop()
        progress(0.1, desc="Starting Hybrid Ingestion...")

        # 1. Run Docling (Text/Tables) & PyMuPDF (Images) in Parallel
        # This cuts latency significantly
        docling_future = loop.run_in_executor(self.executor, self._run_docling, pdf_path)
        images_future = loop.run_in_executor(self.executor, self._extract_images_fast, pdf_path)
        
        text_chunks, raw_images = await asyncio.gather(docling_future, images_future)
        
        progress(0.4, desc=f"Analyzing {len(raw_images)} Charts with Gemini...")
        
        # 2. Analyze Images with VLM
        image_tasks = [self.analyze_image(img["bytes"], img["page"]) for img in raw_images]
        image_descriptions = await asyncio.gather(*image_tasks)
        
        # 3. Merge All Data
        final_points = []
        
        # Add Docling Text/Tables
        for c in text_chunks:
            final_points.append({
                "text": c["text"],
                "type": "docling_text",
                "source": "text"
            })
            
        # Add VLM Chart Descriptions
        for desc in image_descriptions:
            if desc:
                final_points.append({
                    "text": desc,
                    "type": "chart_analysis",
                    "source": "vision"
                })

        progress(0.7, desc=f"Embedding {len(final_points)} vectors...")
        
        # 4. Embed & Index
        texts = [p["text"] for p in final_points]
        embeddings = list(self.embed_model.embed(texts))
        
        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload=final_points[i]
            )
            for i in range(len(final_points))
        ]
        
        # Batch Upload
        batch_size = 50
        for i in range(0, len(qdrant_points), batch_size):
            await self.client.upsert(COLLECTION_NAME, qdrant_points[i:i+batch_size])
            
        return f"✅ Ingested {len(qdrant_points)} chunks (Docling + Gemini Vision)"

    async def query(self, text: str):
        vec = list(self.embed_model.embed([text]))[0].tolist()
        hits = await self.client.query_points(COLLECTION_NAME, query=vec, limit=10).points
        
        ctx = "\n\n".join([f"[Type: {h.payload['type']}]\n{h.payload['text']}" for h in hits])
        
        prompt = f"Role: Financial Analyst. Answer using context.\nContext:\n{ctx}\nQuestion: {text}"
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text, []

# --- UI ---
rag = AsyncDoclingRAG()

with gr.Blocks() as demo:
    gr.Markdown("## 🚀 Fast Docling RAG (Hybrid Pipeline)")
    with gr.Row():
        f = gr.File(label="PDF")
        btn = gr.Button("Ingest")
    out = gr.Textbox(label="Status")
    
    chat = gr.ChatInterface(
        fn=lambda m, h: asyncio.run(rag.query(m)) if asyncio.get_event_loop().is_running() else rag.query(m)[0],
        type="messages"
    )
    
    async def run_ingest(file):
        return await rag.ingest(file.name)
        
    btn.click(run_ingest, f, out)

if __name__ == "__main__":
    demo.launch()