import os
import sys

# --- FIX 1: WINDOWS COMPATIBILITY CONFIGURATION (MUST BE FIRST) ---
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
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "docling_financial"

if not GEMINI_API_KEY: raise ValueError("❌ GEMINI_API_KEY missing")

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. DOCLING SETUP ---
def get_docling_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

# --- 2. ASYNC PIPELINE ---
class AsyncDoclingRAG:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Qdrant Client
        self.client = AsyncQdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY, 
            timeout=60
        )
        
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.sem = asyncio.Semaphore(10) 

        # Warmup Docling
        print("🔄 Initializing Docling (This may download models on first run)...")
        # We don't call it here to allow lazy loading, preventing startup crashes
        # get_docling_converter() 
        print("✅ RAG System Initialized.")

    async def check_database(self) -> str:
        """Checks if the collection exists and has data."""
        try:
            print(f"🔍 Checking Qdrant collection: {COLLECTION_NAME}...")
            if await self.client.collection_exists(COLLECTION_NAME):
                count_result = await self.client.count(COLLECTION_NAME)
                cnt = count_result.count
                print(f"✅ Collection found with {cnt} vectors.")
                
                if cnt > 0:
                    return f"### ✅ Knowledge Base Ready\n**Collection:** `{COLLECTION_NAME}`\n**Vectors Loaded:** {cnt}\n\n*System is ready for chat. No ingestion needed.*"
                else:
                    return f"### ⚠️ Knowledge Base Empty\nCollection `{COLLECTION_NAME}` exists but has 0 vectors.\nPlease ingest a document."
            else:
                return f"### ❌ Knowledge Base Not Found\nCollection `{COLLECTION_NAME}` does not exist.\nPlease upload a PDF to create it."
        except Exception as e:
            return f"### ❌ Connection Error\nCould not connect to Qdrant.\nError: {str(e)}"

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    async def analyze_image(self, image_bytes: bytes, page_num: int) -> str:
        async with self.sem:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                prompt = "Analyze this chart/figure. Identify trends, axes, and key values. Be concise."
                response = await self.gemini_model.generate_content_async([prompt, img])
                return f"Chart on Page {page_num}: {response.text}"
            except Exception:
                return ""

    def _run_docling(self, pdf_path: str):
        """CPU-bound Docling execution."""
        converter = get_docling_converter()
        result = converter.convert(pdf_path)
        
        chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5") 
        chunk_iter = chunker.chunk(result.document)
        
        chunks = []
        for chunk in chunk_iter:
            chunks.append({
                "text": chunk.text,
                "metadata": chunk.meta.export_json_dict()
            })
        return chunks

    def _extract_images_fast(self, pdf_path: str):
        """Fast Image Extraction via PyMuPDF."""
        doc = fitz.open(pdf_path)
        images = []
        for i, page in enumerate(doc):
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    if len(base["image"]) > 5000:
                        images.append({"bytes": base["image"], "page": i + 1})
                except: pass
        return images

    async def ingest(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found"
        
        await self.initialize_db()
        loop = asyncio.get_running_loop()
        
        progress(0.1, desc="Parsing Structure (Docling) & Images...")
        
        try:
            # Run heavyweight tasks in parallel threads
            docling_future = loop.run_in_executor(self.executor, self._run_docling, pdf_path)
            images_future = loop.run_in_executor(self.executor, self._extract_images_fast, pdf_path)
            
            text_chunks, raw_images = await asyncio.gather(docling_future, images_future)
        except Exception as e:
            return f"❌ Parsing Error: {str(e)}"
        
        progress(0.4, desc=f"Analyzing {len(raw_images)} Charts...")
        image_tasks = [self.analyze_image(img["bytes"], img["page"]) for img in raw_images]
        image_descriptions = await asyncio.gather(*image_tasks)
        
        final_points = []
        for c in text_chunks:
            final_points.append({
                "text": c["text"],
                "type": "text",
                "source": "docling"
            })
            
        for desc in image_descriptions:
            if desc:
                final_points.append({
                    "text": desc,
                    "type": "chart",
                    "source": "vision"
                })

        if not final_points: return "⚠️ No content extracted."

        progress(0.7, desc=f"Embedding {len(final_points)} vectors...")
        
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
        
        batch_size = 50
        for i in range(0, len(qdrant_points), batch_size):
            await self.client.upsert(COLLECTION_NAME, qdrant_points[i:i+batch_size])
            
        return f"✅ Ingested {len(qdrant_points)} chunks!"

    async def query(self, text: str):
        vec = list(self.embed_model.embed([text]))[0].tolist()
        
        # FIX: Await the response object FIRST, then access .points
        response = await self.client.query_points(COLLECTION_NAME, query=vec, limit=10)
        hits = response.points
        
        ctx = "\n\n".join([f"[Type: {h.payload.get('type', 'text')}]\n{h.payload['text']}" for h in hits])
        
        prompt = f"Role: Financial Analyst. Answer using context.\nContext:\n{ctx}\nQuestion: {text}"
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text, []

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
    # Update status after ingest
    status = await rag.check_database()
    return result, status

async def chat_wrapper(message, history):
    rag = get_rag()
    response_text, _ = await rag.query(message)
    return response_text

# --- LAYOUT ---
with gr.Blocks(title="Fast Docling RAG") as demo:
    gr.Markdown("## 🚀 Fast Docling RAG (Hybrid Pipeline)")
    
    # STATUS BOX
    status_box = gr.Markdown("🔄 Checking Knowledge Base...")
    demo.load(update_status_on_load, outputs=[status_box])
    
    with gr.Accordion("📂 Data Ingestion", open=False):
        with gr.Row():
            f_in = gr.File(label="PDF")
            u_in = gr.Textbox(label="URL")
        btn = gr.Button("Ingest")
        out = gr.Textbox(label="Result")
        
        # Updates both result box and status box
        btn.click(run_ingest, [f_in, u_in], [out, status_box])
    
    chat = gr.ChatInterface(
        fn=chat_wrapper,
        type="messages"
    )

if __name__ == "__main__":
    demo.launch()