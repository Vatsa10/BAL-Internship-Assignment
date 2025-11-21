import asyncio
import os
import io
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
import requests
import gradio as gr
import qdrant_client 
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dotenv import load_dotenv

# --- LIBRARIES ---
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai

# Safe Import for PaddleOCR to prevent crashes if engine is missing
PADDLE_AVAILABLE = False
try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: 'paddlepaddle' or 'paddleocr' not found. OCR will be skipped.")

# pip install flashrank
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️ Warning: 'flashrank' not found. Re-ranking will be skipped.")

# --- CONFIGURATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

COLLECTION_NAME = "imf_policy_reports_final_v4" 
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Settings 
WINDOW_SIZE = 6  
WINDOW_OVERLAP = 2

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. HELPER FUNCTIONS (CPU BOUND) ---

def _run_paddle_ocr(image_bytes: bytes) -> str:
    """Runs PaddleOCR on an image byte stream."""
    if not PADDLE_AVAILABLE:
        return ""

    try:
        # FIX: Removed 'show_log=False' (deprecated in newer paddleocr)
        ocr = PaddleOCR(use_angle_cls=True, lang='en') 
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = ocr.ocr(img, cls=True)
        
        extracted_text = []
        if result and isinstance(result, list) and len(result) > 0 and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    extracted_text.append(line[1][0])
        return "\n".join(extracted_text)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def _create_sentence_window_chunks(text: str, page_num: int) -> List[Dict]:
    """Splits text into sentences and creates overlapping windows."""
    if not text: return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10] 
    
    if not sentences: return []
    
    if len(sentences) <= WINDOW_SIZE:
        return [{
            "text": f"Page {page_num} Policy Text: {text}",
            "type": "text_window",
            "metadata": {"window_span": f"0-{len(sentences)}"}
        }]

    chunks = []
    step = WINDOW_SIZE - WINDOW_OVERLAP
    for i in range(0, len(sentences), step):
        window = sentences[i : i + WINDOW_SIZE]
        chunk_text = " ".join(window)
        
        if len(chunk_text) < 50: continue 
        
        context_aware_text = f"Page {page_num} Policy Text: {chunk_text}"
        
        chunks.append({
            "text": context_aware_text,
            "type": "text_window",
            "metadata": {
                "sentence_start": i,
                "sentence_end": i + len(window)
            }
        })
        
        if i + WINDOW_SIZE >= len(sentences):
            break
            
    return chunks

def _cpu_process_page_context_aware(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Processes a page using Table Extraction + Sentence Window Chunking."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        result_data = {
            "page_num": page_num,
            "chunks": [],
            "images": [],
            "needs_ocr": False
        }

        # A. TABLE EXTRACTION 
        tables = page.find_tables()
        if tables.tables:
            for tab in tables:
                md_text = tab.to_markdown()
                if len(md_text) > 30:
                    result_data["chunks"].append({
                        "text": f"Macroeconomic Table on Page {page_num}:\n{md_text}",
                        "type": "table",
                        "metadata": {"source": "imf_table"}
                    })
                page.add_redact_annot(tab.bbox)
            page.apply_redactions()

        # B. IMAGE EXTRACTION
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if len(base_image["image"]) > 5000: 
                result_data["images"].append(base_image["image"])

        # C. TEXT CHUNKING
        text_content = page.get_text()
        if len(text_content) < 50:
            result_data["needs_ocr"] = True
            pix = page.get_pixmap()
            result_data["scan_bytes"] = pix.tobytes("png")
        else:
            window_chunks = _create_sentence_window_chunks(text_content, page_num)
            result_data["chunks"].extend(window_chunks)

        doc.close()
        return result_data
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return {"page_num": page_num, "chunks": [], "images": [], "needs_ocr": False}

# --- 2. ASYNC RAG PIPELINE (OPTIMIZED) ---

class AsyncContextRAG:
    def __init__(self):
        self.process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        
        # Initialize Qdrant Client
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        self.embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Local Re-Ranker (FlashRank)
        if FLASHRANK_AVAILABLE:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        else:
            self.reranker = None
            
        self.query_cache = {}
        self.sem = asyncio.Semaphore(10) 

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    async def analyze_image(self, image_bytes: bytes) -> str:
        """Rate-limited image analysis"""
        async with self.sem: # Wait for slot
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                prompt = "Identify chart type, axes, and key trends. Be concise."
                response = await self.gemini_model.generate_content_async([prompt, img])
                return response.text
            except Exception as e:
                return f"Chart analysis unavailable: {str(e)}"

    async def _enrich_page_async(self, page_data):
        """Helper to process a SINGLE page's OCR and Vision tasks concurrently."""
        loop = asyncio.get_running_loop()
        page_num = page_data["page_num"]
        
        # 1. Run OCR (if needed) in Parallel Executor
        if page_data["needs_ocr"] and PADDLE_AVAILABLE:
            try:
                ocr_text = await loop.run_in_executor(
                    self.process_executor, _run_paddle_ocr, page_data["scan_bytes"]
                )
                ocr_chunks = _create_sentence_window_chunks(ocr_text, page_num)
                page_data["chunks"].extend(ocr_chunks)
            except Exception as e:
                print(f"OCR Failed on page {page_num}: {e}")

        # 2. Run Vision AI (if needed) concurrently
        if page_data["images"]:
            img_tasks = [self.analyze_image(b) for b in page_data["images"]]
            captions = await asyncio.gather(*img_tasks)
            
            for cap in captions:
                page_data["chunks"].append({
                    "text": f"Chart/Figure on Page {page_num}: {cap}",
                    "type": "chart", 
                    "metadata": {"source": "imf_chart"}
                })
        
        return page_data["chunks"]

    async def ingest_document(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found."
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        progress(0.1, desc=f"Parsing {total_pages} pages (CPU Parallel)...")
        loop = asyncio.get_running_loop()
        
        # --- STAGE 1: CPU Parallel Parsing ---
        parse_tasks = [
            loop.run_in_executor(self.process_executor, _cpu_process_page_context_aware, pdf_path, i)
            for i in range(total_pages)
        ]
        raw_pages = await asyncio.gather(*parse_tasks)

        # --- STAGE 2: Async Parallel Enrichment ---
        progress(0.3, desc="Enriching pages (OCR + Vision AI)...")
        enrichment_tasks = [self._enrich_page_async(page) for page in raw_pages]
        all_page_chunks = await asyncio.gather(*enrichment_tasks)
        
        # Flatten chunks
        all_chunks = [chunk for page_chunks in all_page_chunks for chunk in page_chunks]

        if not all_chunks:
            return "⚠️ No indexable content found."

        # --- STAGE 3: Bulk Embedding & Indexing ---
        progress(0.7, desc=f"Embedding {len(all_chunks)} chunks...")
        
        points_to_upsert = []
        import uuid

        # Batch Embedding
        texts = [c["text"] for c in all_chunks]
        embeddings = list(self.embed_model.embed(texts))

        for i, chunk in enumerate(all_chunks):
            points_to_upsert.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunk["text"],
                    "type": chunk["type"],
                    "page": chunk.get("metadata", {}).get("page_num", 0), 
                    "metadata": chunk.get("metadata", {})
                }
            ))

        # Batch Upload
        batch_size = 100
        for i in range(0, len(points_to_upsert), batch_size):
            await self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upsert[i:i+batch_size]
            )
            
        return f"✅ Speed Ingestion Complete! Indexed {len(points_to_upsert)} chunks in {total_pages} pages."

    async def query(self, user_query: str):
        if user_query in self.query_cache:
            return self.query_cache[user_query]

        # 1. Embed
        query_vec = list(self.embed_model.embed([user_query]))[0].tolist()
        
        # 2. Search
        search_results = await self.client.search_points(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=25
        )

        if not search_results:
            return "No relevant data found.", []

        # 3. Re-Rank (if available)
        if FLASHRANK_AVAILABLE and self.reranker:
            passages = [
                {"id": hit.id, "text": hit.payload["text"], "meta": hit.payload}
                for hit in search_results
            ]
            rerank_request = RerankRequest(query=user_query, passages=passages)
            ranked_results = self.reranker.rank(rerank_request)
            top_context = ranked_results[:5]
        else:
            top_context = search_results[:5]

        context_str = ""
        citations = []
        for hit in top_context:
            # Handle differences between Qdrant hits and FlashRank hits
            meta = hit["meta"] if isinstance(hit, dict) else hit.payload
            text = hit["text"] if isinstance(hit, dict) else hit.payload["text"]
            
            page_num = meta.get('page', '?')
            citations.append(f"Page {page_num}")
            context_str += f"\n[Source: Page {page_num} | {meta.get('type', 'text')}]\n{text}\n"

        # 4. Generation
        prompt = f"""
        Role: Expert Financial Policy Analyst.
        Task: Answer the question using ONLY the context below.
        Rules:
        1. Be concise. No fluff.
        2. If numbers are present, quote them exactly.
        3. Cite sources (e.g. [Page 4]).
        
        CONTEXT:
        {context_str}

        QUESTION: 
        {user_query}
        """
        
        response = await self.gemini_model.generate_content_async(prompt)
        final_answer = response.text
        
        self.query_cache[user_query] = (final_answer, list(set(citations)))
        return final_answer, list(set(citations))

# --- 3. GRADIO INTERFACE ---

rag_system = AsyncContextRAG()

async def handle_ingest(file_obj, url_text):
    await rag_system.initialize_db()
    target_path = ""
    
    if url_text and url_text.strip():
        try:
            response = requests.get(url_text)
            if response.status_code == 200:
                filename = "downloaded_report.pdf"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                target_path = filename
            else:
                return f"❌ Error: {response.status_code}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    elif file_obj is not None:
        target_path = file_obj.name
    else:
        return "⚠️ Upload file or enter URL."

    return await rag_system.ingest_document(target_path)

async def chat_response(message, history):
    answer, citations = await rag_system.query(message)
    citation_str = "\n\n📚 **Sources:** " + ", ".join(citations) if citations else ""
    return answer + citation_str

# --- 4. UI ---

with gr.Blocks(title="Optimized Financial RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Optimized Financial Policy RAG")
    gr.Markdown("Features: Async Ingestion • Re-Ranking (FlashRank) • Caching • Multi-Modal")

    with gr.Accordion("📂 Ingestion", open=True):
        with gr.Row():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            url_input = gr.Textbox(label="OR PDF URL", placeholder="https://...")
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_status = gr.Textbox(label="Status")

    chatbot = gr.ChatInterface(
        fn=chat_response,
        type="messages", 
        title="💬 Analyst Chat",
        description="Ask about GDP, Debt, or Charts.",
    )

    ingest_btn.click(handle_ingest, [file_input, url_input], [ingest_status])

if __name__ == "__main__":
    demo.launch()