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
import uuid
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dotenv import load_dotenv

# --- LIBRARIES ---
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai

# --- VERSION & IMPORT CHECKS ---
# print(f"🔌 Qdrant Client Version: {qdrant_client.__version__}")

# Safe Import for PaddleOCR
PADDLE_AVAILABLE = False
try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    # Suppress Paddle Logs
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
except ImportError:
    print("⚠️ Warning: 'paddleocr' engine not found. OCR will be skipped.")

# Safe Import for FlashRank
FLASHRANK_AVAILABLE = False
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: 'flashrank' not found. Re-ranking will be skipped.")

# --- CONFIGURATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

COLLECTION_NAME = "imf_policy_reports_final_v4_2" 
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

WINDOW_SIZE = 6  
WINDOW_OVERLAP = 2

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. HELPER FUNCTIONS ---

def _run_paddle_ocr(image_bytes: bytes) -> str:
    """Runs PaddleOCR on an image byte stream."""
    if not PADDLE_AVAILABLE: return ""
    try:
        # Initialize PaddleOCR
        # use_angle_cls=False prevents 'cls' argument errors in some versions
        ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False) 
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run OCR
        result = ocr.ocr(img, cls=False)
        
        extracted_text = []
        if result and isinstance(result, list) and len(result) > 0 and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    extracted_text.append(line[1][0])
        return "\n".join(extracted_text)
    except Exception as e:
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
            "metadata": {
                "window_span": f"0-{len(sentences)}",
                "page_num": page_num 
            }
        }]

    chunks = []
    step = WINDOW_SIZE - WINDOW_OVERLAP
    for i in range(0, len(sentences), step):
        window = sentences[i : i + WINDOW_SIZE]
        chunk_text = " ".join(window)
        if len(chunk_text) < 50: continue 
        
        chunks.append({
            "text": f"Page {page_num} Policy Text: {chunk_text}",
            "type": "text_window",
            "metadata": {
                "sentence_start": i, 
                "sentence_end": i + len(window),
                "page_num": page_num
            }
        })
        if i + WINDOW_SIZE >= len(sentences): break
    return chunks

def _cpu_process_page(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Processes a page using Table Extraction + Sentence Window Chunking."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        display_page_num = page_num + 1
        
        result_data = {
            "page_num": display_page_num, 
            "chunks": [], 
            "images": [], 
            "needs_ocr": False
        }

        # Table Extraction
        tables = page.find_tables()
        if tables.tables:
            for tab in tables:
                md_text = tab.to_markdown()
                if len(md_text) > 30:
                    result_data["chunks"].append({
                        "text": f"Table on Page {display_page_num}:\n{md_text}",
                        "type": "table",
                        "metadata": {
                            "source": "imf_table",
                            "page_num": display_page_num
                        }
                    })
                page.add_redact_annot(tab.bbox)
            page.apply_redactions()

        # Image Extraction
        for img in page.get_images(full=True):
            try:
                base_image = doc.extract_image(img[0])
                if len(base_image["image"]) > 5000: 
                    result_data["images"].append(base_image["image"])
            except:
                pass

        # Text Extraction
        text_content = page.get_text()
        if len(text_content) < 50:
            result_data["needs_ocr"] = True
            result_data["scan_bytes"] = page.get_pixmap().tobytes("png")
        else:
            result_data["chunks"].extend(_create_sentence_window_chunks(text_content, display_page_num))

        doc.close()
        return result_data
    except Exception as e:
        print(f"Error Page {page_num}: {e}")
        return {"page_num": page_num + 1, "chunks": [], "images": [], "needs_ocr": False}

# --- 2. ASYNC RAG PIPELINE ---

class AsyncContextRAG:
    def __init__(self):
        self.process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        
        # FIX 1: Increased Timeout for Qdrant Client to prevent ReadTimeout errors
        self.client = AsyncQdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            timeout=60.0  # 60 seconds timeout
        )
        
        self.embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        if FLASHRANK_AVAILABLE:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        else:
            self.reranker = None
            
        self.query_cache = {}
        self.sem = asyncio.Semaphore(10)

    async def get_collection_status(self):
        try:
            if await self.client.collection_exists(COLLECTION_NAME):
                count_result = await self.client.count(COLLECTION_NAME)
                return True, count_result.count
            return False, 0
        except Exception as e:
            return False, f"Error: {str(e)}"

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    async def analyze_image(self, image_bytes: bytes) -> str:
        async with self.sem:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                prompt = "Identify chart type, axes, and key trends. Be concise."
                response = await self.gemini_model.generate_content_async([prompt, img])
                return response.text
            except Exception:
                return "Chart analysis unavailable."

    async def _enrich_page(self, page_data):
        loop = asyncio.get_running_loop()
        page_num = page_data["page_num"]
        
        if page_data["needs_ocr"] and PADDLE_AVAILABLE:
            try:
                ocr_text = await loop.run_in_executor(self.process_executor, _run_paddle_ocr, page_data["scan_bytes"])
                if ocr_text:
                    page_data["chunks"].extend(_create_sentence_window_chunks(ocr_text, page_num))
            except Exception as e:
                print(f"Page {page_num} OCR Skip: {e}")

        if page_data["images"]:
            img_tasks = [self.analyze_image(b) for b in page_data["images"]]
            captions = await asyncio.gather(*img_tasks)
            for cap in captions:
                page_data["chunks"].append({
                    "text": f"Chart/Figure on Page {page_num}: {cap}",
                    "type": "chart", 
                    "metadata": {
                        "source": "imf_chart",
                        "page_num": page_num
                    }
                })
        return page_data["chunks"]

    async def ingest_document(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found."
        
        await self.initialize_db()
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        progress(0.1, desc=f"Parsing {total_pages} pages...")
        loop = asyncio.get_running_loop()
        
        # 1. CPU Parsing
        parse_tasks = [loop.run_in_executor(self.process_executor, _cpu_process_page, pdf_path, i) for i in range(total_pages)]
        raw_pages = await asyncio.gather(*parse_tasks)

        # 2. Enrichment
        progress(0.3, desc="Enriching (OCR/Vision)...")
        enrichment_tasks = [self._enrich_page(page) for page in raw_pages]
        all_page_chunks = await asyncio.gather(*enrichment_tasks)
        all_chunks = [c for p in all_page_chunks for c in p]

        if not all_chunks: return "⚠️ No content found."

        # 3. Embedding
        progress(0.7, desc=f"Embedding {len(all_chunks)} chunks...")
        texts = [c["text"] for c in all_chunks]
        embeddings = list(self.embed_model.embed(texts))
        
        points = []
        for i, c in enumerate(all_chunks):
            meta = c.get("metadata", {})
            if meta is None: meta = {}
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": c["text"],
                    "type": c["type"],
                    "page": meta.get("page_num", 0),
                    "metadata": meta
                }
            ))

        # FIX 2: Reduced Batch Size to 50 (Prevents Timeouts) & Added Retry Logic
        batch_size = 50
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            progress(0.8 + (0.2 * (i / len(points))), desc=f"Uploading Batch {(i//batch_size)+1}/{total_batches}...")
            
            retries = 3
            for attempt in range(retries):
                try:
                    await self.client.upsert(COLLECTION_NAME, batch)
                    break # Success
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"❌ Batch Upload Failed after {retries} attempts: {e}")
                    else:
                        await asyncio.sleep(1) # Wait before retry
            
        return f"✅ Indexed {len(points)} chunks from {total_pages} pages."

    async def query(self, user_query: str):
        if user_query in self.query_cache: return self.query_cache[user_query]

        query_vec = list(self.embed_model.embed([user_query]))[0].tolist()
        
        try:
            response = await self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                limit=25
            )
            search_results = response.points
        except AttributeError:
            try:
                search_results = await self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vec,
                    limit=25
                )
            except Exception as e:
                return f"❌ DB Error: {str(e)}", []

        if not search_results: return "No data found.", []

        # Re-Rank
        if FLASHRANK_AVAILABLE and self.reranker:
            passages = [{"id": hit.id, "text": hit.payload["text"], "meta": hit.payload} for hit in search_results]
            ranked = self.reranker.rerank(RerankRequest(query=user_query, passages=passages))
            top_context = ranked[:5]
        else:
            top_context = search_results[:5]

        context_str = "\n".join([f"[Source: Page {hit['meta'].get('page','?')}]\n{hit['text']}" for hit in top_context])
        citations = [f"Page {hit['meta'].get('page','?')}" for hit in top_context]

        prompt = f"""
        Role: Financial Analyst. 
        Task: Answer using ONLY context.
        Context: {context_str}
        Question: {user_query}
        """
        
        response = await self.gemini_model.generate_content_async(prompt)
        self.query_cache[user_query] = (response.text, list(set(citations)))
        return response.text, list(set(citations))

# --- 3. GRADIO UI ---

rag_system = AsyncContextRAG()

async def check_db_status():
    exists, count = await rag_system.get_collection_status()
    if exists:
        return f"✅ **Knowledge Base Ready**\nCollection: `{COLLECTION_NAME}`\nVectors: {count}"
    else:
        return f"⚠️ **Empty Knowledge Base**\nCollection `{COLLECTION_NAME}` not found.\nPlease ingest a document."

async def handle_ingest(file_obj, url_text):
    target_path = ""
    if url_text.strip():
        try:
            r = requests.get(url_text)
            if r.status_code == 200:
                with open("download.pdf", 'wb') as f: f.write(r.content)
                target_path = "download.pdf"
        except Exception as e: return f"Error: {e}"
    elif file_obj:
        target_path = file_obj.name
    else:
        return "Please upload a file."
    
    return await rag_system.ingest_document(target_path)

async def chat_Wrapper(msg, hist):
    ans, cits = await rag_system.query(msg)
    return f"{ans}\n\n📚 **Sources:** {', '.join(map(str, cits))}" if cits else ans

with gr.Blocks(title="IMF Analyst RAG") as demo:
    gr.Markdown("# 📈 Financial Policy Analyst RAG")
    
    # STATUS BOX
    status_display = gr.Markdown("🔄 Checking Database...")
    demo.load(check_db_status, outputs=[status_display])

    with gr.Accordion("📂 Add New Documents (Optional)", open=False):
        with gr.Row():
            f_in = gr.File(file_types=[".pdf"])
            u_in = gr.Textbox(placeholder="PDF URL")
        btn = gr.Button("Ingest")
        out = gr.Textbox(label="Result")
        btn.click(handle_ingest, [f_in, u_in], out)

    # CHAT
    gr.ChatInterface(chat_Wrapper, type="messages")

if __name__ == "__main__":
    demo.launch()