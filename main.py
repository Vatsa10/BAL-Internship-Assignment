import asyncio
import os
import io
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

# --- LIBRARIES ---
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "financial_docs_context_v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Settings
WINDOW_SIZE = 5  # Number of sentences per chunk
WINDOW_OVERLAP = 2  # Number of sentences to overlap

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. CPU-BOUND HELPER FUNCTIONS ---

def _run_paddle_ocr(image_bytes: bytes) -> str:
    """Runs PaddleOCR on an image byte stream."""
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = ocr.ocr(img, cls=True)
    extracted_text = []
    if result and result[0]:
        for line in result[0]:
            extracted_text.append(line[1][0])
    return "\n".join(extracted_text)

def _create_sentence_window_chunks(text: str, page_num: int) -> List[Dict]:
    """
    Splits text into sentences and creates overlapping windows.
    Strategy: Context-Aware Sliding Window.
    """
    # 1. Split into sentences (Simple Regex for speed)
    # Matches punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10] # Filter noise
    
    chunks = []
    num_sentences = len(sentences)
    
    # 2. Sliding Window Loop
    if num_sentences == 0: return []
    
    # If text is short, just take it all
    if num_sentences <= WINDOW_SIZE:
        return [{
            "text": text,
            "type": "text_window",
            "metadata": {"window_span": f"0-{num_sentences}"}
        }]

    step = WINDOW_SIZE - WINDOW_OVERLAP
    for i in range(0, num_sentences, step):
        # Create the window
        window = sentences[i : i + WINDOW_SIZE]
        chunk_text = " ".join(window)
        
        if len(chunk_text) < 50: continue # Skip tiny chunks
        
        # Context Injection: Prepend metadata to the text content for embedding
        # This helps the LLM understand *where* this snippet fits.
        context_aware_text = f"Page {page_num} Excerpt: {chunk_text}"
        
        chunks.append({
            "text": context_aware_text,
            "type": "text_window",
            "metadata": {
                "sentence_start": i,
                "sentence_end": i + len(window)
            }
        })
        
        if i + WINDOW_SIZE >= num_sentences:
            break
            
    return chunks

def _cpu_process_page_context_aware(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """
    Processes a page using Table Extraction + Sentence Window Chunking.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    result_data = {
        "page_num": page_num,
        "chunks": [],
        "images": [],
        "needs_ocr": False
    }

    # --- A. TABLE EXTRACTION (Structured) ---
    tables = page.find_tables()
    if tables.tables:
        for tab in tables:
            md_text = tab.to_markdown()
            if len(md_text) > 30:
                result_data["chunks"].append({
                    "text": f"Table on Page {page_num}:\n{md_text}",
                    "type": "table",
                    "metadata": {"source": "pymupdf_table"}
                })
            # Mask table to avoid duplicate text processing
            page.add_redact_annot(tab.bbox)
        page.apply_redactions()

    # --- B. IMAGE EXTRACTION ---
    image_list = page.get_images(full=True)
    for img in image_list:
        xref = img[0]
        base_image = doc.extract_image(xref)
        if len(base_image["image"]) > 5000: 
            result_data["images"].append(base_image["image"])

    # --- C. CONTEXT-AWARE TEXT CHUNKING ---
    text_content = page.get_text()
    
    # Check for Scans
    if len(text_content) < 50:
        result_data["needs_ocr"] = True
        pix = page.get_pixmap()
        result_data["scan_bytes"] = pix.tobytes("png")
    else:
        # Apply Sentence Windowing
        window_chunks = _create_sentence_window_chunks(text_content, page_num)
        result_data["chunks"].extend(window_chunks)

    doc.close()
    return result_data

# --- 2. ASYNC RAG PIPELINE ---

class AsyncContextRAG:
    def __init__(self):
        self.process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        self.client = AsyncQdrantClient(url=QDRANT_URL)
        self.embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    async def analyze_image(self, image_bytes: bytes) -> str:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            # VLM Context Prompt
            prompt = "Analyze this image for a financial report. Detail any trends, numbers, or axis labels."
            response = await self.gemini_model.generate_content_async([prompt, img])
            return response.text
        except Exception:
            return "Image analysis failed."

    async def ingest_document(self, pdf_path: str):
        if not os.path.exists(pdf_path): return
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        print(f"🚀 Ingesting {pdf_path} ({total_pages} pages) with Window Chunking...")

        loop = asyncio.get_running_loop()
        
        # 1. Parallel Processing
        tasks = [
            loop.run_in_executor(self.process_executor, _cpu_process_page_context_aware, pdf_path, i)
            for i in range(total_pages)
        ]
        raw_pages = await asyncio.gather(*tasks)

        # 2. Enrichment & Upload
        points_to_upsert = []
        import uuid

        for page_data in raw_pages:
            page_num = page_data["page_num"]
            
            # OCR Fallback
            if page_data["needs_ocr"]:
                ocr_text = await loop.run_in_executor(
                    self.process_executor, _run_paddle_ocr, page_data["scan_bytes"]
                )
                # Chunk OCR results too!
                ocr_chunks = _create_sentence_window_chunks(ocr_text, page_num)
                page_data["chunks"].extend(ocr_chunks)

            # Image Analysis
            if page_data["images"]:
                img_tasks = [self.analyze_image(b) for b in page_data["images"]]
                captions = await asyncio.gather(*img_tasks)
                for cap in captions:
                    page_data["chunks"].append({
                        "text": f"Chart on Page {page_num}: {cap}",
                        "type": "chart", 
                        "metadata": {}
                    })

            if not page_data["chunks"]: continue

            # Embed & Create Points
            texts = [c["text"] for c in page_data["chunks"]]
            embeddings = list(self.embed_model.embed(texts))

            for i, chunk in enumerate(page_data["chunks"]):
                points_to_upsert.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": chunk["text"],
                        "type": chunk["type"],
                        "page": page_num,
                        "metadata": chunk["metadata"]
                    }
                ))

        # Batch Upsert
        if points_to_upsert:
            batch_size = 100
            for i in range(0, len(points_to_upsert), batch_size):
                await self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_upsert[i:i+batch_size]
                )
            print(f"✅ Context-Aware Indexing Complete: {len(points_to_upsert)} vectors.")

    async def query(self, user_query: str):
        query_vec = list(self.embed_model.embed([user_query]))[0].tolist()
        
        # Retrieve more chunks because they are smaller/overlapping
        search_results = await self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=8 
        )

        context_str = ""
        citations = []
        for hit in search_results:
            meta = hit.payload
            citations.append(f"Page {meta['page']}")
            context_str += f"\n[Source: Page {meta['page']} | Type: {meta['type']}]\n{meta['text']}\n"

        prompt = f"""
        Answer the question using the provided context chunks. 
        Note that chunks may overlap (sliding window). Merge information logically.
        
        Context:
        {context_str}
        
        Question: {user_query}
        """
        
        response = await self.gemini_model.generate_content_async(prompt)
        return response.text, list(set(citations))

# --- EXECUTION ---
async def main():
    rag = AsyncContextRAG()
    await rag.initialize_db()
    
    pdf_file = "imf_report.pdf" # <--- CHANGE THIS
    if os.path.exists(pdf_file):
        # INTERACTIVE TOGGLE: Helps you control when to ingest vs. when to chat
        choice = input(f"Found {pdf_file}. Do you want to Index/Ingest it? (y/n): ")
        if choice.lower().strip() == 'y':
            await rag.ingest_document(pdf_file)
    else:
        print(f" PDF {pdf_file} not found. Please check path.")
        
    print("\n RAG Ready. Ask away!")
    while True:
        q = input("\nQuery: ")
        if q == "exit": break
        ans, sources = await rag.query(q)
        print(f"AI: {ans}\nSources: {sources}")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())