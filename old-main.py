import asyncio
import os
import io
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
import requests
import gradio as gr
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dotenv import load_dotenv

# --- LIBRARIES ---
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
import google.generativeai as genai
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validation
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found in .env file")

COLLECTION_NAME = "imf_policy_reports_v1" 
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Settings 
WINDOW_SIZE = 6  
WINDOW_OVERLAP = 2

genai.configure(api_key=GEMINI_API_KEY)

# --- 1. HELPER FUNCTIONS ---

def _run_paddle_ocr(image_bytes: bytes) -> str:
    """Runs PaddleOCR on an image byte stream."""
    # Using a lightweight CPU config for PaddleOCR
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
    """Splits text into sentences and creates overlapping windows."""
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
    """Processes a page using Table Extraction + Sentence Window Chunking with pymupdf-layout."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    # Initialize layout analyzer
    layout_analyzer = LayoutAnalyzer(page)
    
    result_data = {
        "page_num": page_num,
        "chunks": [],
        "images": [],
        "needs_ocr": False
    }

    try:
        # Get text blocks with layout information
        blocks = layout_analyzer.get_blocks()
        
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
                # Redact the table area to avoid duplicate text extraction
                page.add_redact_annot(tab.bbox)
            page.apply_redactions()

        # B. IMAGE EXTRACTION
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if len(base_image["image"]) > 5000: 
                result_data["images"].append(base_image["image"])

        # C. TEXT EXTRACTION WITH LAYOUT
        text_blocks = []
        for block in blocks:
            if hasattr(block, 'text') and block.text.strip():
                text_blocks.append(block.text.strip())
        
        # Combine text blocks with layout awareness
        text_content = "\n\n".join(text_blocks)
        
        if len(text_content) < 50:
            result_data["needs_ocr"] = True
            pix = page.get_pixmap()
            result_data["scan_bytes"] = pix.tobytes("png")
        else:
            window_chunks = _create_sentence_window_chunks(text_content, page_num)
            result_data["chunks"].extend(window_chunks)
            
    except Exception as e:
        print(f"Error processing page {page_num}: {str(e)}")
        # Fallback to basic text extraction if layout analysis fails
        text_content = page.get_text()
        if text_content:
            window_chunks = _create_sentence_window_chunks(text_content, page_num)
            result_data["chunks"].extend(window_chunks)
    
    doc.close()
    return result_data

# --- 2. ASYNC RAG PIPELINE ---

class AsyncContextRAG:
    def __init__(self):
        self.process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Configure SSL context to handle certificate verification
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize embedding model with custom HTTP client
        self.embed_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            ssl_verify=False,  # Disable SSL verification
            cache_dir="./model_cache"  # Cache the model locally
        )
        
        # Configure Gemini model
        genai.configure(api_key=GEMINI_API_KEY)
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
            prompt = "Analyze this image from an IMF/Financial Policy report. Identify charts, trends, fan charts, or heatmaps. Summarize the economic signal."
            response = await self.gemini_model.generate_content_async([prompt, img])
            return response.text
        except Exception:
            return "Chart analysis unavailable."

    async def ingest_document(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found."
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        progress(0, desc=f"Starting ingestion for {total_pages} pages...")
        loop = asyncio.get_running_loop()
        
        # 1. Parsing
        tasks = [
            loop.run_in_executor(self.process_executor, _cpu_process_page_context_aware, pdf_path, i)
            for i in range(total_pages)
        ]
        raw_pages = await asyncio.gather(*tasks)

        # 2. Enrichment & Upload
        points_to_upsert = []
        import uuid

        for i, page_data in enumerate(raw_pages):
            progress((i / total_pages), desc=f"Enriching Page {i+1} (OCR/Vision)...")
            page_num = page_data["page_num"]
            
            # OCR
            if page_data["needs_ocr"]:
                ocr_text = await loop.run_in_executor(
                    self.process_executor, _run_paddle_ocr, page_data["scan_bytes"]
                )
                ocr_chunks = _create_sentence_window_chunks(ocr_text, page_num)
                page_data["chunks"].extend(ocr_chunks)

            # Vision
            if page_data["images"]:
                img_tasks = [self.analyze_image(b) for b in page_data["images"]]
                captions = await asyncio.gather(*img_tasks)
                for cap in captions:
                    page_data["chunks"].append({
                        "text": f"Chart/Figure on Page {page_num}: {cap}",
                        "type": "chart", 
                        "metadata": {"source": "imf_chart"}
                    })

            if not page_data["chunks"]: continue

            texts = [c["text"] for c in page_data["chunks"]]
            embeddings = list(self.embed_model.embed(texts))

            for idx, chunk in enumerate(page_data["chunks"]):
                points_to_upsert.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx].tolist(),
                    payload={
                        "text": chunk["text"],
                        "type": chunk["type"],
                        "page": page_num,
                        "metadata": chunk["metadata"]
                    }
                ))

        if points_to_upsert:
            batch_size = 100
            for i in range(0, len(points_to_upsert), batch_size):
                await self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_upsert[i:i+batch_size]
                )
            return f"Success! Indexed {len(points_to_upsert)} chunks from {total_pages} pages."
        else:
            return "No indexable content found."

    async def query(self, user_query: str):
        query_vec = list(self.embed_model.embed([user_query]))[0].tolist()
        
        search_results = await self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=10
        )

        context_str = ""
        citations = []
        for hit in search_results:
            meta = hit.payload
            citations.append(f"Page {meta['page']}")
            context_str += f"\n[Source: Page {meta['page']} | Type: {meta['type']}]\n{meta['text']}\n"

        prompt = f"""
        You are an expert economic policy analyst. Answer using ONLY the context.
        Cite sources (e.g., [Source: Page 4]). If data is missing, state "Data not found."

        CONTEXT:
        {context_str}

        QUESTION: 
        {user_query}
        """
        
        response = await self.gemini_model.generate_content_async(prompt)
        return response.text, list(set(citations))

# --- 3. GRADIO INTERFACE LOGIC ---

rag_system = AsyncContextRAG()

async def handle_ingest(file_obj, url_text):
    # Initialize DB first
    await rag_system.initialize_db()
    
    target_path = ""
    
    # Scenario A: URL provided
    if url_text and url_text.strip():
        try:
            response = requests.get(url_text)
            if response.status_code == 200:
                filename = "downloaded_report.pdf"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                target_path = filename
            else:
                return f"Error downloading URL: Status {response.status_code}"
        except Exception as e:
            return f"Error downloading URL: {str(e)}"
            
    # Scenario B: File Uploaded
    elif file_obj is not None:
        target_path = file_obj.name
    
    else:
        return "Please upload a file or provide a URL."

    return await rag_system.ingest_document(target_path)

async def chat_response(message, history):
    answer, citations = await rag_system.query(message)
    
    citation_str = "\n\nSources: " + ", ".join(citations) if citations else ""
    final_response = answer + citation_str
    
    return final_response

# --- 4. UI BUILDER ---

with gr.Blocks(title="Financial Policy Analyst RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Financial Policy Analyst RAG")
    gr.Markdown("Async, Multi-Modal (Text + Tables + Charts) analysis of IMF & Finance Reports.")

    with gr.Accordion("Knowledge Base Ingestion", open=True):
        with gr.Row():
            file_input = gr.File(label="Upload PDF Report", file_types=[".pdf"])
            url_input = gr.Textbox(label="OR Enter PDF URL", placeholder="https://www.imf.org/.../report.pdf")
        
        ingest_btn = gr.Button("Ingest Document", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        chatbot = gr.ChatInterface(
            fn=chat_response,
            title="Policy Analyst Chat",
            description="Ask about GDP projections, debt sustainability, or specific charts.",
            examples=["What are the downside risks to growth?", "Summarize the debt sustainability analysis.", "What does the fan chart indicate about inflation?"],
            textbox=gr.Textbox(placeholder="Ask a question about the document...", container=False, scale=7),
            submit_btn=gr.Button(value="Send", variant="primary"),
            stop_btn=None,
            retry_btn=None,
            undo_btn=None,
            clear_btn=None,
            chatbot=gr.Chatbot(
                value=[],
                height=500,
                show_copy_button=True,
                show_share_button=True,
                likeable=True,
                layout="panel",
                bubble_full_width=False,
                container=True,
                type="messages"
            )
        )

    # Event Handlers
    ingest_btn.click(
        fn=handle_ingest, 
        inputs=[file_input, url_input], 
        outputs=[ingest_status]
    )

# --- 5. LAUNCH ---

if __name__ == "__main__":
    demo.launch()