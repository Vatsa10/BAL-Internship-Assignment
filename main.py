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
import re
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
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, SparseIndexParams, FusionQuery, Fusion, Prefetch
from fastembed import TextEmbedding, SparseTextEmbedding
import google.generativeai as genai

# FlashRank
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("Warning: 'flashrank' not found. Reranking disabled.")

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# FIX: New Collection Name for Fresh Context-Aware Ingestion
COLLECTION_NAME = "docling_financial_context_v4"

if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY missing")

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

# --- 2. PIPELINE CLASS ---
class AsyncDoclingRAG:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Models
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") 
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.sem = asyncio.Semaphore(10) 
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache") if FLASHRANK_AVAILABLE else None

        print("Advanced RAG System Initialized.")

    async def check_database(self) -> str:
        try:
            if await self.client.collection_exists(COLLECTION_NAME):
                cnt = (await self.client.count(COLLECTION_NAME)).count
                if cnt > 0:
                    return f"### Ready\n**Collection:** `{COLLECTION_NAME}`\n**Vectors:** {cnt}\n*Context-Aware Mode Active*"
                else:
                    return f"### Empty\nCollection `{COLLECTION_NAME}` exists but is empty."
            return f"### Not Found\nCollection `{COLLECTION_NAME}` missing. Please ingest."
        except Exception as e:
            return f"### Connection Error\n{str(e)}"

    async def initialize_db(self):
        if not await self.client.collection_exists(COLLECTION_NAME):
            print(f"Creating Collection: {COLLECTION_NAME}")
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            )

    # --- IMAGE & CAPTION ANALYSIS (The Fix for "Figure 4") ---
    async def analyze_image_with_caption(self, image_bytes: bytes, caption: str, page_num: int) -> str:
        """Sends image + discovered caption to Gemini for deep analysis."""
        async with self.sem:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                
                prompt = f"""
                This image is labeled in the document as: "{caption}".
                
                Task:
                1. If it's a Chart/Graph: Extract the exact data points, trends, and X/Y axis labels.
                2. If it's a Table Image: Transcribe the key numbers.
                3. If it's generic: Describe it briefly.
                
                Output format:
                [Image: {caption}]
                Analysis: <detailed description>
                """
                response = await self.gemini_model.generate_content_async([prompt, img])
                return f"{response.text}"
            except: return ""

    # --- HELPER: FIND CAPTIONS NEAR IMAGES ---
    def _find_caption(self, text_blocks, img_rect):
        """
        Scans text blocks near the image to find "Figure X", "Table Y", "Box Z".
        """
        img_y0, img_y1 = img_rect[1], img_rect[3]
        best_caption = "Unlabeled Chart"
        min_dist = 200 # Search within 200px vertical distance
        
        for b in text_blocks:
            text = b[4].strip().replace("\n", " ")
            if len(text) < 5: continue
            
            block_y0, block_y1 = b[1], b[3]
            
            # Check if block is strictly above or strictly below
            is_above = block_y1 < img_y0
            is_below = block_y0 > img_y1
            
            if is_above: dist = img_y0 - block_y1
            elif is_below: dist = block_y0 - img_y1
            else: dist = 0 # Overlap
            
            if dist < min_dist:
                # Heuristic: Captions usually start with...
                if re.match(r"^(Figure|Fig|Table|Box|Chart)\s+\d+", text, re.IGNORECASE):
                    return text # Strong match
                
                # Weak match (closest text)
                if len(text) < 100: 
                    best_caption = text
                    min_dist = dist
                    
        return best_caption

    # --- PYMUPDF EXTRACTION (Context Aware) ---
    def _extract_images_with_context(self, pdf_path: str):
        doc = fitz.open(pdf_path)
        images_data = []
        
        for i, page in enumerate(doc):
            # Get text blocks for caption hunting
            text_blocks = page.get_text("blocks")
            
            # Iterate images
            image_list = page.get_images(full=True)
            for img in image_list:
                try:
                    xref = img[0]
                    # Get image location on page
                    rects = page.get_image_rects(xref)
                    if not rects: continue
                    img_rect = rects[0] 
                    
                    # Extract bytes
                    base = doc.extract_image(xref)
                    if len(base["image"]) > 5000: # Filter icons
                        # Find the caption!
                        caption = self._find_caption(text_blocks, img_rect)
                        images_data.append({
                            "bytes": base["image"], 
                            "page": i + 1,
                            "caption": caption
                        })
                except: pass
        return images_data

    # --- DOCLING TEXT EXTRACTION ---
    def _run_docling(self, pdf_path: str):
        converter = get_docling_converter()
        result = converter.convert(pdf_path)
        chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5") 
        chunks = []
        for chunk in chunker.chunk(result.document):
            # Enhance metadata
            meta = chunk.meta.export_json_dict()
            # Force page number to be int
            if 'page_label' in meta:
                try: meta['page_no'] = int(meta['page_label'])
                except: pass
            chunks.append({"text": chunk.text, "metadata": meta})
        return chunks

    # --- INGESTION ---
    async def ingest(self, pdf_path: str, progress=gr.Progress()):
        if not os.path.exists(pdf_path): return "Error: File not found"
        await self.initialize_db()
        loop = asyncio.get_running_loop()
        
        progress(0.1, desc="Parsing Layout & Finding Captions...")
        
        # Parallel Execution
        docling_fut = loop.run_in_executor(self.executor, self._run_docling, pdf_path)
        images_fut = loop.run_in_executor(self.executor, self._extract_images_with_context, pdf_path)
        
        text_chunks, raw_images = await asyncio.gather(docling_fut, images_fut)
        
        progress(0.4, desc=f"Analyzing {len(raw_images)} Visual Elements...")
        img_tasks = [
            self.analyze_image_with_caption(img["bytes"], img["caption"], img["page"]) 
            for img in raw_images
        ]
        img_descs = await asyncio.gather(*img_tasks)
        
        final_points = []
        
        # 1. Process Text Chunks
        for c in text_chunks:
            # "Sticky Header": Prepend header if available to keep context
            header = ""
            if 'headings' in c['metadata'] and c['metadata']['headings']:
                header = f"Section: {c['metadata']['headings'][0]}\n"
            
            final_text = header + c["text"]
            
            final_points.append({
                "text": final_text,
                "type": "text",
                "source": "docling",
                "metadata": c["metadata"]
            })
            
        # 2. Process Image Chunks (Now with Captions glued!)
        for i, desc in enumerate(img_descs):
            if desc:
                caption = raw_images[i]['caption']
                page = raw_images[i]['page']
                # IMPORTANT: Put the caption in the text so Hybrid Search finds it
                final_text = f"Figure/Table: {caption}\nLocation: Page {page}\nAnalysis: {desc}"
                
                final_points.append({
                    "text": final_text,
                    "type": "chart",
                    "source": "vision",
                    "metadata": {"page_no": page, "caption": caption}
                })

        progress(0.7, desc="Indexing Vectors...")
        texts = [p["text"] for p in final_points]
        
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
            
        return f"Success! Ingested {len(points)} chunks."

    # --- QUERY LOGIC ---
    async def query(self, text: str):
        dense_vec = list(self.dense_model.embed([text]))[0].tolist()
        sparse_vec = list(self.sparse_model.embed([text]))[0].as_object()
        
        # Detect query type for enhanced retrieval
        is_table_query = any(keyword in text.lower() for keyword in ["table", "macroeconomic", "indicators", "data"])
        is_chart_query = any(keyword in text.lower() for keyword in ["figure", "chart", "graph", "visual"])
        is_text_query = not is_table_query and not is_chart_query
        
        # Adjust retrieval limits based on query type
        if is_table_query:
            limit_size = 30
            top_k = 12
        elif is_chart_query:
            limit_size = 25
            top_k = 10
        else:  # text queries
            limit_size = 25
            top_k = 10
        
        # 1. Hybrid Retrieval (Increased limit to catch specific figures)
        response = await self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=30),
                Prefetch(query=sparse_vec, using="sparse", limit=30),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit_size
        )
        hits = response.points
        if not hits: return "No data found.", []

        # 2. Reranking
        if FLASHRANK_AVAILABLE and self.reranker:
            passages = [{"id": h.id, "text": h.payload["text"], "meta": h.payload} for h in hits]
            ranked = self.reranker.rerank(RerankRequest(query=text, passages=passages))
            top_hits = ranked[:top_k]
        else:
            top_hits = hits[:top_k]

        # 3. Formatting
        ctx_parts = []
        citations = []
        for h in top_hits:
            payload = h["meta"] if isinstance(h, dict) else h.payload
            text_content = h["text"] if isinstance(h, dict) else h.payload["text"]
            
            meta = payload.get("metadata", {})
            page = meta.get("page_no", "?")
            
            # If it's a chart, highlight it
            src_type = payload.get('type', 'text').upper()
            
            citation_label = f"Page {page}"
            if src_type == "CHART": citation_label += " (Graph/Chart)"
            
            citations.append(citation_label)
            ctx_parts.append(f"--- SOURCE: {citation_label} ---\n{text_content}\n")

        ctx = "\n".join(ctx_parts)
        
        # 4. Adaptive Prompt Based on Query Type
        if is_table_query:
            prompt = f"""
        You are a Senior Financial Analyst evaluating an IMF Article IV report.
        
        TASK: Analyze the table data comprehensively.
        
        INSTRUCTIONS:
        1. Answer using ONLY the provided context.
        2. Provide comprehensive analysis including:
           - What the table measures and its significance
           - Key trends across the time period (historical vs. projections)
           - Notable changes or anomalies in the data
           - Comparative analysis between different indicators
           - Economic implications of the trends shown
           - Sector-specific insights (if applicable)
        3. Extract specific data points, percentages, and years exactly as they appear.
        4. Do NOT summarize vaguely.
        
        CONTEXT:
        {ctx}
        
        QUESTION: 
        {text}
        
        RESPONSE FORMAT:
        - Start with a brief overview of what the table represents
        - Provide detailed analysis of key indicators and trends
        - Highlight significant changes or patterns
        - Explain the economic implications
        - Include specific numbers and percentages from the table
        """
        elif is_chart_query:
            prompt = f"""
        You are a Senior Financial Analyst evaluating an IMF Article IV report.
        
        TASK: Analyze the chart/figure data comprehensively.
        
        INSTRUCTIONS:
        1. Answer using ONLY the provided context.
        2. If the user asks about a Figure (e.g., Figure 4), look for "Figure 4" in the source text.
        3. Describe what the chart shows, including:
           - Main data points and trends
           - Key patterns and anomalies
           - Comparative insights
           - Economic implications
        4. If the context contains a Chart Analysis, treat it as factual data from the report.
        5. Extract specific data points and percentages exactly as they appear.
        
        CONTEXT:
        {ctx}
        
        QUESTION: 
        {text}
        
        RESPONSE FORMAT:
        - Describe what the chart/figure shows
        - Highlight key trends and patterns
        - Provide specific data points
        - Explain economic implications
        """
        else:  # text queries
            prompt = f"""
        You are a Senior Financial Analyst evaluating an IMF Article IV report.
        
        TASK: Provide a comprehensive answer to the question.
        
        INSTRUCTIONS:
        1. Answer using ONLY the provided context.
        2. Be thorough and detailed in your explanation.
        3. Include specific examples, data points, and percentages from the report.
        4. Explain the economic context and implications.
        5. Do NOT summarize vaguely - provide substantive analysis.
        6. If relevant, mention specific sections, tables, or figures from the report.
        
        CONTEXT:
        {ctx}
        
        QUESTION: 
        {text}
        
        RESPONSE FORMAT:
        - Start with a direct answer to the question
        - Provide detailed explanation with supporting evidence
        - Include specific data points and examples
        - Explain the broader economic implications
        - Reference sources where relevant
        """
        
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text, list(set(citations))

    # --- BRIEFING ---
    async def generate_briefing(self):
        dense_vec = list(self.dense_model.embed(["executive summary fiscal risks gdp outlook"]))[0].tolist()
        response = await self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="dense",
            limit=40
        )
        hits = response.points
        if not hits: return "No data."
        
        summary_ctx = "\n".join([h.payload["text"] for h in hits])
        prompt = f"Role: Senior Economist.\nTask: Create a structured Executive Briefing (Outlook, Risks, Policy Recs) from this text:\n{summary_ctx[:40000]}"
        res = await self.gemini_model.generate_content_async(prompt)
        return res.text

# --- UI ---
_rag_instance = None
def get_rag():
    global _rag_instance
    if _rag_instance is None: _rag_instance = AsyncDoclingRAG()
    return _rag_instance

async def update_status(): return await get_rag().check_database()

async def run_ingest(f, u):
    rag = get_rag()
    path = "download.pdf"
    if u: 
        with open(path, "wb") as fd: fd.write(requests.get(u).content)
    elif f: path = f.name
    else: return "No input."
    
    res = await rag.ingest(path)
    return res, await rag.check_database()

async def chat_wrapper(msg, h):
    rag = get_rag()
    ans, cits = await rag.query(msg)
    return f"{ans}\n\n📚 **Sources:**\n" + "\n".join([f"- {c}" for c in cits])

async def briefing_wrapper(): return await get_rag().generate_briefing()

with gr.Blocks(title="IMF Context RAG") as demo:
    gr.Markdown("# Context-Aware Financial RAG")
    stat = gr.Markdown("Checking...")
    demo.load(update_status, outputs=[stat])
    
    with gr.Tab("Chat"):
        with gr.Accordion("Ingest", open=False):
            with gr.Row():
                f_in = gr.File(); u_in = gr.Textbox()
            btn = gr.Button("Ingest"); out = gr.Textbox()
            btn.click(run_ingest, [f_in, u_in], [out, stat])
        gr.ChatInterface(fn=chat_wrapper, type="messages")
        
    with gr.Tab("Briefing"):
        b_btn = gr.Button("Generate Report"); b_out = gr.Markdown()
        b_btn.click(briefing_wrapper, outputs=[b_out])

if __name__ == "__main__":
    demo.launch()