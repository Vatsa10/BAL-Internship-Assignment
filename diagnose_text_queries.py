#!/usr/bin/env python3
"""
Diagnostic script to analyze text query performance
Shows what keywords are being matched and why scores are low
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import AsyncDoclingRAG
from evaluation import TEST_QUERIES

async def diagnose():
    """Diagnose text query performance"""
    
    print("\n" + "="*80)
    print("TEXT QUERY DIAGNOSTIC")
    print("="*80 + "\n")
    
    rag = AsyncDoclingRAG()
    
    # Check database
    db_status = await rag.check_database()
    print(f"{db_status}\n")
    
    if "Not Found" in db_status or "Empty" in db_status:
        print("❌ Database is empty. Please ingest PDF first.")
        return
    
    # Test each text query
    text_queries = TEST_QUERIES["text"]
    
    for query_data in text_queries:
        query_id = query_data["id"]
        query_text = query_data["query"]
        expected_keywords = query_data["expected_keywords"]
        
        print(f"\n{'='*80}")
        print(f"Query: {query_id}")
        print(f"{'='*80}")
        print(f"Question: {query_text}\n")
        print(f"Expected Keywords: {expected_keywords}\n")
        
        # Get response
        response, sources = await rag.query(query_text)
        
        print(f"Response Length: {len(response)} characters")
        print(f"Sources Retrieved: {len(sources)}\n")
        
        # Check keyword matches
        response_lower = response.lower()
        matched = []
        not_matched = []
        
        for kw in expected_keywords:
            if kw.lower() in response_lower:
                matched.append(kw)
            else:
                not_matched.append(kw)
        
        print(f"Matched Keywords ({len(matched)}/{len(expected_keywords)}):")
        for kw in matched:
            print(f"   - {kw}")
        
        if not_matched:
            print(f"\nMissing Keywords ({len(not_matched)}/{len(expected_keywords)}):")
            for kw in not_matched:
                print(f"   - {kw}")
        
        # Calculate scores
        keyword_percentage = len(matched) / len(expected_keywords) if expected_keywords else 0
        
        if keyword_percentage < 0.2:
            relevance = 2.0
        elif keyword_percentage < 0.4:
            relevance = 3.0
        elif keyword_percentage < 0.7:
            relevance = 4.0
        else:
            relevance = 5.0
        
        print(f"\nScores:")
        print(f"   Keyword Match Rate: {keyword_percentage*100:.1f}%")
        print(f"   Relevance Score: {relevance:.1f}/5.0")
        
        # Show first 500 chars of response
        print(f"\nResponse Preview:")
        print(f"   {response[:500]}...")
        
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("""
1. If keyword match rate is low:
   - Update expected_keywords in evaluation.py to match actual response content
   - Use more generic keywords that appear in financial analysis
   - Focus on semantic meaning rather than exact phrases

2. If response length is short:
   - Increase top_k in main.py query method
   - Improve prompt to encourage longer responses
   - Check if PDF was ingested correctly

3. If sources are low:
   - Verify hybrid search is working
   - Check Qdrant database configuration
   - Ensure vectors are properly indexed

4. General improvements:
   - Run: python evaluate_rag.py
   - Review benchmark_results.json
   - Adjust RAG configuration based on results
    """)

if __name__ == "__main__":
    asyncio.run(diagnose())
