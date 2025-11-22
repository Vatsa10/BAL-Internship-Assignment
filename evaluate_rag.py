#!/usr/bin/env python3
"""
Complete RAG Evaluation Pipeline
Runs evaluation across all modalities and saves results to benchmark_results.json
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import AsyncDoclingRAG
from evaluation import run_evaluation, TEST_QUERIES

async def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*80)
    print("RAG EVALUATION PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Check database
    print("Step 1: Checking Database Status...")
    rag = AsyncDoclingRAG()
    db_status = await rag.check_database()
    print(f"{db_status}\n")
    
    if "Not Found" in db_status or "Empty" in db_status:
        print("Database is empty. Please ingest qatar_test_doc.pdf first.\n")
        print("To ingest:")
        print("  1. Run: python main.py")
        print("  2. Upload qatar_test_doc.pdf in the Ingest tab")
        print("  3. Then run this script again\n")
        return False
    
    # Step 2: Run evaluation
    print("Step 2: Running Evaluation Suite...")
    print(f"   Testing {len(TEST_QUERIES['text'])} text queries")
    print(f"   Testing {len(TEST_QUERIES['table'])} table queries")
    print(f"   Testing {len(TEST_QUERIES['chart'])} chart queries\n")
    
    evaluator, results = await run_evaluation(rag)
    
    # Step 3: Summary
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    print("Output Files:")
    print("   - benchmark_results.json - Raw evaluation data\n")
    
    print("Quick Summary:")
    print(f"   Total Queries: {results.total_queries}")
    print(f"   Avg Response Time: {results.avg_response_time:.2f}s")
    print(f"   Avg Relevance: {results.avg_relevance_score:.2f}/5.0")
    print(f"   Avg Completeness: {results.avg_completeness_score:.2f}/5.0")
    print(f"   Avg Accuracy: {results.avg_accuracy_score:.2f}/5.0\n")
    
    print("Modality Performance:")
    for modality, stats in results.modality_breakdown.items():
        print(f"   {modality.upper()}:")
        print(f"      Queries: {stats['count']}")
        print(f"      Avg Time: {stats['avg_response_time']:.2f}s")
        print(f"      Avg Score: {(stats['avg_relevance'] + stats['avg_completeness'] + stats['avg_accuracy'])/3:.2f}/5.0")
    
    print("\nNext Steps:")
    print("   1. Review benchmark_results.json for detailed metrics")
    print("   2. Run diagnose_text_queries.py for detailed text query analysis")
    print("   3. Identify optimization opportunities")
    print("   4. Iterate on RAG configuration\n")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
