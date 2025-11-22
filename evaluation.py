import asyncio
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

# --- EVALUATION METRICS ---
@dataclass
class EvaluationMetrics:
    """Stores evaluation results for a single query"""
    query_id: str
    query_text: str
    modality: str  # "text", "table", "chart"
    response_time: float
    num_sources: int
    response_length: int
    relevance_score: float  # 1-5 (manual or heuristic)
    completeness_score: float  # 1-5
    accuracy_score: float  # 1-5
    timestamp: str

@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    total_queries: int
    avg_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    avg_sources_retrieved: float
    avg_response_length: int
    avg_relevance_score: float
    avg_completeness_score: float
    avg_accuracy_score: float
    modality_breakdown: Dict[str, Dict[str, float]]
    total_duration: float

# --- TEST QUERIES (Document-Specific: qatar_test_doc.pdf) ---
TEST_QUERIES = {
    "text": [
        {
            "id": "text_001",
            "query": "What are the main economic challenges and context for Qatar mentioned in the report?",
            "expected_keywords": ["growth", "challenges", "context", "economic", "Qatar", "outlook", "fiscal", "external"],
            "modality": "text"
        },
        {
            "id": "text_002",
            "query": "Explain Qatar's medium-term outlook and the factors driving it",
            "expected_keywords": ["outlook", "growth", "factors", "medium-term", "expansion", "reforms", "LNG"],
            "modality": "text"
        },
        {
            "id": "text_003",
            "query": "What are the main risks to Qatar's economic outlook?",
            "expected_keywords": ["risks", "outlook", "economic", "challenges", "vulnerabilities", "downside"],
            "modality": "text"
        },
        {
            "id": "text_004",
            "query": "Describe the banking sector situation in Qatar",
            "expected_keywords": ["banking", "sector", "financial", "credit", "banks", "vulnerabilities"],
            "modality": "text"
        },
        {
            "id": "text_005",
            "query": "What structural reforms are being implemented in Qatar?",
            "expected_keywords": ["reforms", "structural", "policy", "implementation", "strategy", "development"],
            "modality": "text"
        }
    ],
    "table": [
        {
            "id": "table_001",
            "query": "What does Table 1 on Qatar's Selected Macroeconomic Indicators show for 2021-2025?",
            "expected_keywords": ["GDP", "inflation", "revenue", "expenditure", "exports", "imports", "2021", "2025"],
            "modality": "table"
        },
        {
            "id": "table_002",
            "query": "Analyze the production and prices indicators in the macroeconomic table",
            "expected_keywords": ["real GDP", "hydrocarbon", "nonhydrocarbon", "CPI", "inflation", "percent change"],
            "modality": "table"
        },
        {
            "id": "table_003",
            "query": "What are the public finance indicators shown in the table?",
            "expected_keywords": ["revenue", "expenditure", "current", "capital", "fiscal balance", "percent of GDP"],
            "modality": "table"
        },
        {
            "id": "table_004",
            "query": "Describe the external sector data in the macroeconomic indicators table",
            "expected_keywords": ["exports", "imports", "current account", "external debt", "reserves", "exchange rate"],
            "modality": "table"
        },
        {
            "id": "table_005",
            "query": "What trends do you observe in Qatar's fiscal balance from 2021 to 2025?",
            "expected_keywords": ["fiscal balance", "2021", "2022", "2023", "2024", "2025", "trend", "projection"],
            "modality": "table"
        }
    ],
    "chart": [
        {
            "id": "chart_001",
            "query": "What does Figure 1 show about Qatar's economic performance?",
            "expected_keywords": ["figure", "chart", "graph", "data", "trend", "growth", "GDP"],
            "modality": "chart"
        },
        {
            "id": "chart_002",
            "query": "Describe the visual representation of Qatar's fiscal position in the charts",
            "expected_keywords": ["fiscal", "chart", "graph", "revenue", "expenditure", "balance", "visual"],
            "modality": "chart"
        },
        {
            "id": "chart_003",
            "query": "What insights can be drawn from the graphical representation of external sector data?",
            "expected_keywords": ["external", "chart", "graph", "exports", "imports", "current account", "trend"],
            "modality": "chart"
        },
        {
            "id": "chart_004",
            "query": "Analyze the inflation trends shown in the charts",
            "expected_keywords": ["inflation", "CPI", "chart", "trend", "percent", "price", "graph"],
            "modality": "chart"
        },
        {
            "id": "chart_005",
            "query": "What does the chart show about Qatar's LNG production and hydrocarbon sector?",
            "expected_keywords": ["LNG", "hydrocarbon", "production", "chart", "expansion", "graph", "trend"],
            "modality": "chart"
        }
    ]
}

# --- EVALUATION CLASS ---
class RAGEvaluator:
    def __init__(self, rag_instance):
        """Initialize evaluator with RAG instance"""
        self.rag = rag_instance
        self.results: List[EvaluationMetrics] = []
        
    def _calculate_relevance(self, response: str, expected_keywords: List[str]) -> float:
        """
        Heuristic relevance scoring based on keyword presence.
        Returns score 1-5.
        More lenient for text queries, stricter for table/chart queries.
        """
        response_lower = response.lower()
        matched_keywords = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        
        # Calculate percentage of keywords matched
        keyword_percentage = matched_keywords / len(expected_keywords) if expected_keywords else 0
        
        # More lenient scoring - focus on semantic relevance
        if keyword_percentage == 0:
            return 1.0
        elif keyword_percentage < 0.2:  # Less than 20%
            return 2.0
        elif keyword_percentage < 0.4:  # 20-40%
            return 3.0
        elif keyword_percentage < 0.7:  # 40-70%
            return 4.0
        else:  # 70%+
            return 5.0
    
    def _calculate_completeness(self, response: str, num_sources: int) -> float:
        """
        Completeness based on response length and source count.
        Returns score 1-5.
        More balanced scoring between length and sources.
        """
        response_length = len(response)
        
        # Score based on response length (primary factor)
        if response_length < 150:
            length_score = 1.0
        elif response_length < 300:
            length_score = 2.0
        elif response_length < 600:
            length_score = 3.0
        elif response_length < 1000:
            length_score = 4.0
        else:
            length_score = 5.0
        
        # Score based on sources (secondary factor)
        if num_sources == 0:
            source_score = 1.0
        elif num_sources == 1:
            source_score = 2.0
        elif num_sources < 3:
            source_score = 3.0
        elif num_sources < 5:
            source_score = 4.0
        else:
            source_score = 5.0
        
        # Average the two scores
        return (length_score + source_score) / 2
    
    def _calculate_accuracy(self, response: str, modality: str) -> float:
        """
        Heuristic accuracy scoring based on response structure and modality.
        Returns score 1-5.
        """
        response_lower = response.lower()
        response_length = len(response)
        
        # Base score on response length (longer = more detailed)
        if response_length < 100:
            score = 1.0
        elif response_length < 300:
            score = 2.0
        elif response_length < 600:
            score = 3.0
        elif response_length < 1000:
            score = 4.0
        else:
            score = 4.5
        
        # Check for common quality indicators
        has_numbers = any(char.isdigit() for char in response)
        has_citations = "page" in response_lower or "source" in response_lower
        has_analysis = any(word in response_lower for word in ["analysis", "trend", "indicates", "shows", "demonstrates", "explain", "discuss", "describe"])
        has_structure = "\n" in response  # Multi-line response suggests structure
        
        # Add points for quality indicators
        if has_analysis:
            score += 0.5
        if has_citations:
            score += 0.3
        if has_structure:
            score += 0.2
        
        # Modality-specific checks
        if modality == "table" and ("table" in response_lower or "data" in response_lower or "indicator" in response_lower):
            score += 0.5
        elif modality == "chart" and ("figure" in response_lower or "chart" in response_lower or "graph" in response_lower):
            score += 0.5
        elif modality == "text" and ("explain" in response_lower or "discuss" in response_lower or "describe" in response_lower):
            score += 0.5
        
        return min(score, 5.0)
    
    async def evaluate_query(self, query_data: Dict[str, Any]) -> EvaluationMetrics:
        """Evaluate a single query"""
        query_id = query_data["id"]
        query_text = query_data["query"]
        modality = query_data["modality"]
        expected_keywords = query_data["expected_keywords"]
        
        # Measure response time
        start_time = time.time()
        try:
            response, sources = await self.rag.query(query_text)
        except Exception as e:
            print(f"❌ Error evaluating {query_id}: {str(e)}")
            response = ""
            sources = []
        
        response_time = time.time() - start_time
        
        # Calculate metrics
        relevance = self._calculate_relevance(response, expected_keywords)
        completeness = self._calculate_completeness(response, len(sources))
        accuracy = self._calculate_accuracy(response, modality)
        
        metrics = EvaluationMetrics(
            query_id=query_id,
            query_text=query_text,
            modality=modality,
            response_time=response_time,
            num_sources=len(sources),
            response_length=len(response),
            relevance_score=relevance,
            completeness_score=completeness,
            accuracy_score=accuracy,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(metrics)
        return metrics
    
    async def run_benchmark(self, test_queries: Dict[str, List[Dict]] = None) -> BenchmarkResults:
        """Run full benchmark across all modalities"""
        if test_queries is None:
            test_queries = TEST_QUERIES
        
        print("\n" + "="*80)
        print("STARTING RAG EVALUATION BENCHMARK")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        # Run all queries
        for modality, queries in test_queries.items():
            print(f"\nEvaluating {modality.upper()} queries ({len(queries)} queries)...")
            for query_data in queries:
                metrics = await self.evaluate_query(query_data)
                print(f"  [{metrics.query_id}] {metrics.response_time:.2f}s | "
                      f"Relevance: {metrics.relevance_score:.1f}/5 | "
                      f"Completeness: {metrics.completeness_score:.1f}/5 | "
                      f"Accuracy: {metrics.accuracy_score:.1f}/5")
        
        total_duration = time.time() - start_time
        
        # Aggregate results
        results = self._aggregate_results(total_duration)
        
        return results
    
    def _aggregate_results(self, total_duration: float) -> BenchmarkResults:
        """Aggregate individual query results into benchmark summary"""
        if not self.results:
            return None
        
        response_times = [r.response_time for r in self.results]
        sources_counts = [r.num_sources for r in self.results]
        response_lengths = [r.response_length for r in self.results]
        relevance_scores = [r.relevance_score for r in self.results]
        completeness_scores = [r.completeness_score for r in self.results]
        accuracy_scores = [r.accuracy_score for r in self.results]
        
        # Modality breakdown
        modality_breakdown = {}
        for modality in set(r.modality for r in self.results):
            modality_results = [r for r in self.results if r.modality == modality]
            modality_breakdown[modality] = {
                "count": len(modality_results),
                "avg_response_time": statistics.mean([r.response_time for r in modality_results]),
                "avg_relevance": statistics.mean([r.relevance_score for r in modality_results]),
                "avg_completeness": statistics.mean([r.completeness_score for r in modality_results]),
                "avg_accuracy": statistics.mean([r.accuracy_score for r in modality_results]),
                "avg_sources": statistics.mean([r.num_sources for r in modality_results]),
            }
        
        return BenchmarkResults(
            total_queries=len(self.results),
            avg_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            avg_sources_retrieved=statistics.mean(sources_counts),
            avg_response_length=int(statistics.mean(response_lengths)),
            avg_relevance_score=statistics.mean(relevance_scores),
            avg_completeness_score=statistics.mean(completeness_scores),
            avg_accuracy_score=statistics.mean(accuracy_scores),
            modality_breakdown=modality_breakdown,
            total_duration=total_duration
        )
    
    def print_results(self, results: BenchmarkResults):
        """Print formatted benchmark results"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80 + "\n")
        
        print(f"Total Queries Evaluated: {results.total_queries}")
        print(f"Total Duration: {results.total_duration:.2f}s\n")
        
        print("RESPONSE TIME METRICS:")
        print(f"  Average: {results.avg_response_time:.2f}s")
        print(f"  Median:  {results.median_response_time:.2f}s")
        print(f"  Min:     {results.min_response_time:.2f}s")
        print(f"  Max:     {results.max_response_time:.2f}s\n")
        
        print("RETRIEVAL METRICS:")
        print(f"  Avg Sources Retrieved: {results.avg_sources_retrieved:.1f}")
        print(f"  Avg Response Length: {results.avg_response_length} chars\n")
        
        print("QUALITY METRICS (1-5 scale):")
        print(f"  Avg Relevance Score:    {results.avg_relevance_score:.2f}/5.0")
        print(f"  Avg Completeness Score: {results.avg_completeness_score:.2f}/5.0")
        print(f"  Avg Accuracy Score:     {results.avg_accuracy_score:.2f}/5.0\n")
        
        print("MODALITY BREAKDOWN:")
        for modality, stats in results.modality_breakdown.items():
            print(f"\n  {modality.upper()}:")
            print(f"    Queries: {stats['count']}")
            print(f"    Avg Response Time: {stats['avg_response_time']:.2f}s")
            print(f"    Avg Relevance: {stats['avg_relevance']:.2f}/5.0")
            print(f"    Avg Completeness: {stats['avg_completeness']:.2f}/5.0")
            print(f"    Avg Accuracy: {stats['avg_accuracy']:.2f}/5.0")
            print(f"    Avg Sources: {stats['avg_sources']:.1f}")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, results: BenchmarkResults, filename: str = "benchmark_results.json"):
        """Save detailed results to JSON"""
        output = {
            "summary": asdict(results),
            "detailed_queries": [asdict(r) for r in self.results]
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filename}")

# --- MAIN EVALUATION RUNNER ---
async def run_evaluation(rag_instance, custom_queries: Dict = None):
    """Main function to run evaluation"""
    evaluator = RAGEvaluator(rag_instance)
    
    # Run benchmark
    results = await evaluator.run_benchmark(custom_queries or TEST_QUERIES)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    return evaluator, results

if __name__ == "__main__":
    print("Evaluation module loaded. Use with RAG instance:")
    print("  from evaluation import run_evaluation")
    print("  await run_evaluation(rag_instance)")
