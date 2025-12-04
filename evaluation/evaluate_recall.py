"""
Evaluation script demonstrating recall improvements.
Compares baseline vs improved retrieval on test queries.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.baseline_retriever import BaselineRetriever
from retrieval.improved_retriever import ImprovedRetriever


# Test queries with expected relevant documents
TEST_QUERIES = [
    {
        "query": "Snorkeling near Lisbon",
        "expected_docs": ["Lisbon"],  # Should find Lisbon destination
        "expected_activities": ["snorkeling"]
    },
    {
        "query": "Wine tasting in Tuscany",
        "expected_docs": ["Tuscany", "Marco Rossi"],  # Should find Tuscany and wine guide
        "expected_activities": ["wine tasting"]
    },
    {
        "query": "City tours in Paris",
        "expected_docs": ["Paris", "Jean-Pierre Dubois"],  # Should find Paris and guide
        "expected_activities": ["city tours"]
    },
    {
        "query": "Hiking in Iceland",
        "expected_docs": ["Iceland", "Sarah Johnson"],  # Should find Iceland and guide
        "expected_activities": ["hiking"]
    },
    {
        "query": "Culinary tours in Tokyo",
        "expected_docs": ["Tokyo", "Yuki Tanaka"],  # Should find Tokyo and guide
        "expected_activities": ["culinary tours"]
    },
    {
        "query": "Beaches and snorkeling",
        "expected_docs": ["Lisbon", "Santorini", "Bali"],  # Multiple destinations
        "expected_activities": ["beaches", "snorkeling"]
    },
    {
        "query": "Photography tours",
        "expected_docs": ["Santorini", "Iceland", "Kyoto"],  # Multiple destinations
        "expected_activities": ["photography tours"]
    },
    {
        "query": "Historical tours in Japan",
        "expected_docs": ["Tokyo", "Kyoto", "Kenji Yamamoto"],  # Multiple Japan destinations
        "expected_activities": ["historical tours"]
    }
]


def evaluate_retrieval(retriever, query: str, expected_docs: List[str], top_k: int = 10) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query.
    
    Returns:
        Dict with precision, recall, and retrieved documents
    """
    if isinstance(retriever, ImprovedRetriever):
        result = retriever.search(query, limit=top_k)
        retrieved = result["results"]
        rewritten_query = result.get("rewritten_query")
    else:
        retrieved = retriever.search(query, limit=top_k)
        rewritten_query = None
    
    # Extract document names
    retrieved_names = [doc["name"] for doc in retrieved]
    
    # Calculate metrics
    relevant_retrieved = sum(1 for name in retrieved_names if any(exp in name for exp in expected_docs))
    recall = relevant_retrieved / len(expected_docs) if expected_docs else 0
    precision = relevant_retrieved / len(retrieved_names) if retrieved_names else 0
    
    result_dict = {
        "query": query,
        "expected_docs": expected_docs,
        "retrieved_names": retrieved_names,
        "relevant_retrieved": relevant_retrieved,
        "recall": recall,
        "precision": precision,
        "num_retrieved": len(retrieved_names)
    }
    
    if rewritten_query:
        result_dict["rewritten_query"] = rewritten_query
    
    return result_dict


def run_evaluation():
    """Run full evaluation comparing baseline vs improved."""
    index_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes")
    baseline_path = os.path.join(index_dir, "baseline")
    improved_path = os.path.join(index_dir, "improved")
    
    if not os.path.exists(baseline_path):
        print("ERROR: Baseline index not found. Please run indexing/index_builder.py first.")
        return
    
    if not os.path.exists(improved_path):
        print("ERROR: Improved index not found. Please run indexing/index_builder.py first.")
        return
    
    # Initialize retrievers
    print("Initializing retrievers...")
    baseline_retriever = BaselineRetriever(baseline_path)
    improved_retriever = ImprovedRetriever(improved_path)
    
    print("\n" + "="*80)
    print("EVALUATION: Baseline vs Improved Retrieval")
    print("="*80)
    
    baseline_results = []
    improved_results = []
    
    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected_docs = test_case["expected_docs"]
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Expected documents: {expected_docs}")
        print(f"{'='*80}")
        
        # Baseline evaluation
        baseline_result = evaluate_retrieval(baseline_retriever, query, expected_docs)
        baseline_results.append(baseline_result)
        
        print(f"\n📊 BASELINE RESULTS:")
        print(f"  Retrieved: {baseline_result['retrieved_names'][:5]}")
        print(f"  Relevant Retrieved: {baseline_result['relevant_retrieved']}/{len(expected_docs)}")
        print(f"  Recall: {baseline_result['recall']:.2%}")
        print(f"  Precision: {baseline_result['precision']:.2%}")
        
        # Improved evaluation
        improved_result = evaluate_retrieval(improved_retriever, query, expected_docs)
        improved_results.append(improved_result)
        
        print(f"\n✨ IMPROVED RESULTS:")
        if improved_result.get('rewritten_query'):
            rq = improved_result['rewritten_query']
            if isinstance(rq, dict):
                print(f"  Rewritten Query:")
                print(f"    City: {rq.get('city', 'None')}")
                print(f"    Country: {rq.get('country', 'None')}")
                print(f"    Activities: {rq.get('activities', [])}")
        print(f"  Retrieved: {improved_result['retrieved_names'][:5]}")
        print(f"  Relevant Retrieved: {improved_result['relevant_retrieved']}/{len(expected_docs)}")
        print(f"  Recall: {improved_result['recall']:.2%}")
        print(f"  Precision: {improved_result['precision']:.2%}")
        
        # Improvement
        recall_improvement = improved_result['recall'] - baseline_result['recall']
        print(f"\n📈 IMPROVEMENT:")
        print(f"  Recall Improvement: {recall_improvement:+.2%}")
    
    # Summary statistics
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    baseline_avg_recall = sum(r['recall'] for r in baseline_results) / len(baseline_results)
    improved_avg_recall = sum(r['recall'] for r in improved_results) / len(improved_results)
    baseline_avg_precision = sum(r['precision'] for r in baseline_results) / len(baseline_results)
    improved_avg_precision = sum(r['precision'] for r in improved_results) / len(improved_results)
    
    print(f"\n📊 BASELINE AVERAGES:")
    print(f"  Average Recall: {baseline_avg_recall:.2%}")
    print(f"  Average Precision: {baseline_avg_precision:.2%}")
    
    print(f"\n✨ IMPROVED AVERAGES:")
    print(f"  Average Recall: {improved_avg_recall:.2%}")
    print(f"  Average Precision: {improved_avg_precision:.2%}")
    
    print(f"\n📈 OVERALL IMPROVEMENT:")
    print(f"  Recall Improvement: {improved_avg_recall - baseline_avg_recall:+.2%}")
    print(f"  Precision Improvement: {improved_avg_precision - baseline_avg_precision:+.2%}")
    
    # Save results
    results_dir = os.path.dirname(__file__)
    results_path = os.path.join(results_dir, "evaluation_results.json")
    
    with open(results_path, 'w') as f:
        json.dump({
            "baseline_results": baseline_results,
            "improved_results": improved_results,
            "summary": {
                "baseline_avg_recall": baseline_avg_recall,
                "improved_avg_recall": improved_avg_recall,
                "baseline_avg_precision": baseline_avg_precision,
                "improved_avg_precision": improved_avg_precision,
                "recall_improvement": improved_avg_recall - baseline_avg_recall,
                "precision_improvement": improved_avg_precision - baseline_avg_precision
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    # Cleanup
    baseline_retriever.close()
    improved_retriever.close()


if __name__ == "__main__":
    run_evaluation()

