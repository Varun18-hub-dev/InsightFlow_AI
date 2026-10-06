#!/usr/bin/env python
"""
InsightFlow AI — RAG Evaluation Runner

Usage:
    python run_evaluation.py [--api-url http://localhost:8000] [--token YOUR_JWT_TOKEN]
    python run_evaluation.py --help
"""

import argparse
import json
import os
import sys

from evaluators.retrieval_evaluator import RetrievalEvaluator
from evaluators.generation_evaluator import GenerationEvaluator

def main():
    parser = argparse.ArgumentParser(description="InsightFlow AI Evaluation Runner")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend API URL")
    parser.add_argument("--token", default="", help="JWT Token for auth")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--dataset", default="datasets/sample_eval.json", help="Path to evaluation dataset")
    parser.add_argument("--mlflow-uri", default="", help="MLflow Tracking URI (optional)")
    parser.add_argument("--experiment-name", default="RAG_Evaluation", help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # 1. Load evaluation dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)
        
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
        
    # 2. Setup Evaluators
    retrieval_evaluator = RetrievalEvaluator(args.api_url, args.token)
    generation_evaluator = GenerationEvaluator(args.api_url, args.token)
    
    # 3. & 4. Run Evaluators
    print("Starting evaluation...")
    retrieval_results = retrieval_evaluator.evaluate(dataset)
    generation_results = generation_evaluator.evaluate(dataset)
    
    # 5. Aggregate metrics
    all_results = {
        "retrieval": retrieval_results,
        "generation": generation_results
    }
    
    # 6. Print results table
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print("--- Retrieval Metrics ---")
    for k, v in retrieval_results.items():
        print(f"{k:20}: {v:.4f}")
        
    print("\n--- Generation Metrics ---")
    for k, v in generation_results.items():
        print(f"{k:20}: {v:.4f}")
    print("="*40)
    
    # 7. Save to JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # 8. Log to MLflow
    if args.mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment_name)
            
            with mlflow.start_run():
                mlflow.log_param("dataset", args.dataset)
                for category, metrics in all_results.items():
                    for name, value in metrics.items():
                        mlflow.log_metric(f"{category}_{name.replace(' ', '_').lower()}", value)
            print(f"Logged results to MLflow at {args.mlflow_uri}")
        except ImportError:
            print("MLflow not installed. Skipping MLflow logging.")
        except Exception as e:
            print(f"Failed to log to MLflow: {e}")

if __name__ == "__main__":
    main()
