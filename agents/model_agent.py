import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

try:
    from src.model_selection import ModelSelector, ModelEvaluator, ModelComparator
    MODEL_SELECTION_AVAILABLE = True
except ImportError:
    MODEL_SELECTION_AVAILABLE = False
    print("⚠️ Model Selection module not available")

def model_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Model Selection Agent - Automatically selects the best model
    """
    try:
        dataset = state.get("dataset")
        target_column = state.get("target_column")
        
        if dataset is None:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "❌ No dataset provided for model selection."}],
                "dataset": None,
                "cleaned_dataset": None,
                "next_agent": "EDA_Agent"
            }
        
        if target_column is None or target_column not in dataset.columns:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "⚠️ Please select a target column for model selection."}],
                "dataset": dataset,
                "cleaned_dataset": dataset,
                "next_agent": "EDA_Agent"
            }
        
        if not MODEL_SELECTION_AVAILABLE:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "⚠️ Model selection module not available. Skipping."}],
                "dataset": dataset,
                "cleaned_dataset": dataset,
                "next_agent": "EDA_Agent"
            }
        
        print("=" * 60)
        print("🧪 MODEL SELECTION AGENT STARTED")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Target column: {target_column}")
        print("=" * 60)
        
        # Prepare features and target
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        # Initialize model selector
        selector = ModelSelector(problem_type='auto')
        
        # Select models
        results = selector.select_models(X, y, test_size=0.2, cv_folds=5)
        
        # Get results DataFrame
        results_df = selector.get_model_results()
        
        # Get best model info
        best_model_name = selector.get_best_model_name()
        best_model = selector.get_best_model()
        
        # Generate report
        report = []
        report.append("🧪 **MODEL SELECTION REPORT**")
        report.append("")
        report.append("---")
        report.append(f"**Problem Type:** {selector.problem_type.upper()}")
        report.append(f"**Target Column:** {target_column}")
        report.append(f"**Total Models Tested:** {len(results)}")
        report.append("")
        report.append("---")
        report.append("**📊 Model Performance Comparison:**")
        report.append("")
        report.append("| Model | CV Score | Test Score | Training Time (s) |")
        report.append("|-------|----------|------------|-------------------|")
        
        for _, row in results_df.iterrows():
            report.append(f"| {row['Model']} | {row['CV Mean']:.4f} | {row['Test Score']:.4f} | {row['Training Time (s)']:.2f} |")
        
        report.append("")
        report.append("---")
        report.append(f"🏆 **Best Model: {best_model_name}**")
        report.append(f"   Test Score: {results[best_model_name]['test_score']:.4f}")
        report.append("")
        report.append("✅ **Model Selection Complete!**")
        
        formatted_report = "\n".join(report)
        
        return {
            "messages": state["messages"] + [
                {"role": "assistant", "content": "🧪 Model Selection Complete!"},
                {"role": "assistant", "content": formatted_report}
            ],
            "dataset": dataset,
            "cleaned_dataset": dataset,
            "next_agent": "EDA_Agent",
            "model_report": formatted_report,
            "model_results": results,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "model_complete": True
        }
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Model selection error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
            "dataset": state.get("dataset"),
            "cleaned_dataset": state.get("dataset"),
            "next_agent": "EDA_Agent"
        }