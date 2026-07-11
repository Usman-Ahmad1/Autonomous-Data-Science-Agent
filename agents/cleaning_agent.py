# agents/cleaning_agent.py
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Dict, Any

# Try to import the advanced cleaning pipeline
try:
    from cleaning import CleaningPipeline
    ADVANCED_CLEANING_AVAILABLE = True
except ImportError:
    try:
        from src.cleaning import CleaningPipeline
        ADVANCED_CLEANING_AVAILABLE = True
    except ImportError:
        ADVANCED_CLEANING_AVAILABLE = False
        # Fallback - define inline if imports fail
        class CleaningPipeline:
            def clean(self, df, **kwargs):
                return df
            def get_report(self):
                return {"error": "Cleaning module not available"}

def cleaning_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleaning Agent - Applies advanced cleaning if available, otherwise passes through
    """
    try:
        dataset = state.get("dataset")
        
        if dataset is None:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "No dataset provided."}],
                "dataset": None,
                "cleaned_dataset": None,
                "next_agent": "EDA_Agent"
            }
        
        # Check if cleaning report already exists (from UI manual dropping)
        existing_report = state.get("cleaning_report", {})
        columns_dropped = existing_report.get("columns_dropped", [])
        
        # If cleaning already done in UI (manual column dropping), use that data
        if state.get("cleaned_dataset") is not None and columns_dropped:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": f"✅ Using manually cleaned dataset: {dataset.shape[0]} rows × {dataset.shape[1]} columns"}],
                "dataset": dataset,
                "cleaned_dataset": dataset,
                "next_agent": "EDA_Agent",
                "cleaning_report": {
                    "status": "Manual column dropping applied",
                    "columns_dropped": columns_dropped,
                    "dataset_shape": dataset.shape
                }
            }
        
        # If advanced cleaning is available and not already applied
        if ADVANCED_CLEANING_AVAILABLE:
            # Initialize cleaning pipeline
            pipeline = CleaningPipeline()
            
            # Run cleaning with smart defaults
            cleaned_df = pipeline.clean(
                dataset,
                missing_strategy='median',      # Default: use median
                outlier_method='iqr',           # Default: IQR method
                outlier_handling='cap',         # Default: cap outliers
                remove_duplicates=True,
                remove_duplicate_cols=True
            )
            
            # Get cleaning report
            report = pipeline.get_report()
            
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": f"✅ Advanced cleaning complete! {dataset.shape} → {cleaned_df.shape}"}],
                "dataset": dataset,
                "cleaned_dataset": cleaned_df,
                "next_agent": "EDA_Agent",
                "cleaning_report": report
            }
        else:
            # Fallback: simple pass-through (original behavior)
            cleaning_report = {
                "status": "No advanced cleaning available - using original dataset",
                "dataset_shape": dataset.shape
            }
            
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": f"✅ Dataset ready: {dataset.shape[0]} rows × {dataset.shape[1]} columns"}],
                "dataset": dataset,
                "cleaned_dataset": dataset,
                "next_agent": "EDA_Agent",
                "cleaning_report": cleaning_report
            }
        
    except Exception as e:
        # On error, just pass through (graceful degradation)
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": f"⚠️ Cleaning skipped: {str(e)}"}],
            "dataset": state.get("dataset"),
            "cleaned_dataset": state.get("dataset"),
            "next_agent": "EDA_Agent",
            "cleaning_report": {
                "status": "Cleaning skipped due to error",
                "error": str(e)
            }
        }