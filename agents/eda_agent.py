import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def eda_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    EDA Agent - Performs Exploratory Data Analysis on the dataset
    Returns a clean, readable report without visualizations
    """
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("📊 EDA AGENT STARTED")
        print("=" * 60)
        
        # Get dataset
        dataset = state.get("cleaned_dataset")
        if dataset is None:
            dataset = state.get("dataset")
        
        print(f"Dataset shape: {dataset.shape if dataset is not None else 'None'}")
        
        if dataset is None:
            print("❌ No dataset provided")
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "❌ No dataset provided for EDA."}],
                "dataset": None,
                "cleaned_dataset": None,
                "next_agent": "EDA_Agent",
                "eda_report": "❌ No dataset provided for EDA.",
                "eda_plot": None,
                "eda_complete": True
            }
        
        print(f"✅ Dataset loaded: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        
        # ============ GENERATE READABLE EDA REPORT ============
        eda_report = []
        eda_report.append("📊 **EDA REPORT**")
        eda_report.append("")
        
        # 1. Dataset Overview - Clean and simple
        eda_report.append("---")
        eda_report.append("**📋 1. DATASET OVERVIEW**")
        eda_report.append("")
        eda_report.append(f"| Metric | Value |")
        eda_report.append(f"|--------|-------|")
        eda_report.append(f"| Total Rows | {dataset.shape[0]:,} |")
        eda_report.append(f"| Total Columns | {dataset.shape[1]} |")
        eda_report.append(f"| Total Cells | {dataset.shape[0] * dataset.shape[1]:,} |")
        eda_report.append("")
        
        # 2. Column Information - As a table
        eda_report.append("---")
        eda_report.append("**📋 2. COLUMN INFORMATION**")
        eda_report.append("")
        eda_report.append("| Column Name | Data Type | Missing Values | Missing % | Unique Values |")
        eda_report.append("|-------------|-----------|----------------|-----------|---------------|")
        
        for col in dataset.columns:
            dtype = dataset[col].dtype
            null_count = dataset[col].isnull().sum()
            null_pct = (null_count / len(dataset)) * 100
            unique_count = dataset[col].nunique()
            eda_report.append(f"| {col} | {dtype} | {null_count:,} | {null_pct:.1f}% | {unique_count:,} |")
        eda_report.append("")
        
        # 3. Numeric Columns Statistics - As a table
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            eda_report.append("---")
            eda_report.append("**📊 3. NUMERIC COLUMN STATISTICS**")
            eda_report.append("")
            eda_report.append("| Column | Count | Mean | Median | Min | Max | Std Dev |")
            eda_report.append("|--------|-------|------|--------|-----|-----|---------|")
            
            for col in numeric_cols:
                stats = dataset[col].describe()
                eda_report.append(f"| {col} | {stats['count']:.0f} | {stats['mean']:.2f} | {stats['50%']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['std']:.2f} |")
            eda_report.append("")
        
        # 4. Missing Values Summary
        missing_cols = dataset.columns[dataset.isnull().any()].tolist()
        eda_report.append("---")
        eda_report.append("**🔍 4. MISSING VALUES**")
        eda_report.append("")
        
        if missing_cols:
            eda_report.append("| Column | Missing Count | Missing % |")
            eda_report.append("|--------|---------------|-----------|")
            for col in missing_cols:
                count = dataset[col].isnull().sum()
                pct = (count / len(dataset)) * 100
                eda_report.append(f"| {col} | {count:,} | {pct:.1f}% |")
        else:
            eda_report.append("✅ **No missing values found in the dataset!**")
        eda_report.append("")
        
        # 5. Dataset Quality Summary
        eda_report.append("---")
        eda_report.append("**📈 5. DATASET QUALITY SUMMARY**")
        eda_report.append("")
        total_missing = dataset.isnull().sum().sum()
        total_cells = dataset.shape[0] * dataset.shape[1]
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        complete_rows = dataset.dropna().shape[0]
        
        eda_report.append("| Quality Metric | Value |")
        eda_report.append("|----------------|-------|")
        eda_report.append(f"| Total Missing Values | {total_missing:,} ({missing_pct:.2f}% of all cells) |")
        eda_report.append(f"| Complete Rows | {complete_rows:,} ({complete_rows/len(dataset)*100:.1f}%) |")
        eda_report.append(f"| Columns with Missing Values | {len(missing_cols)} |")
        
        # 6. Data Types Summary
        eda_report.append("")
        eda_report.append("---")
        eda_report.append("**📝 6. DATA TYPES SUMMARY**")
        eda_report.append("")
        dtype_counts = dataset.dtypes.value_counts()
        eda_report.append("| Data Type | Count |")
        eda_report.append("|-----------|-------|")
        for dtype, count in dtype_counts.items():
            eda_report.append(f"| {dtype} | {count} |")
        
        eda_report.append("")
        eda_report.append("---")
        eda_report.append("✅ **EDA Complete!**")
        
        formatted_report = "\n".join(eda_report)
        print("✅ EDA Report generated successfully")
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ EDA completed in {elapsed_time:.2f} seconds")
        
        # ============ RETURN WITHOUT VISUALIZATIONS ============
        return {
            "messages": state["messages"] + [
                {"role": "assistant", "content": "📊 EDA Analysis Complete!"},
                {"role": "assistant", "content": formatted_report}
            ],
            "dataset": dataset,
            "cleaned_dataset": dataset,
            "next_agent": "EDA_Agent",
            "eda_report": formatted_report,
            "eda_plot": None,  # No visualizations
            "eda_complete": True,
            "eda_status": "success"
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"❌ EDA Error after {elapsed_time:.2f}s: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": f"❌ EDA failed: {str(e)}"}],
            "dataset": state.get("dataset"),
            "cleaned_dataset": state.get("cleaned_dataset"),
            "next_agent": "EDA_Agent",
            "eda_report": f"❌ EDA failed: {str(e)}",
            "eda_plot": None,
            "eda_complete": True,
            "eda_status": "error"
        }