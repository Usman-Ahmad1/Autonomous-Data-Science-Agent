import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Dict, Any

try:
    from src.feature_engineering import FeatureEngineeringAgent
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False
    print("⚠️ Feature Engineering module not available")

def feature_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Feature Engineering Agent - Creates and transforms features
    """
    try:
        dataset = state.get("dataset")
        
        if dataset is None:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "❌ No dataset provided for feature engineering."}],
                "dataset": None,
                "cleaned_dataset": None,
                "next_agent": "EDA_Agent"
            }
        
        if not FEATURE_ENG_AVAILABLE:
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "⚠️ Feature engineering not available. Skipping."}],
                "dataset": dataset,
                "cleaned_dataset": dataset,
                "next_agent": "EDA_Agent"
            }
        
        print("=" * 60)
        print("🔧 FEATURE AGENT STARTED")
        print(f"Dataset shape: {dataset.shape}")
        print("=" * 60)
        
        # Initialize feature engineering agent
        feature_eng = FeatureEngineeringAgent()
        
        # Apply feature engineering
        engineered_df = feature_eng.engineer_features(
            dataset,
            create_interactions=True,
            create_ratios=False,  # Can be enabled if needed
            create_polynomial=False,  # Can be enabled if needed
            create_bins=False,  # Can be enabled if needed
            scale_features=True,
            scale_method='standard',
            encode_categorical=True,
            encoding_method='label',
            log_transform=False,
            select_features=False  # Set to True if target column is provided
        )
        
        # Get report
        report = feature_eng.get_report()
        
        return {
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"✅ Feature engineering complete! {dataset.shape} → {engineered_df.shape}"}
            ],
            "dataset": dataset,
            "cleaned_dataset": engineered_df,
            "next_agent": "EDA_Agent",
            "feature_report": report
        }
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Feature engineering error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
            "dataset": state.get("dataset"),
            "cleaned_dataset": state.get("dataset"),
            "next_agent": "EDA_Agent"
        }