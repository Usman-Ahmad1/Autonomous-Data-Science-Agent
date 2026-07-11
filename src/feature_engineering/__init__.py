"""
Feature Engineering Module
Provides advanced feature engineering capabilities:
- Feature creation (interactions, polynomials, ratios)
- Feature transformation (scaling, encoding, binning)
- Feature selection (correlation, importance, RFE)
- Automated feature engineering
"""

from .feature_creator import FeatureCreator
from .feature_transformer import FeatureTransformer
from .feature_selector import FeatureSelector
from .feature_engineering_agent import FeatureEngineeringAgent

__all__ = [
    'FeatureCreator',
    'FeatureTransformer', 
    'FeatureSelector',
    'FeatureEngineeringAgent'
]