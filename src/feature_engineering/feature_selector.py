import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Optional, Tuple

class FeatureSelector:
    """
    Selects best features using various techniques
    """
    
    def __init__(self):
        self.selection_report = {}
        self.selected_features = []
    
    def select_by_correlation(self, df: pd.DataFrame,
                             target_column: str,
                             threshold: float = 0.1) -> List[str]:
        """
        Select features based on correlation with target
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column not in num_cols:
            return []
        
        correlations = df[num_cols].corr()[target_column].abs()
        selected = correlations[correlations > threshold].index.tolist()
        
        # Remove target column itself
        if target_column in selected:
            selected.remove(target_column)
        
        self.selection_report['correlation'] = {
            'features_selected': len(selected),
            'selected_features': selected,
            'threshold': threshold
        }
        
        self.selected_features.extend(selected)
        return selected
    
    def select_by_mutual_info(self, df: pd.DataFrame,
                              target_column: str,
                              k: int = 10) -> List[str]:
        """
        Select features using mutual information
        """
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
        y = df[target_column]
        
        if X.empty or y.empty:
            return []
        
        k = min(k, X.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected = X.columns[selected_indices].tolist()
        
        self.selection_report['mutual_info'] = {
            'features_selected': len(selected),
            'selected_features': selected,
            'k': k
        }
        
        self.selected_features.extend(selected)
        return selected
    
    def select_by_rfe(self, df: pd.DataFrame,
                      target_column: str,
                      n_features: int = 10) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        """
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
        y = df[target_column]
        
        if X.empty or y.empty:
            return []
        
        n_features = min(n_features, X.shape[1])
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        selected = X.columns[rfe.support_].tolist()
        
        self.selection_report['rfe'] = {
            'features_selected': len(selected),
            'selected_features': selected,
            'n_features': n_features
        }
        
        self.selected_features.extend(selected)
        return selected
    
    def get_report(self) -> Dict[str, Any]:
        return self.selection_report