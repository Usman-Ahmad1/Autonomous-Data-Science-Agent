import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List, Optional

class OutlierHandler:
    """Comprehensive outlier detection and handling"""
    
    DETECTION_METHODS = {
        'zscore': 'Z-Score method (requires normal distribution)',
        'iqr': 'IQR (Interquartile Range) method',
        'isolation_forest': 'Isolation Forest (works with high dimensions)',
    }
    
    HANDLING_METHODS = {
        'cap': 'Cap/winsorize outliers at thresholds',
        'remove': 'Remove rows with outliers',
        'transform': 'Transform to reduce impact (log, sqrt)'
    }
    
    def __init__(self):
        self.outlier_report = {}
    
    def detect_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       threshold: float = 1.5) -> Dict[str, List]:
        """Detect outliers in numerical columns"""
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = {}
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
                
            data = df[col].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = data.index[z_scores > 3].tolist()
                outlier_indices[col] = outliers
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data.index[(data < lower_bound) | (data > upper_bound)].tolist()
                outlier_indices[col] = outliers
                
            elif method == 'isolation_forest':
                if len(data) > 10:
                    X = data.values.reshape(-1, 1)
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    preds = iso_forest.fit_predict(X)
                    outliers = data.index[preds == -1].tolist()
                    outlier_indices[col] = outliers
        
        self.outlier_report['detected'] = outlier_indices
        return outlier_indices
    
    def handle_outliers(self, df: pd.DataFrame,
                       method: str = 'cap',
                       columns: Optional[List[str]] = None,
                       detection_method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in numerical columns"""
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.copy()
        outliers_count = 0
        
        for col in columns:
            if col not in df_clean.columns or df_clean[col].dtype not in ['int64', 'float64']:
                continue
                
            data = df_clean[col].dropna()
            if len(data) == 0:
                continue
            
            # Detect outliers
            if detection_method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_idx = data.index[z_scores > 3].tolist()
                
            elif detection_method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_idx = data.index[(data < lower_bound) | (data > upper_bound)].tolist()
            else:
                outlier_idx = []
            
            if not outlier_idx:
                continue
                
            outliers_count += len(outlier_idx)
            
            # Handle outliers
            if method == 'cap':
                if detection_method == 'iqr':
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                elif detection_method == 'zscore':
                    mean = data.mean()
                    std = data.std()
                    df_clean.loc[df_clean[col] < mean - 3*std, col] = mean - 3*std
                    df_clean.loc[df_clean[col] > mean + 3*std, col] = mean + 3*std
                    
            elif method == 'remove':
                df_clean = df_clean.drop(index=outlier_idx)
                
            elif method == 'transform':
                if (df_clean[col] >= 0).all():
                    df_clean[col] = np.log1p(df_clean[col])
                else:
                    df_clean[col] = np.sqrt(df_clean[col] - df_clean[col].min() + 1)
        
        self.outlier_report['handled'] = {
            'method': method,
            'outliers_handled': outliers_count,
            'columns_processed': columns
        }
        
        return df_clean
    
    def get_report(self) -> Dict[str, Any]:
        return self.outlier_report