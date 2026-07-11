import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MissingValueHandler:
    """Advanced missing value handling with multiple strategies"""
    
    STRATEGIES = {
        'mean': 'Replace with column mean (numerical only)',
        'median': 'Replace with column median (numerical only)',
        'mode': 'Replace with column mode (categorical/numerical)',
        'knn': 'K-Nearest Neighbors imputation (numerical)',
        'forward_fill': 'Forward fill (time series)',
        'backward_fill': 'Backward fill (time series)',
        'interpolate': 'Linear interpolation (time series)',
        'constant': 'Replace with constant value',
        'drop_rows': 'Drop rows with missing values',
        'drop_columns': 'Drop columns with > threshold missing'
    }
    
    def __init__(self):
        self.imputation_report = {}
    
    def analyze_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing values in dataset"""
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values,
            'Data_Type': df.dtypes.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        return missing_df
    
    def handle_missing(self, df: pd.DataFrame, strategy: str, 
                      columns: Optional[List[str]] = None,
                      threshold: float = 40,
                      fill_value: Any = None,
                      n_neighbors: int = 5) -> pd.DataFrame:
        """
        Handle missing values using specified strategy
        """
        if columns is None:
            columns = df.columns.tolist()
        
        df_clean = df.copy()
        cols_to_process = [col for col in columns if col in df_clean.columns]
        
        if strategy == 'drop_rows':
            df_clean = df_clean.dropna(subset=cols_to_process)
            self.imputation_report['drop_rows'] = f"Dropped {len(df) - len(df_clean)} rows"
            
        elif strategy == 'drop_columns':
            cols_to_drop = []
            for col in cols_to_process:
                if df_clean[col].isnull().mean() * 100 > threshold:
                    cols_to_drop.append(col)
            df_clean = df_clean.drop(columns=cols_to_drop)
            self.imputation_report['drop_columns'] = f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}"
            
        elif strategy in ['mean', 'median']:
            # Only apply to NUMERICAL columns
            for col in cols_to_process:
                # Check if column is numerical
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if df_clean[col].isnull().sum() > 0:
                        if strategy == 'mean':
                            fill_val = df_clean[col].mean()
                        else:  # median
                            fill_val = df_clean[col].median()
                        
                        if fill_val is not None and not np.isnan(fill_val):
                            df_clean[col] = df_clean[col].fillna(fill_val)
                            self.imputation_report[col] = f"{strategy}: {fill_val:.2f}"
                else:
                    # For non-numerical columns, use mode
                    if df_clean[col].isnull().sum() > 0:
                        mode_val = df_clean[col].mode()
                        if not mode_val.empty:
                            fill_val = mode_val[0]
                            df_clean[col] = df_clean[col].fillna(fill_val)
                            self.imputation_report[col] = f"mode (fallback): {fill_val}"
                        else:
                            df_clean[col] = df_clean[col].fillna("Unknown")
                            self.imputation_report[col] = "mode: Unknown"
        
        elif strategy == 'mode':
            # Apply to ALL columns (works for both numerical and categorical)
            for col in cols_to_process:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        fill_val = mode_val[0]
                        df_clean[col] = df_clean[col].fillna(fill_val)
                        self.imputation_report[col] = f"mode: {fill_val}"
                    else:
                        df_clean[col] = df_clean[col].fillna("Unknown")
                        self.imputation_report[col] = "mode: Unknown"
        
        elif strategy == 'knn':
            # Only apply to NUMERICAL columns
            num_cols = df_clean[cols_to_process].select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) > 1:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
                self.imputation_report['knn'] = f"Applied KNN with {n_neighbors} neighbors on {len(num_cols)} columns"
            else:
                self.imputation_report['knn'] = "Skipped KNN - need at least 2 numeric columns"
                
        elif strategy == 'forward_fill':
            df_clean[cols_to_process] = df_clean[cols_to_process].fillna(method='ffill')
            self.imputation_report['forward_fill'] = "Applied forward fill"
            
        elif strategy == 'backward_fill':
            df_clean[cols_to_process] = df_clean[cols_to_process].fillna(method='bfill')
            self.imputation_report['backward_fill'] = "Applied backward fill"
            
        elif strategy == 'interpolate':
            # Only apply to NUMERICAL columns
            num_cols = df_clean[cols_to_process].select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df_clean[num_cols] = df_clean[num_cols].interpolate(method='linear', limit_direction='forward')
                self.imputation_report['interpolate'] = f"Applied linear interpolation on {len(num_cols)} columns"
            else:
                self.imputation_report['interpolate'] = "Skipped - no numeric columns"
            
        elif strategy == 'constant':
            if fill_value is not None:
                df_clean[cols_to_process] = df_clean[cols_to_process].fillna(fill_value)
                self.imputation_report['constant'] = f"Filled with constant: {fill_value}"
            else:
                raise ValueError("fill_value must be provided for 'constant' strategy")
        
        return df_clean
    
    def get_report(self) -> Dict[str, Any]:
        return self.imputation_report