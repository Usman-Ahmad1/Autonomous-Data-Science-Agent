import pandas as pd
from typing import Dict, Any, List, Optional

class DuplicateHandler:
    """Handle duplicate rows and columns"""
    
    def __init__(self):
        self.duplicate_report = {}
    
    def detect_duplicate_rows(self, df: pd.DataFrame, 
                             subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Detect duplicate rows"""
        if subset is None:
            duplicates = df.duplicated(keep=False)
        else:
            duplicates = df.duplicated(subset=subset, keep=False)
        
        duplicate_df = df[duplicates].sort_values(df.columns.tolist())
        
        self.duplicate_report['rows'] = {
            'total_duplicates': duplicates.sum(),
            'duplicate_indices': duplicate_df.index.tolist()
        }
        
        return duplicate_df
    
    def remove_duplicate_rows(self, df: pd.DataFrame,
                             subset: Optional[List[str]] = None,
                             keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        original_len = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        
        self.duplicate_report['rows_removed'] = {
            'original_rows': original_len,
            'remaining_rows': len(df_clean),
            'removed_rows': original_len - len(df_clean),
            'keep_strategy': keep
        }
        
        return df_clean
    
    def detect_duplicate_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect duplicate columns (identical values)"""
        duplicate_cols = {}
        columns = df.columns.tolist()
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if df[columns[i]].equals(df[columns[j]]):
                    duplicate_cols[columns[j]] = columns[i]
        
        self.duplicate_report['columns'] = duplicate_cols
        return duplicate_cols
    
    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns"""
        duplicate_map = self.detect_duplicate_columns(df)
        cols_to_drop = list(duplicate_map.keys())
        df_clean = df.drop(columns=cols_to_drop)
        
        self.duplicate_report['columns_removed'] = {
            'dropped_columns': cols_to_drop,
            'remaining_columns': len(df_clean.columns)
        }
        
        return df_clean
    
    def get_report(self) -> Dict[str, Any]:
        return self.duplicate_report