import pandas as pd
import numpy as np  # ADD THIS IMPORT
from typing import Dict, Any, Optional, List
from .missing_value_handler import MissingValueHandler
from .outlier_handler import OutlierHandler
from .duplicate_handler import DuplicateHandler

class CleaningPipeline:
    """Complete data cleaning pipeline"""
    
    def __init__(self):
        self.missing_handler = MissingValueHandler()
        self.outlier_handler = OutlierHandler()
        self.duplicate_handler = DuplicateHandler()
        self.pipeline_report = {}
    
    def clean(self, df: pd.DataFrame,
             missing_strategy: str = 'median',
             outlier_method: str = 'iqr',
             outlier_handling: str = 'cap',
             remove_duplicates: bool = True,
             remove_duplicate_cols: bool = True,
             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Run complete cleaning pipeline"""
        
        df_clean = df.copy()
        report = {}
        
        # First, identify numerical and categorical columns
        if columns is not None:
            num_cols = df_clean[columns].select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df_clean[columns].select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Handle missing values
        if missing_strategy:
            # For mean/median, only apply to numerical columns
            if missing_strategy in ['mean', 'median']:
                if num_cols:
                    df_clean = self.missing_handler.handle_missing(
                        df_clean, 
                        strategy=missing_strategy,
                        columns=num_cols
                    )
                # Also handle categorical with mode
                if cat_cols:
                    df_clean = self.missing_handler.handle_missing(
                        df_clean,
                        strategy='mode',
                        columns=cat_cols
                    )
            else:
                # Apply to all columns
                df_clean = self.missing_handler.handle_missing(
                    df_clean, 
                    strategy=missing_strategy,
                    columns=columns
                )
            report['missing'] = self.missing_handler.get_report()
        
        # 2. Handle outliers (only on numerical columns)
        if outlier_method and outlier_handling and num_cols:
            df_clean = self.outlier_handler.handle_outliers(
                df_clean,
                method=outlier_handling,
                detection_method=outlier_method,
                columns=num_cols
            )
            report['outliers'] = self.outlier_handler.get_report()
        
        # 3. Remove duplicates
        if remove_duplicates:
            df_clean = self.duplicate_handler.remove_duplicate_rows(df_clean)
            report['duplicates_rows'] = self.duplicate_handler.duplicate_report
        
        # 4. Remove duplicate columns
        if remove_duplicate_cols:
            df_clean = self.duplicate_handler.remove_duplicate_columns(df_clean)
            report['duplicates_columns'] = self.duplicate_handler.duplicate_report
        
        self.pipeline_report = {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'details': report
        }
        
        return df_clean
    
    def get_report(self) -> Dict[str, Any]:
        return self.pipeline_report