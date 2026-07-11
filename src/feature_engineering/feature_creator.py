import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class FeatureCreator:
    """
    Creates new features from existing data
    """
    
    def __init__(self):
        self.created_features = []
        self.feature_report = {}
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   max_features: int = 10) -> pd.DataFrame:
        """
        Create interaction features (multiplication of features)
        
        Args:
            df: Input dataframe
            columns: Specific columns to use (None = all numeric)
            max_features: Maximum number of interaction features to create
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only use numeric columns
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        if len(num_cols) < 2:
            return df_new
        
        # Create interaction features
        interactions_created = 0
        for col1, col2 in combinations(num_cols, 2):
            if interactions_created >= max_features:
                break
                
            new_feature_name = f"{col1}_x_{col2}"
            df_new[new_feature_name] = df_new[col1] * df_new[col2]
            self.created_features.append(new_feature_name)
            interactions_created += 1
        
        self.feature_report['interactions'] = {
            'created': interactions_created,
            'features': self.created_features[-interactions_created:] if interactions_created > 0 else []
        }
        
        return df_new
    
    def create_ratio_features(self, df: pd.DataFrame,
                             columns: Optional[List[str]] = None,
                             max_features: int = 5) -> pd.DataFrame:
        """
        Create ratio features (division of features)
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        if len(num_cols) < 2:
            return df_new
        
        ratios_created = 0
        for col1, col2 in combinations(num_cols, 2):
            if ratios_created >= max_features:
                break
            
            # Avoid division by zero
            if (df_new[col2] != 0).all():
                new_feature_name = f"{col1}_div_{col2}"
                df_new[new_feature_name] = df_new[col1] / df_new[col2]
                self.created_features.append(new_feature_name)
                ratios_created += 1
        
        self.feature_report['ratios'] = {
            'created': ratios_created,
            'features': self.created_features[-ratios_created:] if ratios_created > 0 else []
        }
        
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                  columns: Optional[List[str]] = None,
                                  degree: int = 2,
                                  max_features: int = 10) -> pd.DataFrame:
        """
        Create polynomial features (square, cube, etc.)
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        poly_created = 0
        for col in num_cols:
            if poly_created >= max_features:
                break
                
            for d in range(2, degree + 1):
                if poly_created >= max_features:
                    break
                    
                new_feature_name = f"{col}_power_{d}"
                df_new[new_feature_name] = df_new[col] ** d
                self.created_features.append(new_feature_name)
                poly_created += 1
        
        self.feature_report['polynomial'] = {
            'created': poly_created,
            'features': self.created_features[-poly_created:] if poly_created > 0 else []
        }
        
        return df_new
    
    def create_binned_features(self, df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              bins: int = 5) -> pd.DataFrame:
        """
        Create binned/categorized features from continuous features
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        binned_created = 0
        for col in num_cols[:5]:  # Limit to 5 columns
            new_feature_name = f"{col}_binned"
            df_new[new_feature_name] = pd.cut(df_new[col], bins=bins, labels=False)
            self.created_features.append(new_feature_name)
            binned_created += 1
        
        self.feature_report['binned'] = {
            'created': binned_created,
            'features': self.created_features[-binned_created:] if binned_created > 0 else []
        }
        
        return df_new
    
    def create_date_features(self, df: pd.DataFrame,
                            date_column: str) -> pd.DataFrame:
        """
        Extract features from date column
        """
        df_new = df.copy()
        
        if date_column not in df_new.columns:
            return df_new
        
        # Convert to datetime if not already
        df_new[date_column] = pd.to_datetime(df_new[date_column])
        
        # Extract features
        df_new[f'{date_column}_year'] = df_new[date_column].dt.year
        df_new[f'{date_column}_month'] = df_new[date_column].dt.month
        df_new[f'{date_column}_day'] = df_new[date_column].dt.day
        df_new[f'{date_column}_dayofweek'] = df_new[date_column].dt.dayofweek
        df_new[f'{date_column}_quarter'] = df_new[date_column].dt.quarter
        
        # Add to created features list
        date_features = [f'{date_column}_year', f'{date_column}_month', 
                        f'{date_column}_day', f'{date_column}_dayofweek',
                        f'{date_column}_quarter']
        self.created_features.extend(date_features)
        
        self.feature_report['date'] = {
            'created': len(date_features),
            'features': date_features
        }
        
        return df_new
    
    def get_report(self) -> Dict[str, Any]:
        return self.feature_report