import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureTransformer:
    """
    Transforms features using various techniques
    """
    
    def __init__(self):
        self.transform_report = {}
        self.scalers = {}
    
    def scale_features(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Methods: 'standard', 'minmax', 'robust'
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        if not num_cols:
            return df_new
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        df_new[num_cols] = scaler.fit_transform(df_new[num_cols])
        self.scalers[method] = scaler
        
        self.transform_report['scaling'] = {
            'method': method,
            'columns_scaled': num_cols
        }
        
        return df_new
    
    def encode_categorical(self, df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        
        Methods: 'label', 'onehot'
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        cat_cols = [col for col in columns if col in df_new.columns]
        
        if not cat_cols:
            return df_new
        
        if method == 'label':
            for col in cat_cols:
                le = LabelEncoder()
                df_new[col] = le.fit_transform(df_new[col].astype(str))
            
            self.transform_report['encoding'] = {
                'method': 'label',
                'columns_encoded': cat_cols
            }
            
        elif method == 'onehot':
            df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)
            
            self.transform_report['encoding'] = {
                'method': 'onehot',
                'columns_encoded': cat_cols,
                'new_columns': [col for col in df_new.columns if col not in df.columns]
            }
        
        return df_new
    
    def log_transform(self, df: pd.DataFrame,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply log transformation to reduce skewness
        """
        df_new = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        num_cols = [col for col in columns if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]
        
        for col in num_cols:
            # Add 1 to avoid log(0)
            if (df_new[col] >= 0).all():
                df_new[f'log_{col}'] = np.log1p(df_new[col])
            else:
                # Shift to positive
                min_val = df_new[col].min()
                df_new[f'log_{col}'] = np.log1p(df_new[col] - min_val + 1)
        
        self.transform_report['log_transform'] = {
            'columns_transformed': num_cols
        }
        
        return df_new
    
    def get_report(self) -> Dict[str, Any]:
        return self.transform_report