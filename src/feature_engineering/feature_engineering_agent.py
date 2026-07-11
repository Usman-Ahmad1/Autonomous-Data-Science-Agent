import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .feature_creator import FeatureCreator
from .feature_transformer import FeatureTransformer
from .feature_selector import FeatureSelector

class FeatureEngineeringAgent:
    """
    Main Feature Engineering Agent that orchestrates all feature engineering tasks
    """
    
    def __init__(self):
        self.creator = FeatureCreator()
        self.transformer = FeatureTransformer()
        self.selector = FeatureSelector()
        self.feature_report = {}
    
    def engineer_features(self, df: pd.DataFrame,
                         target_column: Optional[str] = None,
                         create_interactions: bool = True,
                         create_ratios: bool = True,
                         create_polynomial: bool = True,
                         create_bins: bool = True,
                         scale_features: bool = True,
                         scale_method: str = 'standard',
                         encode_categorical: bool = True,
                         encoding_method: str = 'label',
                         log_transform: bool = True,
                         select_features: bool = False,
                         selection_method: str = 'correlation',
                         n_features: int = 10) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            df: Input dataframe
            target_column: Target column for feature selection
            create_interactions: Create interaction features
            create_ratios: Create ratio features
            create_polynomial: Create polynomial features
            create_bins: Create binned features
            scale_features: Scale numerical features
            scale_method: 'standard', 'minmax', 'robust'
            encode_categorical: Encode categorical features
            encoding_method: 'label', 'onehot'
            log_transform: Apply log transformation
            select_features: Select best features
            selection_method: 'correlation', 'mutual_info', 'rfe'
            n_features: Number of features to select
        """
        
        df_engineered = df.copy()
        report = {}
        total_features_created = 0
        
        print("=" * 60)
        print("🔧 FEATURE ENGINEERING STARTED")
        print(f"Original shape: {df_engineered.shape}")
        print("=" * 60)
        
        # 1. Create new features
        if create_interactions:
            df_engineered = self.creator.create_interaction_features(df_engineered)
            report['interactions'] = self.creator.feature_report.get('interactions', {})
            total_features_created += len(report['interactions'].get('features', []))
        
        if create_ratios:
            df_engineered = self.creator.create_ratio_features(df_engineered)
            report['ratios'] = self.creator.feature_report.get('ratios', {})
            total_features_created += len(report['ratios'].get('features', []))
        
        if create_polynomial:
            df_engineered = self.creator.create_polynomial_features(df_engineered)
            report['polynomial'] = self.creator.feature_report.get('polynomial', {})
            total_features_created += len(report['polynomial'].get('features', []))
        
        if create_bins:
            df_engineered = self.creator.create_binned_features(df_engineered)
            report['binned'] = self.creator.feature_report.get('binned', {})
            total_features_created += len(report['binned'].get('features', []))
        
        # 2. Transform features
        if scale_features:
            df_engineered = self.transformer.scale_features(df_engineered, method=scale_method)
            report['scaling'] = self.transformer.transform_report.get('scaling', {})
        
        if encode_categorical:
            df_engineered = self.transformer.encode_categorical(df_engineered, method=encoding_method)
            report['encoding'] = self.transformer.transform_report.get('encoding', {})
        
        if log_transform:
            df_engineered = self.transformer.log_transform(df_engineered)
            report['log_transform'] = self.transformer.transform_report.get('log_transform', {})
        
        # 3. Select features (if target column provided)
        if select_features and target_column and target_column in df_engineered.columns:
            if selection_method == 'correlation':
                selected = self.selector.select_by_correlation(df_engineered, target_column)
            elif selection_method == 'mutual_info':
                selected = self.selector.select_by_mutual_info(df_engineered, target_column, k=n_features)
            elif selection_method == 'rfe':
                selected = self.selector.select_by_rfe(df_engineered, target_column, n_features=n_features)
            
            report['feature_selection'] = self.selector.selection_report.get(selection_method, {})
            
            # Keep only selected features + target
            if selected:
                keep_cols = [target_column] + selected
                df_engineered = df_engineered[keep_cols]
        
        # Store report
        self.feature_report = {
            'original_shape': df.shape,
            'new_shape': df_engineered.shape,
            'features_created': total_features_created,
            'details': report
        }
        
        print(f"✅ Feature Engineering Complete!")
        print(f"New shape: {df_engineered.shape}")
        print(f"Features created: {total_features_created}")
        print("=" * 60)
        
        return df_engineered
    
    def get_report(self) -> Dict[str, Any]:
        return self.feature_report