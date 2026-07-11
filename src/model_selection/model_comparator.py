import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """
    Compare multiple models and visualize results
    """
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their evaluation metrics
        
        Args:
            model_results: Dictionary of model_name -> evaluation_results
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            
            # Extract metrics based on what's available
            if 'accuracy' in results:
                row['Accuracy'] = results['accuracy']
                row['F1 Score'] = results.get('f1_score', 0)
                row['Precision'] = results.get('precision', 0)
                row['Recall'] = results.get('recall', 0)
            elif 'r2_score' in results:
                row['R2 Score'] = results['r2_score']
                row['RMSE'] = results.get('rmse', 0)
                row['MAE'] = results.get('mae', 0)
                row['MSE'] = results.get('mse', 0)
            
            # Add training time if available
            row['Training Time (s)'] = results.get('training_time', 0)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (accuracy or r2)
        if 'Accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        elif 'R2 Score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('R2 Score', ascending=False)
        
        self.comparison_results = comparison_df
        return comparison_df
    
    def create_comparison_chart(self, comparison_df: pd.DataFrame, 
                               metric: str = 'Accuracy') -> go.Figure:
        """
        Create a bar chart comparing models by a specific metric
        """
        if metric not in comparison_df.columns:
            # Find first numeric column
            for col in comparison_df.columns:
                if comparison_df[col].dtype in ['float64', 'int64'] and col != 'Training Time (s)':
                    metric = col
                    break
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=metric,
            title=f'Model Comparison: {metric}',
            color=metric,
            color_continuous_scale='Viridis',
            text=comparison_df[metric].round(4)
        )
        
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title=metric,
            showlegend=False,
            height=500
        )
        
        fig.update_traces(textposition='outside')
        
        return fig
    
    def create_radar_chart(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create a radar chart comparing models across multiple metrics
        """
        # Select numeric columns (exclude non-metric columns)
        metric_cols = [col for col in comparison_df.columns if col != 'Model' and 
                      col != 'Training Time (s)' and 
                      comparison_df[col].dtype in ['float64', 'int64']]
        
        if len(metric_cols) < 2:
            return None
        
        fig = go.Figure()
        
        # Normalize metrics for radar chart
        for _, row in comparison_df.iterrows():
            values = []
            for col in metric_cols:
                max_val = comparison_df[col].max()
                if max_val > 0:
                    normalized = row[col] / max_val
                else:
                    normalized = 0
                values.append(normalized)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_cols,
                name=row['Model'],
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title='Model Comparison Radar Chart',
            height=500
        )
        
        return fig