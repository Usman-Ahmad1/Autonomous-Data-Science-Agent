import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import io
import base64

class VisualizationAgent:
    """
    Interactive Visualization Agent for data exploration
    """
    
    # Define chart types with their requirements
    CHART_TYPES = {
        'bar': {
            'name': '📊 Bar Chart',
            'description': 'Shows categorical data with rectangular bars',
            'requires': ['categorical'],
            'best_for': 'Comparing categories'
        },
        'pie': {
            'name': '🥧 Pie Chart',
            'description': 'Shows proportions of a whole',
            'requires': ['categorical'],
            'best_for': 'Showing percentage distribution'
        },
        'histogram': {
            'name': '📈 Histogram',
            'description': 'Shows distribution of numeric data',
            'requires': ['numeric'],
            'best_for': 'Understanding data distribution'
        },
        'box': {
            'name': '📦 Box Plot',
            'description': 'Shows distribution and outliers',
            'requires': ['numeric'],
            'best_for': 'Identifying outliers and spread'
        },
        'violin': {
            'name': '🎻 Violin Plot',
            'description': 'Shows distribution with density',
            'requires': ['numeric'],
            'best_for': 'Comparing distributions'
        },
        'scatter': {
            'name': '🔵 Scatter Plot',
            'description': 'Shows relationship between two numeric variables',
            'requires': ['numeric', 'numeric'],
            'best_for': 'Finding correlations'
        },
        'line': {
            'name': '📉 Line Chart',
            'description': 'Shows trends over ordered data',
            'requires': ['numeric', 'numeric'],
            'best_for': 'Showing trends'
        },
        'count': {
            'name': '📊 Count Plot',
            'description': 'Shows frequency of categorical values',
            'requires': ['categorical'],
            'best_for': 'Counting occurrences'
        },
        'heatmap': {
            'name': '🔥 Correlation Heatmap',
            'description': 'Shows correlation between numeric columns',
            'requires': ['numeric', 'numeric', 'numeric'],
            'best_for': 'Finding correlations between multiple columns'
        },
        'pairplot': {
            'name': '🔗 Pair Plot',
            'description': 'Shows relationships between multiple numeric columns',
            'requires': ['numeric', 'numeric', 'numeric'],
            'best_for': 'Exploring multiple relationships'
        }
    }
    
    def __init__(self):
        self.visualization_report = {}
    
    def get_suggested_charts(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """
        Get suggested chart types based on selected columns
        """
        if not columns:
            return list(self.CHART_TYPES.keys())
        
        # Determine column types
        numeric_cols = []
        categorical_cols = []
        
        for col in columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        suggested = []
        
        # Single column
        if len(columns) == 1:
            if numeric_cols:
                suggested = ['histogram', 'box', 'violin']
            if categorical_cols:
                suggested = ['bar', 'pie', 'count']
        
        # Two columns
        elif len(columns) == 2:
            if len(numeric_cols) == 2:
                suggested = ['scatter', 'line', 'histogram']
            elif len(numeric_cols) == 1 and len(categorical_cols) == 1:
                suggested = ['bar', 'box', 'violin']
            elif len(categorical_cols) == 2:
                suggested = ['bar', 'count']
        
        # Multiple columns (3+)
        elif len(columns) >= 3:
            if len(numeric_cols) >= 2:
                suggested = ['heatmap', 'pairplot', 'bar']
            else:
                suggested = ['bar']
        
        # Add default if no suggestions
        if not suggested:
            suggested = ['histogram', 'bar', 'box']
        
        return suggested
    
    def get_chart_compatibility(self, df: pd.DataFrame, columns: List[str], chart_type: str) -> Dict[str, Any]:
        """
        Check if chart type is compatible with selected columns
        Returns compatibility info with reason
        """
        if not columns:
            return {'compatible': False, 'reason': 'No columns selected'}
        
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        
        chart_info = self.CHART_TYPES.get(chart_type, {})
        requires = chart_info.get('requires', [])
        
        # Check compatibility based on requirements
        if chart_type in ['histogram', 'box', 'violin']:
            if not numeric_cols:
                return {'compatible': False, 'reason': f'❌ {chart_info["name"]} requires numeric columns. Please select numeric columns.'}
            return {'compatible': True, 'reason': '✅ Compatible'}
        
        elif chart_type in ['bar', 'pie', 'count']:
            if not categorical_cols:
                return {'compatible': False, 'reason': f'❌ {chart_info["name"]} requires categorical columns. Please select categorical columns.'}
            return {'compatible': True, 'reason': '✅ Compatible'}
        
        elif chart_type in ['scatter', 'line']:
            if len(numeric_cols) < 2:
                return {'compatible': False, 'reason': f'❌ {chart_info["name"]} requires at least 2 numeric columns. Please select 2 numeric columns.'}
            return {'compatible': True, 'reason': '✅ Compatible'}
        
        elif chart_type in ['heatmap', 'pairplot']:
            if len(numeric_cols) < 2:
                return {'compatible': False, 'reason': f'❌ {chart_info["name"]} requires at least 2 numeric columns. Please select multiple numeric columns.'}
            return {'compatible': True, 'reason': '✅ Compatible'}
        
        return {'compatible': True, 'reason': '✅ Compatible'}
    
    def create_visualization(self, df: pd.DataFrame, 
                            chart_type: str,
                            columns: List[str],
                            target_column: Optional[str] = None,
                            title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualization based on chart type and columns
        """
        
        try:
            # Check compatibility first
            compatibility = self.get_chart_compatibility(df, columns, chart_type)
            if not compatibility['compatible']:
                return {
                    'type': 'error',
                    'message': compatibility['reason']
                }
            
            # Remove any rows with NaN in selected columns for clean visualization
            clean_df = df[columns].dropna()
            
            if clean_df.empty:
                return {
                    'type': 'error',
                    'message': 'No valid data available for visualization after removing NaN values'
                }
            
            fig = None
            plotly_fig = None
            
            # Determine column types
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = clean_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # ============ SINGLE COLUMN ============
            if len(columns) == 1:
                col = columns[0]
                
                if chart_type == 'bar' and col in categorical_cols:
                    value_counts = clean_df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'count']
                    fig = px.bar(value_counts, x=col, y='count', 
                                title=title or f'Bar Chart of {col}')
                
                elif chart_type == 'pie' and col in categorical_cols:
                    value_counts = clean_df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'count']
                    fig = px.pie(value_counts, values='count', names=col,
                                title=title or f'Pie Chart of {col}')
                
                elif chart_type == 'histogram' and col in numeric_cols:
                    fig = px.histogram(clean_df, x=col, 
                                      title=title or f'Histogram of {col}',
                                      nbins=20)
                
                elif chart_type == 'box' and col in numeric_cols:
                    fig = px.box(clean_df, y=col, 
                                title=title or f'Box Plot of {col}')
                
                elif chart_type == 'violin' and col in numeric_cols:
                    fig = px.violin(clean_df, y=col, 
                                   title=title or f'Violin Plot of {col}')
                
                elif chart_type == 'count' and col in categorical_cols:
                    fig = px.histogram(clean_df, x=col, 
                                      title=title or f'Count Plot of {col}')
                
                else:
                    # Default: show appropriate distribution
                    if col in numeric_cols:
                        fig = px.histogram(clean_df, x=col, 
                                          title=title or f'Distribution of {col}',
                                          nbins=20)
                    else:
                        value_counts = clean_df[col].value_counts().reset_index()
                        value_counts.columns = [col, 'count']
                        fig = px.bar(value_counts, x=col, y='count',
                                    title=title or f'Distribution of {col}')
            
            # ============ TWO COLUMNS ============
            elif len(columns) == 2:
                col1, col2 = columns
                
                col1_is_numeric = col1 in numeric_cols
                col2_is_numeric = col2 in numeric_cols
                col1_is_categorical = col1 in categorical_cols
                col2_is_categorical = col2 in categorical_cols
                
                if chart_type == 'histogram':
                    if col1_is_numeric and col2_is_numeric:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=clean_df[col1], name=col1, opacity=0.7))
                        fig.add_trace(go.Histogram(x=clean_df[col2], name=col2, opacity=0.7))
                        fig.update_layout(
                            title=title or f'Histogram: {col1} vs {col2}',
                            barmode='overlay',
                            xaxis_title='Value',
                            yaxis_title='Count'
                        )
                    elif col1_is_categorical and col2_is_numeric:
                        fig = px.histogram(clean_df, x=col2, color=col1,
                                          title=title or f'Histogram of {col2} by {col1}',
                                          nbins=20)
                    elif col2_is_categorical and col1_is_numeric:
                        fig = px.histogram(clean_df, x=col1, color=col2,
                                          title=title or f'Histogram of {col1} by {col2}',
                                          nbins=20)
                    else:
                        fig = px.histogram(clean_df, x=col1, color=col2,
                                          title=title or f'Count: {col1} by {col2}')
                
                elif chart_type == 'scatter' and col1_is_numeric and col2_is_numeric:
                    fig = px.scatter(clean_df, x=col1, y=col2, 
                                    title=title or f'Scatter Plot: {col1} vs {col2}')
                
                elif chart_type == 'bar':
                    if col1_is_categorical and col2_is_numeric:
                        grouped = clean_df.groupby(col1)[col2].mean().reset_index()
                        fig = px.bar(grouped, x=col1, y=col2, 
                                    title=title or f'Bar Chart: Average {col2} by {col1}')
                    elif col2_is_categorical and col1_is_numeric:
                        grouped = clean_df.groupby(col2)[col1].mean().reset_index()
                        fig = px.bar(grouped, x=col2, y=col1, 
                                    title=title or f'Bar Chart: Average {col1} by {col2}')
                    elif col1_is_categorical and col2_is_categorical:
                        cross_tab = pd.crosstab(clean_df[col1], clean_df[col2])
                        fig = px.bar(cross_tab, x=cross_tab.index, y=cross_tab.columns,
                                    title=title or f'Bar Chart: {col1} vs {col2}')
                    else:
                        fig = px.bar(clean_df, x=col1, y=col2,
                                    title=title or f'Bar Chart: {col1} vs {col2}')
                
                elif chart_type == 'box':
                    if col1_is_categorical and col2_is_numeric:
                        fig = px.box(clean_df, x=col1, y=col2, 
                                    title=title or f'Box Plot: {col2} by {col1}')
                    elif col2_is_categorical and col1_is_numeric:
                        fig = px.box(clean_df, x=col2, y=col1, 
                                    title=title or f'Box Plot: {col1} by {col2}')
                    else:
                        fig = px.box(clean_df, y=[col1, col2],
                                    title=title or f'Box Plot: {col1} vs {col2}')
                
                elif chart_type == 'violin':
                    if col1_is_categorical and col2_is_numeric:
                        fig = px.violin(clean_df, x=col1, y=col2, 
                                       title=title or f'Violin Plot: {col2} by {col1}')
                    elif col2_is_categorical and col1_is_numeric:
                        fig = px.violin(clean_df, x=col2, y=col1, 
                                       title=title or f'Violin Plot: {col1} by {col2}')
                    else:
                        fig = px.violin(clean_df, y=[col1, col2],
                                       title=title or f'Violin Plot: {col1} vs {col2}')
                
                elif chart_type == 'line' and col1_is_numeric and col2_is_numeric:
                    fig = px.line(clean_df, x=col1, y=col2, 
                                 title=title or f'Line Chart: {col1} vs {col2}')
                
                elif chart_type == 'count':
                    if col1_is_categorical and col2_is_categorical:
                        fig = px.histogram(clean_df, x=col1, color=col2,
                                          title=title or f'Count: {col1} by {col2}')
                    elif col1_is_categorical:
                        fig = px.histogram(clean_df, x=col1,
                                          title=title or f'Count Plot: {col1}')
                    else:
                        fig = px.histogram(clean_df, x=col2,
                                          title=title or f'Count Plot: {col2}')
                
                else:
                    if col1_is_numeric and col2_is_numeric:
                        fig = px.scatter(clean_df, x=col1, y=col2, 
                                        title=title or f'Scatter: {col1} vs {col2}')
                    elif col1_is_categorical and col2_is_numeric:
                        grouped = clean_df.groupby(col1)[col2].mean().reset_index()
                        fig = px.bar(grouped, x=col1, y=col2, 
                                    title=title or f'Bar: Average {col2} by {col1}')
                    else:
                        fig = px.histogram(clean_df, x=col1, color=col2,
                                          title=title or f'Count: {col1} vs {col2}')
            
            # ============ MULTIPLE COLUMNS ============
            elif len(columns) >= 3:
                if chart_type == 'heatmap':
                    numeric_data = clean_df[numeric_cols].corr()
                    fig = px.imshow(numeric_data, text_auto=True,
                                   title=title or 'Correlation Heatmap',
                                   color_continuous_scale='RdBu_r')
                
                elif chart_type == 'pairplot':
                    fig, axes = plt.subplots(figsize=(10, 8))
                    pair_plot = sns.pairplot(clean_df[numeric_cols[:5]] if len(numeric_cols) > 5 else clean_df[numeric_cols])
                    plt.title(title or 'Pair Plot')
                    plotly_fig = fig
                
                elif chart_type == 'bar':
                    if numeric_cols:
                        means = clean_df[numeric_cols[:10]].mean().reset_index()
                        means.columns = ['Column', 'Mean']
                        fig = px.bar(means, x='Column', y='Mean',
                                    title=title or f'Comparison of {len(numeric_cols[:10])} Columns')
                    else:
                        return {
                            'type': 'error',
                            'message': 'No numeric columns available for bar chart'
                        }
                
                else:
                    if numeric_cols:
                        numeric_data = clean_df[numeric_cols].corr()
                        fig = px.imshow(numeric_data, text_auto=True,
                                       title=title or 'Correlation Heatmap',
                                       color_continuous_scale='RdBu_r')
                    else:
                        return {
                            'type': 'error',
                            'message': 'No numeric columns available for heatmap'
                        }
            
            # Convert matplotlib figure to base64 if needed
            if plotly_fig is not None and fig is None:
                buf = io.BytesIO()
                plotly_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                plt.close()
                return {
                    'type': 'matplotlib',
                    'image': img_base64,
                    'title': title or 'Visualization'
                }
            
            # Return plotly figure
            if fig is not None:
                return {
                    'type': 'plotly',
                    'figure': fig,
                    'title': title or 'Visualization'
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Could not create visualization with selected options'
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Error creating visualization: {str(e)}'
            }