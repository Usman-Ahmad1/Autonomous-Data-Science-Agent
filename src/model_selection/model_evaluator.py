import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive Model Evaluation Agent
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, 
                               model=None, X=None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                results['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report as dictionary
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # Cross-validation scores if model and X provided
        if model is not None and X is not None:
            try:
                cv_scores = cross_val_score(model, X, y_true, cv=5, scoring='accuracy')
                results['cv_mean'] = cv_scores.mean()
                results['cv_std'] = cv_scores.std()
                results['cv_scores'] = cv_scores.tolist()
            except:
                pass
        
        self.evaluation_results = results
        return results
    
    def evaluate_regression(self, y_true, y_pred, model=None, X=None) -> Dict[str, Any]:
        """
        Comprehensive regression evaluation
        """
        results = {}
        
        # Basic metrics
        results['r2_score'] = r2_score(y_true, y_pred)
        results['explained_variance'] = explained_variance_score(y_true, y_pred)
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        results['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle zero values)
        try:
            results['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            results['mape'] = None
        
        # Cross-validation scores if model and X provided
        if model is not None and X is not None:
            try:
                cv_scores = cross_val_score(model, X, y_true, cv=5, scoring='r2')
                results['cv_mean'] = cv_scores.mean()
                results['cv_std'] = cv_scores.std()
                results['cv_scores'] = cv_scores.tolist()
            except:
                pass
        
        self.evaluation_results = results
        return results
    
    def create_evaluation_charts(self, y_true, y_pred, problem_type: str = 'regression') -> Dict[str, Any]:
        """
        Create visual evaluation charts
        """
        charts = {}
        
        if problem_type == 'classification':
            # Confusion matrix heatmap
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(
                cm, 
                text_auto=True,
                title='Confusion Matrix',
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale='Blues'
            )
            charts['confusion_matrix'] = fig_cm
            
            # Feature importance if available
            # (will be added separately)
            
        else:  # regression
            # Actual vs Predicted scatter
            fig_scatter = px.scatter(
                x=y_true, y=y_pred,
                title='Actual vs Predicted Values',
                labels=dict(x="Actual", y="Predicted"),
                trendline='ols'
            )
            fig_scatter.add_shape(
                type="line",
                x0=y_true.min(), y0=y_true.min(),
                x1=y_true.max(), y1=y_true.max(),
                line=dict(color="red", dash="dash")
            )
            charts['actual_vs_predicted'] = fig_scatter
            
            # Residual plot
            residuals = y_true - y_pred
            fig_residual = px.scatter(
                x=y_pred, y=residuals,
                title='Residual Plot',
                labels=dict(x="Predicted Values", y="Residuals")
            )
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            charts['residuals'] = fig_residual
            
            # Residual distribution
            fig_resid_dist = px.histogram(
                residuals,
                title='Residual Distribution',
                labels=dict(value="Residuals"),
                nbins=30
            )
            charts['residual_distribution'] = fig_resid_dist
        
        return charts
    
    def get_interpretation(self, results: Dict[str, Any], problem_type: str) -> str:
        """
        Generate human-readable interpretation of results
        """
        interpretations = []
        interpretations.append("📊 **Model Performance Interpretation**")
        interpretations.append("")
        
        if problem_type == 'classification':
            acc = results.get('accuracy', 0)
            f1 = results.get('f1_score', 0)
            
            if acc >= 0.9:
                interpretations.append("✅ **Excellent Performance!** The model is highly accurate.")
            elif acc >= 0.8:
                interpretations.append("👍 **Good Performance!** The model performs well.")
            elif acc >= 0.7:
                interpretations.append("📈 **Moderate Performance.** Consider further optimization.")
            else:
                interpretations.append("⚠️ **Needs Improvement.** Consider more data or feature engineering.")
            
            interpretations.append(f"   - Accuracy: {acc:.2%}")
            interpretations.append(f"   - F1 Score: {f1:.2%}")
            
            if results.get('roc_auc'):
                auc = results['roc_auc']
                interpretations.append(f"   - ROC AUC: {auc:.2%}")
                if auc >= 0.9:
                    interpretations.append("     ✅ Excellent discrimination ability")
                elif auc >= 0.8:
                    interpretations.append("     👍 Good discrimination ability")
            
        else:  # regression
            r2 = results.get('r2_score', 0)
            rmse = results.get('rmse', 0)
            
            if r2 >= 0.8:
                interpretations.append("✅ **Excellent Fit!** The model explains most variance.")
            elif r2 >= 0.6:
                interpretations.append("👍 **Good Fit!** The model captures key patterns.")
            elif r2 >= 0.4:
                interpretations.append("📈 **Moderate Fit.** Some patterns captured.")
            else:
                interpretations.append("⚠️ **Weak Fit.** Consider more features or different model.")
            
            interpretations.append(f"   - R² Score: {r2:.2%}")
            interpretations.append(f"   - RMSE: {rmse:.2f}")
            
            if results.get('mae'):
                interpretations.append(f"   - MAE: {results['mae']:.2f}")
        
        return "\n".join(interpretations)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary"""
        return self.evaluation_results