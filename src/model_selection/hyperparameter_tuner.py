import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Optional, Tuple
import warnings
import time
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """
    Hyperparameter Tuning Agent - Optimizes model parameters
    """
    
    # Default hyperparameter grids for different models
    HYPERPARAMETER_GRIDS = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']  # for classification
        }
    }
    
    def __init__(self, model, model_name: str, problem_type: str = 'classification'):
        self.model = model
        self.model_name = model_name
        self.problem_type = problem_type
        self.best_params = None
        self.best_score = None
        self.tuning_results = None
        self.tuning_time = None
    
    def tune(self, X, y, param_grid: Optional[Dict] = None,
             search_type: str = 'grid', cv_folds: int = 5,
             n_iter: int = 20, scoring: Optional[str] = None,
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Custom parameter grid (uses default if None)
            search_type: 'grid' or 'random'
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric (auto-detected if None)
            test_size: Test set size for final evaluation
        """
        
        # Auto-detect scoring metric
        if scoring is None:
            if self.problem_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'r2'
        
        # Use default grid if not provided
        if param_grid is None:
            param_grid = self.HYPERPARAMETER_GRIDS.get(self.model_name, {})
            
            if not param_grid:
                # Fallback grid for unknown models
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                }
        
        # Encode categorical features if needed
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = LabelEncoder().fit_transform(X_processed[col].astype(str))
        
        # Encode target if classification
        y_processed = y.copy()
        if self.problem_type == 'classification' and y_processed.dtype == 'object':
            y_processed = LabelEncoder().fit_transform(y_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42
        )
        
        start_time = time.time()
        
        print(f"🔧 Tuning {self.model_name} with {search_type} search...")
        print(f"   Parameter grid: {param_grid}")
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv_folds, 
                scoring=scoring, n_jobs=-1, verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                self.model, param_grid, n_iter=n_iter, 
                cv=cv_folds, scoring=scoring, 
                n_jobs=-1, random_state=42, verbose=0
            )
        
        search.fit(X_train, y_train)
        
        self.tuning_time = time.time() - start_time
        
        # Get results
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.tuning_results = {
            'best_params': self.best_params,
            'best_cv_score': self.best_score,
            'search_type': search_type,
            'cv_folds': cv_folds,
            'tuning_time': self.tuning_time,
            'all_results': pd.DataFrame(search.cv_results_)
        }
        
        # Evaluate on test set
        y_pred = search.best_estimator_.predict(X_test)
        
        if self.problem_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            test_metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
        
        self.tuning_results['test_metrics'] = test_metrics
        self.tuning_results['best_estimator'] = search.best_estimator_
        
        print(f"✅ Tuning complete! Best score: {self.best_score:.4f}")
        print(f"   Best parameters: {self.best_params}")
        
        return self.tuning_results
    
    def get_best_model(self):
        """Return the best tuned model"""
        if self.tuning_results and 'best_estimator' in self.tuning_results:
            return self.tuning_results['best_estimator']
        return self.model
    
    def get_improvement_report(self, original_score: float) -> Dict[str, Any]:
        """Get improvement report comparing original vs tuned model"""
        if self.best_score is None:
            return {'error': 'No tuning performed yet'}
        
        improvement = self.best_score - original_score
        improvement_pct = (improvement / original_score * 100) if original_score != 0 else float('inf')
        
        return {
            'original_score': original_score,
            'tuned_score': self.best_score,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'best_params': self.best_params,
            'tuning_time': self.tuning_time
        }