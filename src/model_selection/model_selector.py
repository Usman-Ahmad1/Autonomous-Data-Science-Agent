import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from typing import Dict, Any, List, Optional, Tuple
import warnings
import time
warnings.filterwarnings('ignore')

class ModelSelector:
    """
    Automated model selection with multiple algorithms
    """
    
    # Classification models
    CLASSIFICATION_MODELS = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Regression models
    REGRESSION_MODELS = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    def __init__(self, problem_type: str = 'auto'):
        """
        Initialize Model Selector
        
        Args:
            problem_type: 'classification', 'regression', or 'auto'
        """
        self.problem_type = problem_type
        self.selected_model = None
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None
        
    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Auto-detect if it's classification or regression
        """
        unique_values = y.nunique()
        
        # If target has few unique values, it's classification
        if unique_values <= 20:
            # Check if values are integers (categorical)
            if y.dtype in ['int64', 'int32', 'object', 'category']:
                return 'classification'
        
        # Check if values are floats with many unique values
        if y.dtype in ['float64', 'float32'] and unique_values > 20:
            return 'regression'
        
        # Default: check value range
        if y.dtype in ['int64', 'int32']:
            if y.min() >= 0 and y.max() <= 20:
                return 'classification'
        
        return 'regression'
    
    def select_models(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.2,
                     cv_folds: int = 5,
                     models_to_try: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select and evaluate multiple models
        
        Args:
            X: Feature dataframe
            y: Target series
            test_size: Train-test split ratio
            cv_folds: Number of cross-validation folds
            models_to_try: List of specific models to try (None = all)
        
        Returns:
            Dictionary with results for each model
        """
        # Auto-detect problem type
        if self.problem_type == 'auto':
            self.problem_type = self.detect_problem_type(y)
        
        print(f"🎯 Problem Type: {self.problem_type.upper()}")
        print(f"📊 Target: {y.name} with {y.nunique()} unique values")
        
        # Select appropriate models
        if self.problem_type == 'classification':
            models = self.CLASSIFICATION_MODELS
            metric = 'accuracy'
            scoring = 'accuracy'
        else:
            models = self.REGRESSION_MODELS
            metric = 'r2'
            scoring = 'r2'
        
        # Filter models if specified
        if models_to_try:
            models = {name: model for name, model in models.items() 
                     if name in models_to_try}
        
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
        
        results = {}
        
        print("\n" + "=" * 60)
        print("🧪 Evaluating Models...")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\n📊 Testing: {name}")
            
            try:
                start_time = time.time()
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if self.problem_type == 'classification':
                    test_score = accuracy_score(y_test, y_pred)
                    metric_name = 'accuracy'
                else:
                    test_score = r2_score(y_test, y_pred)
                    metric_name = 'r2'
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score,
                    'metric': metric_name,
                    'training_time': training_time,
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                print(f"  ✅ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"  ✅ Test {metric_name}: {test_score:.4f}")
                print(f"  ⏱️ Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results[name] = {
                    'model': None,
                    'cv_mean': 0,
                    'cv_std': 0,
                    'test_score': 0,
                    'metric': metric,
                    'training_time': 0,
                    'error': str(e)
                }
        
        # Find best model
        if results:
            best_name = max(results.items(), 
                           key=lambda x: x[1].get('test_score', 0) if x[1].get('model') is not None else -1)[0]
            self.best_model_name = best_name
            self.best_model = results[best_name]['model']
            self.model_results = results
            
            print("\n" + "=" * 60)
            print(f"🏆 Best Model: {best_name}")
            print(f"   {results[best_name]['metric']}: {results[best_name]['test_score']:.4f}")
            print("=" * 60)
        
        return results
    
    def get_best_model(self):
        """Return the best model"""
        return self.best_model
    
    def get_best_model_name(self):
        """Return the name of the best model"""
        return self.best_model_name
    
    def get_model_results(self) -> pd.DataFrame:
        """Get results as a DataFrame for comparison"""
        if not self.model_results:
            return pd.DataFrame()
        
        data = []
        for name, result in self.model_results.items():
            row = {
                'Model': name,
                'CV Mean': result.get('cv_mean', 0),
                'CV Std': result.get('cv_std', 0),
                'Test Score': result.get('test_score', 0),
                'Training Time (s)': result.get('training_time', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('Test Score', ascending=False)
        return df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model selected. Run select_models() first.")
        
        # Process features
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = LabelEncoder().fit_transform(X_processed[col].astype(str))
        
        return self.best_model.predict(X_processed)