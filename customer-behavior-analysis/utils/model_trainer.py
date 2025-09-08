import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, silhouette_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import os

warnings.filterwarnings('ignore')

class CustomerMLPipeline:
    """
    Comprehensive ML pipeline for customer analytics
    """
    
    def __init__(self, models_dir='models/'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_features(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        """
        df = data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        categorical_features = []
        for col in categorical_columns:
            if col != target_column:
                # One-hot encoding for categorical variables
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                categorical_features.extend(dummies.columns.tolist())
                df.drop(col, axis=1, inplace=True)
        
        # Feature engineering
        df = self._create_engineered_features(df)
        
        # Prepare target variable
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column]
            df.drop(target_column, axis=1, inplace=True)
        
        # Remove non-numeric columns that might be left
        df = df.select_dtypes(include=[np.number])
        
        return df, y
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for better model performance
        """
        # RFM features
        if all(col in df.columns for col in ['total_spent', 'total_orders', 'days_since_last_purchase']):
            # Recency, Frequency, Monetary features
            df['avg_order_value'] = df['total_spent'] / df['total_orders']
            df['purchase_intensity'] = df['total_orders'] / (df['days_since_last_purchase'] + 1)
            df['spending_momentum'] = df['total_spent'] / (df['days_since_last_purchase'] + 1)
        
        # Age-related features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            df['age_group'] = df['age_group'].cat.codes
        
        # Interaction features
        if 'satisfaction_score' in df.columns and 'total_spent' in df.columns:
            df['satisfaction_value_ratio'] = df['satisfaction_score'] * df['total_spent']
        
        # Behavioral ratios
        if 'website_visits' in df.columns and 'total_orders' in df.columns:
            df['conversion_rate'] = df['total_orders'] / (df['website_visits'] + 1)
        
        return df
    
    def train_churn_prediction_model(self, data: pd.DataFrame, target_col='churn'):
        """
        Train churn prediction models and select the best one
        """
        print("Training churn prediction models...")
        
        # Prepare features
        X, y = self.prepare_features(data, target_col)
        
        if y is None:
            print(f"Target column '{target_col}' not found in data")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to test
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['LogisticRegression', 'SVM']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            model_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            # Select best model based on F1 score
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        # Save best model and scaler
        self.models['churn_prediction'] = best_model
        self.scalers['churn_prediction'] = scaler
        self.feature_columns['churn_prediction'] = X.columns.tolist()
        
        # Save to disk
        joblib.dump(best_model, f'{self.models_dir}churn_predictor.pkl')
        joblib.dump(scaler, f'{self.models_dir}churn_scaler.pkl')
        
        print(f"\nBest model: {best_model_name} (F1 Score: {best_score:.3f})")
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'results': model_results,
            'feature_importance': self._get_feature_importance(best_model, X.columns) if hasattr(best_model, 'feature_importances_') else None
        }
    
    def train_customer_segmentation_model(self, data: pd.DataFrame, n_clusters=5):
        """
        Train customer segmentation model using K-Means clustering
        """
        print("Training customer segmentation model...")
        
        # Select features for segmentation
        segmentation_features = [
            'total_spent', 'total_orders', 'days_since_last_purchase',
            'satisfaction_score', 'age'
        ]
        
        # Filter available features
        available_features = [col for col in segmentation_features if col in data.columns]
        
        if len(available_features) < 3:
            print("Not enough features for segmentation")
            return None
        
        X = data[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(X)//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            
            if len(X) > k:  # Avoid error when dataset is too small
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Select optimal k (you can modify this logic)
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = n_clusters
        
        # Train final model with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Create segment names
        segment_names = {
            0: 'High Value Champions',
            1: 'Loyal Customers', 
            2: 'Potential Loyalists',
            3: 'At Risk',
            4: 'Low Value'
        }
        
        # Add more names if needed
        for i in range(5, optimal_k):
            segment_names[i] = f'Segment {i+1}'
        
        # Save model and scaler
        self.models['customer_segmentation'] = kmeans
        self.scalers['customer_segmentation'] = scaler
        self.feature_columns['customer_segmentation'] = available_features
        
        # Save to disk
        joblib.dump(kmeans, f'{self.models_dir}customer_segmentation.pkl')
        joblib.dump(scaler, f'{self.models_dir}segmentation_scaler.pkl')
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(X, cluster_labels, segment_names)
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {silhouette_score(X_scaled, cluster_labels):.3f}")
        
        return {
            'model': kmeans,
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'segment_names': segment_names,
            'cluster_analysis': cluster_analysis,
            'silhouette_score': silhouette_score(X_scaled, cluster_labels)
        }
    
    def train_clv_prediction_model(self, data: pd.DataFrame, target_col='customer_lifetime_value'):
        """
        Train Customer Lifetime Value prediction model
        """
        print("Training CLV prediction model...")
        
        # Prepare features
        X, y = self.prepare_features(data, target_col)
        
        if y is None:
            print(f"Target column '{target_col}' not found in data")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        self.models['clv_prediction'] = rf_model
        self.scalers['clv_prediction'] = scaler
        self.feature_columns['clv_prediction'] = X.columns.tolist()
        
        joblib.dump(rf_model, f'{self.models_dir}clv_predictor.pkl')
        joblib.dump(scaler, f'{self.models_dir}clv_scaler.pkl')
        
        print(f"CLV Model - RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return {
            'model': rf_model,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'feature_importance': self._get_feature_importance(rf_model, X.columns)
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Get feature importance from tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def _analyze_clusters(self, X, labels, segment_names):
        """Analyze cluster characteristics"""
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            cluster_data = X[labels == cluster_id]
            
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'characteristics': {}
            }
            
            for col in X.columns:
                analysis['characteristics'][col] = {
                    'mean': cluster_data[col].mean(),
                    'median': cluster_data[col].median(),
                    'std': cluster_data[col].std()
                }
            
            segment_name = segment_names.get(cluster_id, f'Cluster {cluster_id}')
            cluster_analysis[segment_name] = analysis
        
        return cluster_analysis
    
    def predict_churn(self, data: pd.DataFrame):
        """Predict churn probability for new customers"""
        if 'churn_prediction' not in self.models:
            print("Churn prediction model not trained")
            return None
        
        # Prepare features (same as training)
        X, _ = self.prepare_features(data)
        
        # Ensure same features as training
        missing_cols = set(self.feature_columns['churn_prediction']) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns['churn_prediction']]
        
        # Scale if needed
        if 'churn_prediction' in self.scalers:
            X_scaled = self.scalers['churn_prediction'].transform(X)
        else:
            X_scaled = X
        
        # Predict
        model = self.models['churn_prediction']
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
            predictions = model.predict(X_scaled)
            
            return {
                'churn_probability': probabilities,
                'churn_prediction': predictions
            }
        else:
            predictions = model.predict(X_scaled)
            return {
                'churn_prediction': predictions
            }
    
    def predict_segments(self, data: pd.DataFrame):
        """Predict customer segments for new customers"""
        if 'customer_segmentation' not in self.models:
            print("Customer segmentation model not trained")
            return None
        
        # Select same features as training
        features = self.feature_columns['customer_segmentation']
        X = data[features].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scalers['customer_segmentation'].transform(X)
        
        # Predict
        segments = self.models['customer_segmentation'].predict(X_scaled)
        
        return segments
    
    def save_all_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            joblib.dump(model, f'{self.models_dir}{model_name}.pkl')
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{self.models_dir}{scaler_name}_scaler.pkl')
        
        # Save feature columns
        with open(f'{self.models_dir}feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print("All models saved successfully!")
    
    def load_models(self):
        """Load saved models"""
        model_files = {
            'churn_prediction': 'churn_predictor.pkl',
            'customer_segmentation': 'customer_segmentation.pkl',
            'clv_prediction': 'clv_predictor.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = f'{self.models_dir}{filename}'
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                
                # Load corresponding scaler
                scaler_file = f'{self.models_dir}{model_name}_scaler.pkl'
                if os.path.exists(scaler_file):
                    self.scalers[model_name] = joblib.load(scaler_file)
        
        # Load feature columns
        feature_file = f'{self.models_dir}feature_columns.pkl'
        if os.path.exists(feature_file):
            with open(feature_file, 'rb') as f:
                self.feature_columns = pickle.load(f)
        
        print(f"Loaded {len(self.models)} models successfully!")
        return len(self.models) > 0

# Convenience functions
def train_models(data: pd.DataFrame, models_dir='models/'):
    """Train all customer analytics models"""
    pipeline = CustomerMLPipeline(models_dir)
    
    results = {}
    
    # Train churn prediction model
    if 'churn' in data.columns or 'churn_probability' in data.columns:
        target_col = 'churn' if 'churn' in data.columns else 'churn_probability'
        results['churn'] = pipeline.train_churn_prediction_model(data, target_col)
    
    # Train segmentation model
    results['segmentation'] = pipeline.train_customer_segmentation_model(data)
    
    # Train CLV model
    if 'customer_lifetime_value' in data.columns:
        results['clv'] = pipeline.train_clv_prediction_model(data)
    
    # Save all models
    pipeline.save_all_models()
    
    return results, pipeline

def load_models(models_dir='models/'):
    """Load saved models"""
    pipeline = CustomerMLPipeline(models_dir)
    success = pipeline.load_models()
    return pipeline if success else None

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(1, 2001)],
        'age': np.random.normal(40, 15, 2000).astype(int),
        'total_spent': np.random.lognormal(7, 1, 2000),
        'total_orders': np.random.poisson(5, 2000) + 1,
        'days_since_last_purchase': np.random.exponential(30, 2000),
        'satisfaction_score': np.random.normal(3.5, 1, 2000),
        'support_tickets': np.random.poisson(1, 2000),
        'churn': np.random.choice([0, 1], 2000, p=[0.8, 0.2]),
        'customer_lifetime_value': np.random.lognormal(8, 0.5, 2000)
    })
    
    # Clip values to realistic ranges
    sample_data['age'] = np.clip(sample_data['age'], 18, 85)
    sample_data['satisfaction_score'] = np.clip(sample_data['satisfaction_score'], 1, 5)
    
    print("Training models on sample data...")
    results, pipeline = train_models(sample_data)
    
    print("\nModel training completed!")
    print(f"Churn model results: {results.get('churn', {}).get('best_model_name', 'Not trained')}")
    print(f"Segmentation model: {results.get('segmentation', {}).get('optimal_k', 'Not trained')} clusters")
    print(f"CLV model R²: {results.get('clv', {}).get('r2_score', 'Not trained')}")