import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib

def load_features(feature_dir: str) -> Dict[str, Any]:
    feature_dir = Path(feature_dir)
    
    # Load dense features
    with np.load(feature_dir / 'dense_features.npz') as data:
        features = {
            'tfidf_features': data['tfidf_features'],
            'phobert_features': data['phobert_features'],
            'viral_scores': data['viral_scores']
        }
    
    # Load metadata
    with np.load(feature_dir / 'metadata.npz') as data:
        features['metadata'] = {key: data[key] for key in data.files}
    
    return features

def prepare_data(features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Combine all features
    X = np.hstack([
        features['tfidf_features'],
        features['phobert_features'],
        np.column_stack([
            features['metadata']['hashtag_counts'],
            features['metadata']['durations'],
            features['metadata']['followers'],
            features['metadata']['views'],
            features['metadata']['likes'],
            features['metadata']['comments'],
            features['metadata']['shares'],
            features['metadata']['saves']
        ])
    ])
    
    # Create labels (viral or not viral)
    y = (features['viral_scores'] > 70).astype(int)  # Threshold at 70 for viral classification
    
    return X, y

def train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'scaler': scaler,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'test_data': (X_test_scaled, y_test),
        'predictions': {
            'rf': (rf_pred, rf_prob),
            'xgb': (xgb_pred, xgb_prob)
        }
    }

def evaluate_models(models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    X_test_scaled, y_test = models['test_data']
    metrics = {}
    
    for model_name, (pred, prob) in models['predictions'].items():
        # Calculate metrics
        metrics[model_name] = {
            'f1': f1_score(y_test, pred),
            'roc_auc': roc_auc_score(y_test, prob),
            'pr_auc': auc(precision_recall_curve(y_test, prob)[1], 
                         precision_recall_curve(y_test, prob)[0])
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f'finalProject/results/{model_name}_confusion_matrix.png')
        plt.close()
    
    return metrics

def save_models(models: Dict[str, Any], output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models and scaler
    joblib.dump(models['scaler'], output_dir / 'scaler.joblib')
    joblib.dump(models['rf_model'], output_dir / 'random_forest_model.joblib')
    joblib.dump(models['xgb_model'], output_dir / 'xgboost_model.joblib')

def main():
    # Load features
    features = load_features('finalProject/data/features')
    
    # Prepare data
    X, y = prepare_data(features)
    
    # Train models
    print("Training models...")
    models = train_models(X, y)
    
    # Evaluate models
    print("Evaluating models...")
    metrics = evaluate_models(models)
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Metrics:")
        for metric_name, value in model_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Save models
    print("\nSaving models...")
    save_models(models, 'finalProject/models')
    print("Done.")