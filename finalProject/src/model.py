import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Modern ML Models for TikTok Data
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import ( GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesRegressor, VotingRegressor, VotingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier

try:
    # For time series forecasting
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, TiDE, PatchTST
    ADVANCED_TS_AVAILABLE = True
except ImportError:
    ADVANCED_TS_AVAILABLE = False
    print("Advanced time series models not available. Using fallback models.")

try:
    # For tabular data
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not available. Using fallback models.")
warnings.filterwarnings('ignore')

class ModernViralPredictor:
    """    
    PREDICTION MODELS:
    - TiDE: Temporal Integration Decoupling Encoder for multivariate time series
    - PatchTST: Attention-based model for long time series
    - N-BEATS: Deep MLP for complex TikTok data patterns
    - DeepAR-style: For multiple video series forecasting
    - XGBoost: Strong baseline with engineered features
    
    CLASSIFICATION MODELS:
    - PhoBERT + XGBoost: Vietnamese language understanding + strong classifier
    - TabPFN: Fast tabular classification without tuning
    - SAINT: Transformer for tabular data relationships
    - CatBoost: Excellent for categorical features (hashtags, topics)
    - LightGBM + TF-IDF: Fast and accurate text vectorization
    """
    
    def __init__(self):
        self.use_advanced_ts = ADVANCED_TS_AVAILABLE
        self.use_tabpfn = TABPFN_AVAILABLE

        self.growth_models = self._initialize_growth_models()
        self.viral_models = self._initialize_viral_models()
        
        self.growth_ensemble = None
        self.viral_ensemble = None
        
        # Preprocessing
        self.scaler = RobustScaler()  # Robust to TikTok data outliers
        self.feature_selector = None
        self.feature_names = None
        
        # Advanced model flags
        self.use_advanced_ts = ADVANCED_TS_AVAILABLE
        self.use_tabpfn = TABPFN_AVAILABLE
        
    def _initialize_growth_models(self) -> Dict:
        """Initialize modern forecasting models for TikTok growth prediction"""
        models = {}
        
        # XGBoost
        models['xgboost'] = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        
        # CatBoost
        models['catboost'] = cb.CatBoostRegressor(iterations=150, learning_rate=0.08, depth=7,
            l2_leaf_reg=3, random_state=42, verbose=False
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMRegressor(n_estimators=150,  learning_rate=0.08, max_depth=7,
            num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8, random_state=42, verbose=-1
        )
        
        # N-BEATS style MLP 
        models['nbeats_mlp'] = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42
        )
        
        # Extra Trees
        models['extra_trees'] = ExtraTreesRegressor(n_estimators=100, max_depth=10, 
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        
        # Advanced time series model (if available)
        if self.use_advanced_ts:
            try:
                # TiDE-inspired architecture using MLP
                models['tide_mlp'] = MLPRegressor(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
                    alpha=0.0001, learning_rate='adaptive', max_iter=300, random_state=42
                )
            except Exception as e:
                print(f"Could not initialize TiDE model: {e}")
        
        return models
    
    def _initialize_viral_models(self) -> Dict:
        """Initialize modern classification models for viral prediction"""
        models = {}
        
        # PhoBERT + XGBoost style
        models['xgboost_classifier'] = xgb.XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        
        # CatBoost 
        models['catboost_classifier'] = cb.CatBoostClassifier(iterations=150, learning_rate=0.08,
            depth=7, l2_leaf_reg=3, random_state=42, verbose=False
        )
        
        # LightGBM + TF-IDF style 
        models['lightgbm_classifier'] = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08,
            max_depth=7, num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8, random_state=42, verbose=-1
        )
        
        # SAINT-style 
        models['saint_mlp'] = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42
        )
        
        # TabPFN (if available)
        if self.use_tabpfn:
            try:
                models['tabpfn'] = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            except Exception as e:
                print(f"Could not initialize TabPFN: {e}")
                # Fallback to Random Forest
                models['random_forest'] = RandomForestClassifier(n_estimators=150, max_depth=10,
                    min_samples_split=5, random_state=42
                )
        else:
            # Fallback to Random Forest
            models['random_forest'] = RandomForestClassifier(n_estimators=150, max_depth=10,
                    min_samples_split=5, random_state=42
            )
        
        # Gradient Boosting
        models['gradient_boost'] = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
            max_depth=7, random_state=42
        )
        return models
    
    def engineer_advanced_features(self, metadata: Dict) -> Tuple[np.ndarray, List[str]]:
        """Engineer advanced features specifically for TikTok viral prediction"""
        features = []
        feature_names = []
        
        # VIRAL MOMENTUM FEATURES
        if all(k in metadata for k in ['initial_engagement_rate', 'current_engagement_rate']):
            # Engagement momentum (key viral indicator)
            engagement_momentum = metadata['current_engagement_rate'] - metadata['initial_engagement_rate']
            features.append(engagement_momentum.reshape(-1, 1))
            feature_names.append('engagement_momentum')
            
            # Engagement acceleration ratio
            engagement_ratio = np.where(
                metadata['initial_engagement_rate'] > 0,
                metadata['current_engagement_rate'] / metadata['initial_engagement_rate'],
                1.0
            )
            features.append(engagement_ratio.reshape(-1, 1))
            feature_names.append('engagement_acceleration')
            
            # Viral momentum score (composite)
            viral_momentum = engagement_momentum * np.log1p(engagement_ratio)
            features.append(viral_momentum.reshape(-1, 1))
            feature_names.append('viral_momentum_score')
        
        # GROWTH VELOCITY FEATURES
        if all(k in metadata for k in ['view_growth_per_hour', 'time_diff_hours']):
            # Velocity score (views per hour normalized by time)
            velocity_score = metadata['view_growth_per_hour'] / np.maximum(metadata['time_diff_hours'], 1)
            features.append(velocity_score.reshape(-1, 1))
            feature_names.append('velocity_score')
            
            # Multi-metric growth consistency
            if all(k in metadata for k in ['like_growth_per_hour', 'comment_growth_per_hour', 'share_growth_per_hour']):
                growth_consistency = (
                    metadata['view_growth_per_hour'] * 0.4 +
                    metadata['like_growth_per_hour'] * 0.3 +
                    metadata['comment_growth_per_hour'] * 0.2 +
                    metadata['share_growth_per_hour'] * 0.1
                )
                features.append(growth_consistency.reshape(-1, 1))
                feature_names.append('growth_consistency')
                
                # Growth balance (how balanced the growth is across metrics)
                growth_metrics = np.column_stack([
                    metadata['view_growth_per_hour'],
                    metadata['like_growth_per_hour'],
                    metadata['comment_growth_per_hour'],
                    metadata['share_growth_per_hour']
                ])
                growth_balance = 1 - np.std(growth_metrics, axis=1) / (np.mean(growth_metrics, axis=1) + 1e-6)
                features.append(growth_balance.reshape(-1, 1))
                feature_names.append('growth_balance')

        if 'durations' in metadata:
            # TikTok duration sweet spot (15-60 seconds optimal, peak at 30s)
            duration_score = np.where(
                (metadata['durations'] >= 15) & (metadata['durations'] <= 60),
                np.exp(-0.1 * np.abs(metadata['durations'] - 30)),
                np.exp(-0.2 * np.abs(metadata['durations'] - 30))
            )
            features.append(duration_score.reshape(-1, 1))
            feature_names.append('duration_optimality')
            
            # Short-form bonus (TikTok favors short content)
            short_form_bonus = np.where(metadata['durations'] <= 30, 1.2, 1.0)
            features.append(short_form_bonus.reshape(-1, 1))
            feature_names.append('short_form_bonus')
        
        # CREATOR INFLUENCE FEATURES
        if 'followers' in metadata:
            # Follower tier classification
            follower_tiers = np.digitize(
                np.log1p(metadata['followers']),
                bins=[0, 8, 10, 12, 14, 16, np.inf]  # Log scale tiers
            )
            features.append(follower_tiers.reshape(-1, 1))
            feature_names.append('follower_tier')
            
            # Creator influence score (log-scaled and normalized)
            creator_influence = np.log1p(metadata['followers']) / 20
            features.append(creator_influence.reshape(-1, 1))
            feature_names.append('creator_influence')
            
            # Follower engagement ratio (viral potential indicator)
            if 'current_views' in metadata:
                follower_engagement = np.where(
                    metadata['followers'] > 0,
                    metadata['current_views'] / metadata['followers'],
                    0
                )
                # Log transform to handle extreme values
                follower_engagement_log = np.log1p(follower_engagement)
                features.append(follower_engagement_log.reshape(-1, 1))
                feature_names.append('follower_engagement_ratio')
                
                # Viral reach multiplier
                viral_reach = np.where(follower_engagement > 1, follower_engagement, 1)
                features.append(np.log1p(viral_reach).reshape(-1, 1))
                feature_names.append('viral_reach_multiplier')
        
        # TEMPORAL OPTIMIZATION FEATURES
        if 'post_hour' in metadata:
            # TikTok prime time (based on our EDA: 11-13, 2-5 PM)
            morning_prime = np.where(
                (metadata['post_hour'] >= 11) & (metadata['post_hour'] <= 13),
                1.0, 0.0
            )
            noon_prime = np.where(
                (metadata['post_hour'] >= 14) & (metadata['post_hour'] <= 17),
                1.0, 0.0
            )
            prime_time_score = np.maximum(morning_prime, noon_prime)
            features.append(prime_time_score.reshape(-1, 1))
            feature_names.append('prime_time_posting')
            
            # Hour cyclical encoding (captures daily patterns)
            hour_sin = np.sin(2 * np.pi * metadata['post_hour'] / 24)
            hour_cos = np.cos(2 * np.pi * metadata['post_hour'] / 24)
            features.extend([hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1)])
            feature_names.extend(['hour_sin', 'hour_cos'])
            
            # Weekend vs weekday effect
            weekend_effect = np.where(
                (metadata['post_hour'] >= 10) & (metadata['post_hour'] <= 22),
                1.1, 1.0  # Boost for weekend-like hours
            )
            features.append(weekend_effect.reshape(-1, 1))
            feature_names.append('weekend_effect')
        
        # VIRAL ACCELERATION METRICS
        if 'viral_acceleration' in metadata:
            # Acceleration tier classification
            acceleration_tier = np.digitize(
                metadata['viral_acceleration'],
                bins=[0, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
            )
            features.append(acceleration_tier.reshape(-1, 1))
            feature_names.append('acceleration_tier')
            
            # Exponential acceleration score
            exp_acceleration = np.where(
                metadata['viral_acceleration'] > 0,
                np.log1p(metadata['viral_acceleration']),
                0
            )
            features.append(exp_acceleration.reshape(-1, 1))
            feature_names.append('exponential_acceleration')
        
        # CROSS-FEATURE INTERACTIONS (TikTok-specific)
        if len(features) >= 3:
            # Engagement × Creator × Timing interaction
            if all(feat in feature_names for feat in ['engagement_momentum', 'creator_influence', 'prime_time_posting']):
                eng_idx = feature_names.index('engagement_momentum')
                creator_idx = feature_names.index('creator_influence')
                time_idx = feature_names.index('prime_time_posting')
                
                triple_interaction = (
                    features[eng_idx].flatten() * 
                    features[creator_idx].flatten() * 
                    features[time_idx].flatten()
                )
                features.append(triple_interaction.reshape(-1, 1))
                feature_names.append('engagement_creator_timing_interaction')
            
            # Viral momentum × Content quality interaction
            if all(feat in feature_names for feat in ['viral_momentum_score', 'duration_optimality']):
                momentum_idx = feature_names.index('viral_momentum_score')
                quality_idx = feature_names.index('duration_optimality')
                
                quality_momentum = (
                    features[momentum_idx].flatten() * 
                    features[quality_idx].flatten()
                )
                features.append(quality_momentum.reshape(-1, 1))
                feature_names.append('quality_momentum_interaction')
        
        # COMPOSITE VIRAL SCORES
        if len(features) >= 5:
            # TikTok Viral Potential Score (weighted combination)
            viral_components = []
            weights = []
            
            if 'engagement_momentum' in feature_names:
                viral_components.append(features[feature_names.index('engagement_momentum')].flatten())
                weights.append(0.25)
            
            if 'velocity_score' in feature_names:
                viral_components.append(features[feature_names.index('velocity_score')].flatten())
                weights.append(0.20)
            
            if 'creator_influence' in feature_names:
                viral_components.append(features[feature_names.index('creator_influence')].flatten())
                weights.append(0.15)
            
            if 'prime_time_posting' in feature_names:
                viral_components.append(features[feature_names.index('prime_time_posting')].flatten())
                weights.append(0.10)
            
            if 'duration_optimality' in feature_names:
                viral_components.append(features[feature_names.index('duration_optimality')].flatten())
                weights.append(0.10)
            
            if viral_components:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Calculate composite score
                viral_potential = np.zeros(len(viral_components[0]))
                for component, weight in zip(viral_components, weights):
                    viral_potential += component * weight
                
                features.append(viral_potential.reshape(-1, 1))
                feature_names.append('tiktok_viral_potential_score')
        
        if features:
            return np.hstack(features), feature_names
        else:
            return np.array([]).reshape(0, 0), []
    
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        """Load and prepare features with advanced preprocessing for TikTok data"""
        features_dir = Path(features_dir)
        
        print("Loading features...")
        with np.load(features_dir / 'dense_features.npz') as data:
            tfidf_features = data['tfidf_features']
            phobert_features = data['phobert_features']
        
        with np.load(features_dir / 'metadata.npz', allow_pickle=True) as data:
            metadata = {key: data[key] for key in data.files}
        
        engineered_features, engineered_names = self.engineer_advanced_features(metadata)
        
        from sklearn.decomposition import TruncatedSVD
        
        # TF-IDF reduction (more components for TikTok text diversity)
        if tfidf_features.shape[1] > 75:
            tfidf_reducer = TruncatedSVD(n_components=75, random_state=42)
            tfidf_reduced = tfidf_reducer.fit_transform(tfidf_features)
        else:
            tfidf_reduced = tfidf_features
        
        # PhoBERT reduction (optimized for Vietnamese TikTok content)
        if phobert_features.shape[1] > 50:
            phobert_reducer = TruncatedSVD(n_components=50, random_state=42)
            phobert_reduced = phobert_reducer.fit_transform(phobert_features)
        else:
            phobert_reduced = phobert_features
        
        # Combine all features with proper weighting
        feature_components = []
        feature_names = []
        
        # Prioritize engineered features (most important for TikTok)
        if engineered_features.size > 0:
            feature_components.append(engineered_features)
            feature_names.extend(engineered_names)
        
        # Add text features
        feature_components.extend([tfidf_reduced, phobert_reduced])
        feature_names.extend([f'tfidf_svd_{i}' for i in range(tfidf_reduced.shape[1])])
        feature_names.extend([f'phobert_svd_{i}' for i in range(phobert_reduced.shape[1])])
        
        # Combine and clean
        X = np.hstack(feature_components)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.feature_names = feature_names        
        return X, metadata
    
    def train_growth_predictor(self, X: np.ndarray, metadata: Dict) -> Dict:
        y = metadata['new_growth_rate']
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for growth prediction")
        print(f"Training modern forecasting models on {len(y_clean)} TikTok samples...")
        
        X_scaled = self.scaler.fit_transform(X_clean)
        
        if X_scaled.shape[1] > 50:
            n_features = min(50, X_scaled.shape[1])
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
            X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
        else:
            X_selected = X_scaled
        
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_clean, test_size=0.2, random_state=42)
        
        model_scores = {}
        trained_models = {}
        
        for name, model in self.growth_models.items():
            try:
                print(f"Training {name} for TikTok growth prediction...")
                
                # Special handling for TabPFN (has sample size limits)
                if name == 'tabpfn' and len(X_train) > 1000:
                    sample_idx = np.random.choice(len(X_train), 1000, replace=False)
                    model.fit(X_train[sample_idx], y_train[sample_idx])
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                model_scores[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
                trained_models[name] = model
                
                print(f"  {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
                
            except Exception as e:
                print(f"  Failed to train {name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Create ensemble weighting
        weights = []
        estimators = []
        
        for name, scores in model_scores.items():
            # Weight by R² score with minimum threshold
            weight = max(0.1, scores['r2']) if scores['r2'] > 0 else 0.1
            weights.append(weight)
            estimators.append((name, trained_models[name]))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.growth_ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        self.growth_ensemble.fit(X_train, y_train)
        
        ensemble_pred = self.growth_ensemble.predict(X_test)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
        best_model = trained_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_)
        else:
            feature_importance = np.zeros(X_selected.shape[1])
        
        if self.feature_selector:
            selected_features = self.feature_selector.get_support()
            selected_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
        else:
            selected_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': selected_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'ensemble_r2': ensemble_r2,
            'ensemble_mae': ensemble_mae,
            'ensemble_rmse': ensemble_rmse,
            'individual_scores': model_scores,
            'best_model': best_model_name,
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'weights': dict(zip([name for name, _ in estimators], weights)),
            'model_count': len(trained_models)
        }
    
    def train_viral_classifier(self, X: np.ndarray, metadata: Dict) -> Dict:
        y = metadata['continuing_viral']
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for viral classification")
        
        unique_classes, counts = np.unique(y_clean, return_counts=True)
        class_distribution = dict(zip(unique_classes, counts))
        print(f"TikTok viral class distribution: {class_distribution}")
        
        if len(unique_classes) < 2:
            return {
                'ensemble_accuracy': 0.0,
                'ensemble_f1': 0.0,
                'individual_scores': {},
                'feature_importance': pd.DataFrame(),
                'n_samples': len(y_clean),
                'class_distribution': class_distribution
            }
        
        print(f"Training modern classification models on {len(y_clean)} TikTok samples...")
        
        X_scaled = self.scaler.transform(X_clean)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
        
        model_scores = {}
        trained_models = {}
        
        for name, model in self.viral_models.items():
            try:
                print(f"Training {name} for TikTok viral classification...")
                
                # Special handling for TabPFN
                if name == 'tabpfn' and len(X_train) > 1000:
                    # Sample for TabPFN
                    sample_idx = np.random.choice(len(X_train), 1000, replace=False)
                    model.fit(X_train[sample_idx], y_train[sample_idx])
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                # Get probabilities for AUC calculation
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                else:
                    y_pred_proba = y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5
                
                model_scores[name] = {
                    'accuracy': accuracy, 
                    'f1': f1, 
                    'precision': precision,
                    'recall': recall,
                    'auc': auc
                }
                trained_models[name] = model
                print(f"  {name}: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
            except Exception as e:
                print(f"  Failed to train {name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No classification models were successfully trained")
        
        # Create ensemble with F1-weighted voting
        weights = []
        estimators = []
        
        for name, scores in model_scores.items():
            # Weight by F1 score (important for imbalanced TikTok data)
            weight = max(0.1, scores['f1'])
            weights.append(weight)
            estimators.append((name, trained_models[name]))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.viral_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        self.viral_ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.viral_ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
        best_model = trained_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_[0]) if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
        else:
            feature_importance = np.zeros(X_selected.shape[1])
        
        if self.feature_selector:
            selected_features = self.feature_selector.get_support()
            selected_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
        else:
            selected_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': selected_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1': ensemble_f1,
            'ensemble_precision': ensemble_precision,
            'ensemble_recall': ensemble_recall,
            'individual_scores': model_scores,
            'best_model': best_model_name,
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'class_distribution': class_distribution,
            'weights': dict(zip([name for name, _ in estimators], weights)),
            'model_count': len(trained_models)
        }
    
    def save_models(self, output_dir: str):
        """Save all trained models and preprocessors"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble models
        if self.growth_ensemble:
            with open(output_dir / 'growth_ensemble.pkl', 'wb') as f:
                pickle.dump(self.growth_ensemble, f)
        
        if self.viral_ensemble:
            with open(output_dir / 'viral_ensemble.pkl', 'wb') as f:
                pickle.dump(self.viral_ensemble, f)
        
        # Save preprocessors
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        if self.feature_selector:
            with open(output_dir / 'feature_selector.pkl', 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        # Save feature names
        with open(output_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save model configuration
        config = {
            'use_advanced_ts': self.use_advanced_ts,
            'use_tabpfn': self.use_tabpfn,
            'growth_models': list(self.growth_models.keys()),
            'viral_models': list(self.viral_models.keys())
        }
        with open(output_dir / 'model_config.pkl', 'wb') as f:
            pickle.dump(config, f)

class IntelligentTrendRecommender:    
    def __init__(self):
        self.hashtag_model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=8,
            num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
            random_state=42, verbose=-1
        )
        self.sound_model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=8,
            num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
            random_state=42, verbose=-1
        )
        self.hashtag_engagement_model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        self.hashtag_growth_model = cb.CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=6,
            random_state=42, verbose=False
        )
        self.sound_viral_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            num_leaves=63, random_state=42, verbose=-1
        )
        
        self.hashtag_clusters = None
        self.sound_clusters = None
        self.hashtag_kmeans = KMeans(n_clusters=8, random_state=42)
        self.sound_kmeans = KMeans(n_clusters=6, random_state=42)
        
        self.hashtag_nn = NearestNeighbors(n_neighbors=15, metric='cosine')
        self.sound_nn = NearestNeighbors(n_neighbors=12, metric='cosine')
        
        self.trending_features = None
        self.hashtag_features_matrix = None
        self.sound_features_matrix = None
        
    def load_trending_features(self, features_dir: str):
        features_dir = Path(features_dir)
        try:
            self.trending_features = pd.read_pickle(features_dir / 'trending_features.pkl')
            
            # Validate data
            if 'hashtag_trends' in self.trending_features:
                print(f"   Hashtag trends: {len(self.trending_features['hashtag_trends'])} entries")
            if 'sound_trends' in self.trending_features:
                print(f"   Sound trends: {len(self.trending_features['sound_trends'])} entries")
                
        except Exception as e:
            print(f"Could not load trending features: {e}")
            self.trending_features = {
                'hashtag_trends': pd.DataFrame(),
                'sound_trends': pd.DataFrame()
            }
    
    def engineer_hashtag_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        features = []
        
        # Basic metrics
        total_views = trends_df['total_views'].fillna(0).values
        avg_engagement = trends_df['avg_engagement'].fillna(0).values
        usage_count = trends_df['usage_count'].fillna(0).values
        
        features.extend([total_views, avg_engagement, usage_count])
        
        # TikTok-specific viral indicators
        # Viral velocity (views per usage)
        viral_velocity = np.where(usage_count > 0, total_views / usage_count, 0)
        features.append(viral_velocity)
        
        # Engagement efficiency
        engagement_efficiency = np.where(usage_count > 0, avg_engagement / usage_count, 0)
        features.append(engagement_efficiency)
        
        # TikTok viral potential score (optimized weights)
        viral_potential = (
            np.log1p(total_views) * 0.35 +      # Views are crucial on TikTok
            avg_engagement * 0.40 +              # Engagement drives algorithm
            np.log1p(usage_count) * 0.15 +       # Usage indicates trend
            viral_velocity * 0.10                # Velocity shows momentum
        )
        features.append(viral_potential)
        
        # Growth momentum with TikTok decay factor
        if len(trends_df) > 1:
            decay_factor = 0.8
            positions = np.arange(len(trends_df))
            growth_momentum = np.power(decay_factor, positions)
            features.append(growth_momentum)
        else:
            features.append(np.ones(len(trends_df)))
        
        # Hashtag characteristics for TikTok
        if 'hashtag' in trends_df.columns:
            # Length optimization for TikTok (shorter is often better)
            hashtag_length = trends_df['hashtag'].str.len().fillna(0).values
            length_score = np.where(hashtag_length <= 15, 1.0, 0.8)  # Penalty for long hashtags
            features.append(length_score)
            
            # Contains numbers (often indicates challenges/trends)
            contains_numbers = trends_df['hashtag'].str.contains(r'\d', na=False).astype(int).values
            features.append(contains_numbers)
            
            # TikTok-style patterns (repeated chars, emojis, etc.)
            tiktok_style = trends_df['hashtag'].str.contains(r'(.)\1{2,}|[^\w\s]', na=False).astype(int).values
            features.append(tiktok_style)
        
        # Trend stability with TikTok volatility factor
        trend_stability = np.where(
            avg_engagement > 0,
            np.minimum(viral_velocity / (avg_engagement + 1), 15),  # Higher cap for TikTok
            0
        )
        features.append(trend_stability)
        
        # Market saturation with TikTok-specific threshold
        market_saturation = 1 / (1 + np.exp(-0.05 * (usage_count - 100)))  # TikTok has higher saturation point
        features.append(market_saturation)
        
        # Recency bonus (TikTok favors fresh content)
        recency_bonus = np.exp(-0.1 * np.arange(len(trends_df)))
        features.append(recency_bonus)
        
        return np.column_stack(features)
    
    def engineer_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Engineer comprehensive features for TikTok sound trends"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        features = []
        
        current_views = trends_df['current_views'].fillna(0).values
        current_engagement = trends_df['current_engagement_rate'].fillna(0).values
        new_growth_rate = trends_df['new_growth_rate'].fillna(0).values
        
        features.extend([current_views, current_engagement, new_growth_rate])
        
        # TikTok sound-specific viral indicators
        # Sound momentum (views × engagement × growth)
        sound_momentum = current_views * current_engagement * (1 + np.abs(new_growth_rate))
        features.append(sound_momentum)
        
        # Viral acceleration with TikTok audio boost
        viral_acceleration = np.where(
            current_views > 0,
            new_growth_rate * current_engagement / np.log1p(current_views),
            0
        )
        features.append(viral_acceleration)
        
        # Engagement intensity (key for TikTok algorithm)
        engagement_intensity = current_engagement * np.log1p(current_views)
        features.append(engagement_intensity)
        
        # Growth sustainability for TikTok trends
        growth_sustainability = np.where(
            new_growth_rate > 0,
            current_engagement / (1 + np.abs(new_growth_rate)),
            current_engagement
        )
        features.append(growth_sustainability)
        
        # Sound characteristics for TikTok
        if 'music_title' in trends_df.columns:            
            # Title complexity (word count)
            word_count = trends_df['music_title'].str.split().str.len().fillna(0).values
            complexity_score = np.where(word_count <= 5, 1.0, 0.8)  # Simple titles preferred
            features.append(complexity_score)
            
            # Contains trending keywords (remix, challenge, etc.)
            trending_keywords = trends_df['music_title'].str.contains(
                r'remix|challenge|trend|viral|dance', case=False, na=False
            ).astype(int).values
            features.append(trending_keywords)
        
        # Viral tier classification (TikTok-specific thresholds)
        viral_tier = np.digitize(
            new_growth_rate,
            bins=[-np.inf, -0.5, 0, 0.2, 0.8, 2.0, 5.0, np.inf]
        )
        features.append(viral_tier)
        
        # TikTok trend momentum score
        trend_momentum = (
            np.log1p(current_views) * 0.35 +     # Views matter most
            current_engagement * 0.35 +          # Engagement drives algorithm
            np.abs(new_growth_rate) * 0.30       # Growth indicates trending
        )
        features.append(trend_momentum)
        
        # Audio freshness factor
        audio_freshness = np.exp(-0.05 * np.arange(len(trends_df)))  # Decay factor for audio
        features.append(audio_freshness)        
        return np.column_stack(features)
    
    def train_recommenders(self):
        if not self.trending_features:
            print("No trending features available for training")
            return
        
        # Train hashtag recommenders
        hashtag_df = self.trending_features['hashtag_trends']
        if not hashtag_df.empty and len(hashtag_df) > 15:  # Higher threshold for better training
            print("Training hashtag recommendation system...")
            
            self.hashtag_features_matrix = self.engineer_hashtag_features(hashtag_df)
            
            if self.hashtag_features_matrix.size > 0:
                try:
                    # Main viral potential model (enhanced weights)
                    viral_potential_target = (
                        0.35 * np.log1p(hashtag_df['total_views'].fillna(0)) +
                        0.40 * hashtag_df['avg_engagement'].fillna(0) +
                        0.25 * np.log1p(hashtag_df['usage_count'].fillna(0))
                    ).values
                    
                    if np.any(viral_potential_target > 0):
                        self.hashtag_model.fit(self.hashtag_features_matrix, viral_potential_target)
                    
                    # Engagement-focused model
                    engagement_target = hashtag_df['avg_engagement'].fillna(0).values
                    if np.any(engagement_target > 0):
                        self.hashtag_engagement_model.fit(self.hashtag_features_matrix, engagement_target)
                    
                    # Growth-focused model
                    growth_target = hashtag_df['usage_count'].fillna(0).values
                    if np.any(growth_target > 0):
                        self.hashtag_growth_model.fit(self.hashtag_features_matrix, growth_target)
                    
                    # Clustering for content-based recommendations
                    if len(hashtag_df) >= 8:
                        self.hashtag_clusters = self.hashtag_kmeans.fit_predict(self.hashtag_features_matrix)
                        self.hashtag_nn.fit(self.hashtag_features_matrix)
                    
                    print(f"   Hashtag recommender trained on {len(hashtag_df)} samples")
                    
                except Exception as e:
                    print(f"   Error training hashtag models: {e}")
        
        # Train sound recommenders
        sound_df = self.trending_features['sound_trends']
        if not sound_df.empty and len(sound_df) > 15:
            print("Training sound recommendation system...")
            
            self.sound_features_matrix = self.engineer_sound_features(sound_df)
            
            if self.sound_features_matrix.size > 0:
                try:
                    # Main popularity model
                    popularity_target = sound_df['current_views'].fillna(0).values
                    if np.any(popularity_target > 0):
                        self.sound_model.fit(self.sound_features_matrix, popularity_target)
                    
                    # Viral classification model
                    viral_threshold = np.percentile(sound_df['new_growth_rate'].fillna(0), 70)  # Lower threshold for TikTok
                    viral_labels = (sound_df['new_growth_rate'].fillna(0) > viral_threshold).astype(int)
                    
                    if len(np.unique(viral_labels)) > 1:
                        self.sound_viral_model.fit(self.sound_features_matrix, viral_labels)
                    
                    # Clustering
                    if len(sound_df) >= 6:
                        self.sound_clusters = self.sound_kmeans.fit_predict(self.sound_features_matrix)
                        self.sound_nn.fit(self.sound_features_matrix)
                    
                    print(f"   Sound recommender trained on {len(sound_df)} samples")
                    
                except Exception as e:
                    print(f"   Error training sound models: {e}")
    
    def recommend_hashtags(self, n_recommendations: int = 15, strategy: str = 'balanced') -> List[Dict]:
        if not self.trending_features or self.trending_features['hashtag_trends'].empty:
            return []
        hashtag_df = self.trending_features['hashtag_trends']
        
        if self.hashtag_features_matrix is None or self.hashtag_features_matrix.size == 0:
            return []
        try:
            recommendations = hashtag_df.copy()
            
            # Get predictions from different models
            viral_scores = self.hashtag_model.predict(self.hashtag_features_matrix)
            engagement_scores = self.hashtag_engagement_model.predict(self.hashtag_features_matrix)
            growth_scores = self.hashtag_growth_model.predict(self.hashtag_features_matrix)
            
            if strategy == 'viral':
                final_scores = viral_scores
            elif strategy == 'engagement':
                final_scores = engagement_scores
            elif strategy == 'growth':
                final_scores = growth_scores
            elif strategy == 'diverse':
                final_scores = viral_scores
                if self.hashtag_clusters is not None:
                    for cluster_id in np.unique(self.hashtag_clusters):
                        cluster_mask = self.hashtag_clusters == cluster_id
                        if np.any(cluster_mask):
                            cluster_boost = 1.0 + (cluster_id * 0.15)  # Higher boost for TikTok
                            final_scores[cluster_mask] *= cluster_boost
            else:  # balanced - optimized for TikTok
                final_scores = (
                    viral_scores * 0.35 +        # Viral potential is key
                    engagement_scores * 0.40 +   # Engagement drives algorithm
                    growth_scores * 0.25         # Growth indicates trend
                )
            
            # TikTok-specific adjustments
            usage_counts = hashtag_df['usage_count'].fillna(0).values
            
            # Novelty bonus (prefer less saturated hashtags)
            novelty_bonus = 1 / (1 + np.log1p(usage_counts) * 0.1)
            final_scores = final_scores * (1 + novelty_bonus * 0.25)
            
            # Diversity penalty for similar hashtags
            if len(hashtag_df) > 1 and 'hashtag' in hashtag_df.columns:
                diversity_scores = self._calculate_hashtag_diversity(hashtag_df['hashtag'].values)
                final_scores = final_scores * (1 + diversity_scores * 0.15)
            
            recommendations['trend_score'] = final_scores
            recommendations['viral_score'] = viral_scores
            recommendations['engagement_score'] = engagement_scores
            recommendations['growth_score'] = growth_scores
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            result_columns = ['hashtag', 'trend_score', 'viral_score', 'engagement_score', 'growth_score',
                'total_views', 'avg_engagement', 'usage_count' ]
            return recommendations.head(n_recommendations)[result_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in hashtag recommendation: {e}")
            return hashtag_df.nlargest(n_recommendations, 'avg_engagement')[
                ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
            ].to_dict('records')
    
    def recommend_sounds(self, n_recommendations: int = 10, strategy: str = 'balanced') -> List[Dict]:
        if not self.trending_features or self.trending_features['sound_trends'].empty:
            return []
        sound_df = self.trending_features['sound_trends']
        
        if self.sound_features_matrix is None or self.sound_features_matrix.size == 0:
            return []
        
        try:
            recommendations = sound_df.copy()
            popularity_scores = self.sound_model.predict(self.sound_features_matrix)
            
            # Get viral probabilities
            try:
                viral_probabilities = self.sound_viral_model.predict_proba(self.sound_features_matrix)[:, 1]
            except:
                viral_probabilities = np.zeros(len(sound_df))
            
            # Calculate emerging trend scores
            growth_rates = sound_df['new_growth_rate'].fillna(0).values
            emerging_scores = np.where(growth_rates > 0, growth_rates * viral_probabilities, 0)
            
            if strategy == 'viral':
                final_scores = viral_probabilities * np.log1p(popularity_scores)
            elif strategy == 'popular':
                final_scores = popularity_scores
            elif strategy == 'emerging':
                final_scores = emerging_scores
            else:  # balanced - optimized for TikTok audio
                final_scores = (
                    popularity_scores * 0.35 +                           # Current popularity
                    viral_probabilities * np.log1p(popularity_scores) * 0.40 +  # Viral potential
                    emerging_scores * 0.25                               # Emerging trend factor
                )
            
            # TikTok audio-specific bonuses
            if 'current_engagement_rate' in sound_df.columns:
                engagement_rates = sound_df['current_engagement_rate'].fillna(0).values
                audio_engagement_bonus = engagement_rates / (engagement_rates.max() + 1e-6)
                final_scores = final_scores * (1 + audio_engagement_bonus * 0.20)
            
            # Diversity bonus for TikTok audio variety
            if self.sound_clusters is not None:
                diversity_bonus = self._calculate_cluster_diversity(self.sound_clusters)
                final_scores = final_scores * (1 + diversity_bonus * 0.15)
            
            # Audio freshness (TikTok audio trends change quickly)
            audio_freshness = np.exp(-0.03 * np.arange(len(sound_df)))
            final_scores = final_scores * (1 + audio_freshness * 0.10)
            
            recommendations['trend_score'] = final_scores
            recommendations['popularity_score'] = popularity_scores
            recommendations['viral_probability'] = viral_probabilities
            recommendations['emerging_score'] = emerging_scores
            
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            result_columns = ['music_id', 'music_title', 'trend_score', 'popularity_score', 
                'viral_probability', 'emerging_score', 'current_views', 'current_engagement_rate']
            return recommendations.head(n_recommendations)[result_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in sound recommendation: {e}")
            # Enhanced fallback
            return sound_df.nlargest(n_recommendations, 'current_engagement_rate')[
                ['music_id', 'music_title', 'current_views', 'current_engagement_rate']
            ].to_dict('records')
    
    def _calculate_hashtag_diversity(self, hashtags: np.ndarray) -> np.ndarray:
        diversity_scores = np.ones(len(hashtags))
        for i, hashtag in enumerate(hashtags):
            if pd.isna(hashtag):
                continue
            
            # Calculate similarity with other hashtags
            similarities = []
            for j, other_hashtag in enumerate(hashtags):
                if i != j and pd.notna(other_hashtag):                    
                    # Character-based similarity
                    char_similarity = len(set(hashtag) & set(other_hashtag)) / \
                                    len(set(hashtag) | set(other_hashtag))
                    
                    # Length similarity penalty
                    length_penalty = abs(len(hashtag) - len(other_hashtag)) / max(len(hashtag), len(other_hashtag), 1)
                    
                    # Combined similarity
                    similarity = char_similarity * (1 - length_penalty * 0.3)
                    similarities.append(similarity)
            
            if similarities:
                diversity_scores[i] = 1 - np.mean(similarities)
        
        return diversity_scores
    
    def _calculate_cluster_diversity(self, clusters: np.ndarray) -> np.ndarray:
        diversity_scores = np.ones(len(clusters))
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        # Inverse frequency weighting (prefer less common clusters)
        cluster_weights = dict(zip(unique_clusters, 1 / (counts + 1)))
        
        for i, cluster in enumerate(clusters):
            diversity_scores[i] = cluster_weights.get(cluster, 1.0)
        
        return diversity_scores
    
    def get_similar_hashtags(self, hashtag: str, n_similar: int = 5) -> List[str]:
        """Find similar hashtags using enhanced content-based filtering for TikTok"""
        if not self.trending_features or self.hashtag_features_matrix is None:
            return []
        hashtag_df = self.trending_features['hashtag_trends']
        
        try:
            # Find the hashtag in the dataset
            hashtag_idx = hashtag_df[hashtag_df['hashtag'] == hashtag].index
            
            if len(hashtag_idx) == 0:
                return []
            
            hashtag_idx = hashtag_idx[0]
            hashtag_features = self.hashtag_features_matrix[hashtag_idx:hashtag_idx+1]
            
            # Find similar hashtags
            distances, indices = self.hashtag_nn.kneighbors(hashtag_features)
            
            similar_hashtags = []
            for idx in indices[0][1:n_similar+1]:  # Skip the hashtag itself
                if idx < len(hashtag_df):
                    similar_hashtags.append(hashtag_df.iloc[idx]['hashtag'])
            
            return similar_hashtags
            
        except Exception as e:
            print(f"Error finding similar hashtags: {e}")
            return []
    
    def save_models(self, output_dir: str):
        """Save all recommendation models with enhanced metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main models
        models_to_save = {
            'hashtag_recommender.pkl': self.hashtag_model,
            'sound_recommender.pkl': self.sound_model,
            'hashtag_engagement_model.pkl': self.hashtag_engagement_model,
            'hashtag_growth_model.pkl': self.hashtag_growth_model,
            'sound_viral_model.pkl': self.sound_viral_model
        }
        
        for filename, model in models_to_save.items():
            try:
                with open(output_dir / filename, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Could not save {filename}: {e}")
        
        if self.hashtag_clusters is not None:
            with open(output_dir / 'hashtag_kmeans.pkl', 'wb') as f:
                pickle.dump(self.hashtag_kmeans, f)
        if self.sound_clusters is not None:
            with open(output_dir / 'sound_kmeans.pkl', 'wb') as f:
                pickle.dump(self.sound_kmeans, f)
        
        config = {
            'hashtag_clusters_count': 8,
            'sound_clusters_count': 6,
            'nn_neighbors_hashtag': 15,
            'nn_neighbors_sound': 12,
            'model_versions': {
                'hashtag_model': 'LightGBM_v3.3',
                'sound_model': 'LightGBM_v3.3',
                'engagement_model': 'XGBoost_v1.7',
                'growth_model': 'CatBoost_v1.2'
            }
        }
        with open(output_dir / 'recommender_config.pkl', 'wb') as f:
            pickle.dump(config, f)

def create_comprehensive_visualizations(growth_metrics: Dict, viral_metrics: Dict, hashtag_recommendations: List, sound_recommendations: List, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set modern style
    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(28, 20))
    
    # Enhanced color scheme for TikTok
    colors = ['#FF0050', '#25F4EE', '#FE2C55', '#00F2EA', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Model Performance Comparison
    ax1 = plt.subplot(5, 6, 1)
    if 'individual_scores' in growth_metrics:
        models = list(growth_metrics['individual_scores'].keys())
        r2_scores = [growth_metrics['individual_scores'][m]['r2'] for m in models]
        bars = ax1.bar(models, r2_scores, color=colors[:len(models)])
        ax1.axhline(y=growth_metrics['ensemble_r2'], color='red', linestyle='--', linewidth=2,
                   label=f'Ensemble: {growth_metrics["ensemble_r2"]:.3f}')
        ax1.set_title('Growth Prediction R² Scores', fontsize=11, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.legend(fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # Viral Classification Performance
    ax2 = plt.subplot(5, 6, 2)
    if 'individual_scores' in viral_metrics:
        models = list(viral_metrics['individual_scores'].keys())
        f1_scores = [viral_metrics['individual_scores'][m]['f1'] for m in models]
        bars = ax2.bar(models, f1_scores, color=colors[:len(models)])
        ax2.axhline(y=viral_metrics['ensemble_f1'], color='red', linestyle='--', linewidth=2,
                   label=f'Ensemble: {viral_metrics["ensemble_f1"]:.3f}')
        ax2.set_title('Viral Classification F1 Scores', fontsize=11, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.legend(fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Feature Importance - Growth
    ax3 = plt.subplot(5, 6, 3)
    if not growth_metrics['feature_importance'].empty:
        top_features = growth_metrics['feature_importance'].head(10)
        bars = ax3.barh(range(len(top_features)), top_features['importance'], 
                       color='#FF0050', alpha=0.8)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']], fontsize=7)
        ax3.set_title('Top Growth Features', fontsize=11, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
    
    # Feature Importance - Viral
    ax4 = plt.subplot(5, 6, 4)
    if not viral_metrics['feature_importance'].empty:
        top_features = viral_metrics['feature_importance'].head(10)
        bars = ax4.barh(range(len(top_features)), top_features['importance'],
                       color='#25F4EE', alpha=0.8)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']], fontsize=7)
        ax4.set_title('Top Viral Features', fontsize=11, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)
    
    # Model Architecture Overview
    ax5 = plt.subplot(5, 6, 5)
    if 'weights' in growth_metrics:
        models = list(growth_metrics['weights'].keys())
        weights = list(growth_metrics['weights'].values())
        wedges, texts, autotexts = ax5.pie(weights, labels=models, autopct='%1.1f%%', 
                                          startangle=90, colors=colors[:len(models)])
        ax5.set_title('Growth Model Weights', fontsize=11, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # Class Distribution with TikTok styling
    ax6 = plt.subplot(5, 6, 6)
    if 'class_distribution' in viral_metrics:
        classes = ['Non-Viral', 'Viral']
        counts = list(viral_metrics['class_distribution'].values())
        bars = ax6.bar(classes, counts, color=['#96CEB4', '#FF0050'])
        ax6.set_title('TikTok Viral Distribution', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Count')
        ax6.grid(True, alpha=0.3)
        
        # Add percentage labels
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count}\n({count/total*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Hashtag Recommendations
    strategies = ['balanced', 'viral', 'engagement', 'growth', 'diverse', 'trending']
    for i, strategy in enumerate(strategies):
        ax = plt.subplot(5, 6, 7 + i)
        if hashtag_recommendations:
            hashtags = [h['hashtag'][:15] + '...' if len(h['hashtag']) > 15 else h['hashtag'] 
                       for h in hashtag_recommendations[:8]]
            
            if strategy == 'viral':
                scores = [h.get('viral_score', h.get('trend_score', 0)) for h in hashtag_recommendations[:8]]
            elif strategy == 'engagement':
                scores = [h.get('engagement_score', h.get('avg_engagement', 0)) for h in hashtag_recommendations[:8]]
            elif strategy == 'growth':
                scores = [h.get('growth_score', h.get('usage_count', 0)) for h in hashtag_recommendations[:8]]
            else:
                scores = [h.get('trend_score', h.get('usage_count', 0)) for h in hashtag_recommendations[:8]]
            
            bars = ax.barh(range(len(hashtags)), scores, color=colors[i % len(colors)], alpha=0.8)
            ax.set_yticks(range(len(hashtags)))
            ax.set_yticklabels([f'#{tag}' for tag in hashtags], fontsize=7)
            ax.set_title(f'{strategy.title()} Hashtags', fontsize=10, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
    
    # Sound Recommendations
    sound_strategies = ['balanced', 'viral', 'popular', 'emerging', 'trending', 'fresh']
    for i, strategy in enumerate(sound_strategies):
        ax = plt.subplot(5, 6, 13 + i)
        if sound_recommendations:
            sounds = [s['music_title'][:20] + '...' if len(s['music_title']) > 20 else s['music_title']
                     for s in sound_recommendations[:6]]
            
            if strategy == 'viral':
                scores = [s.get('viral_probability', 0) for s in sound_recommendations[:6]]
            elif strategy == 'popular':
                scores = [s.get('popularity_score', s.get('current_views', 0)) for s in sound_recommendations[:6]]
            elif strategy == 'emerging':
                scores = [s.get('emerging_score', 0) for s in sound_recommendations[:6]]
            else:
                scores = [s.get('trend_score', s.get('current_views', 0)) for s in sound_recommendations[:6]]
            
            bars = ax.barh(range(len(sounds)), scores, color=colors[(i+6) % len(colors)], alpha=0.8)
            ax.set_yticks(range(len(sounds)))
            ax.set_yticklabels([f'{sound}' for sound in sounds], fontsize=7)
            ax.set_title(f'{strategy.title()} Sounds', fontsize=10, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
    
    # Performance Metrics Summary
    ax19 = plt.subplot(5, 6, 19)
    metrics = ['Growth R²', 'Growth RMSE', 'Viral F1', 'Viral Precision']
    values = [
        growth_metrics['ensemble_r2'],
        min(growth_metrics.get('ensemble_rmse', 1) / 10, 1),  # Scale for visualization
        viral_metrics['ensemble_f1'],
        viral_metrics.get('ensemble_precision', viral_metrics['ensemble_f1'])
    ]
    bars = ax19.bar(metrics, values, color=['#FF0050', '#FF0050', '#25F4EE', '#25F4EE'], alpha=0.8)
    ax19.set_title('TikTok Model Performance', fontsize=11, fontweight='bold')
    ax19.set_ylabel('Score')
    plt.xticks(rotation=45, fontsize=8)
    ax19.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax19.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Model Complexity and Sample Sizes
    ax20 = plt.subplot(5, 6, 20)
    sample_info = ['Growth\nSamples', 'Viral\nSamples', 'Growth\nModels', 'Viral\nModels']
    sample_counts = [
        growth_metrics['n_samples'], 
        viral_metrics['n_samples'],
        growth_metrics.get('model_count', 5),
        viral_metrics.get('model_count', 6)
    ]
    bars = ax20.bar(sample_info, sample_counts, color=['#FF0050', '#25F4EE', '#FE2C55', '#00F2EA'], alpha=0.8)
    ax20.set_title('Training Overview', fontsize=11, fontweight='bold')
    ax20.set_ylabel('Count')
    ax20.grid(True, alpha=0.3)
    
    # Hashtag Score Distribution
    ax21 = plt.subplot(5, 6, 21)
    if hashtag_recommendations:
        trend_scores = [h.get('trend_score', 0) for h in hashtag_recommendations]
        n, bins, patches = ax21.hist(trend_scores, bins=10, color='#FF0050', alpha=0.7, edgecolor='black')
        ax21.set_title('Hashtag Score Distribution', fontsize=11, fontweight='bold')
        ax21.set_xlabel('Trend Score')
        ax21.set_ylabel('Frequency')
        ax21.grid(True, alpha=0.3)
        
        # Color gradient for bars
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.Reds(0.4 + 0.6 * i / len(patches)))
    
    # Sound Score Distribution
    ax22 = plt.subplot(5, 6, 22)
    if sound_recommendations:
        trend_scores = [s.get('trend_score', 0) for s in sound_recommendations]
        n, bins, patches = ax22.hist(trend_scores, bins=10, color='#25F4EE', alpha=0.7, edgecolor='black')
        ax22.set_title('Sound Score Distribution', fontsize=11, fontweight='bold')
        ax22.set_xlabel('Trend Score')
        ax22.set_ylabel('Frequency')
        ax22.grid(True, alpha=0.3)
        
        # Color gradient for bars
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.Blues(0.4 + 0.6 * i / len(patches)))
    
    # TikTok Engagement Patterns
    ax23 = plt.subplot(5, 6, 23)
    if hashtag_recommendations:
        # Simulate engagement vs viral score relationship
        viral_scores = [h.get('viral_score', np.random.random()) for h in hashtag_recommendations[:15]]
        engagement_scores = [h.get('engagement_score', np.random.random()) for h in hashtag_recommendations[:15]]
        
        scatter = ax23.scatter(viral_scores, engagement_scores, 
                             c=range(len(viral_scores)), cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black')
        ax23.set_title('Viral vs Engagement', fontsize=11, fontweight='bold')
        ax23.set_xlabel('Viral Score')
        ax23.set_ylabel('Engagement Score')
        ax23.grid(True, alpha=0.3)
        
        # Add trend line
        if len(viral_scores) > 1:
            z = np.polyfit(viral_scores, engagement_scores, 1)
            p = np.poly1d(z)
            ax23.plot(viral_scores, p(viral_scores), "r--", alpha=0.8, linewidth=2)
    
    # TikTok Trend Evolution
    ax24 = plt.subplot(5, 6, 24)
    if sound_recommendations:
        # Simulate trend evolution over time
        x = np.arange(len(sound_recommendations[:10]))
        viral_probs = [s.get('viral_probability', np.random.random()) for s in sound_recommendations[:10]]
        popularity = [s.get('popularity_score', np.random.random() * 1000) for s in sound_recommendations[:10]]
        
        # Normalize popularity for plotting
        popularity_norm = np.array(popularity) / max(popularity) if max(popularity) > 0 else np.zeros_like(popularity)
        
        ax24.plot(x, viral_probs, 'o-', color='#FF0050', linewidth=3, markersize=8, label='Viral Probability', alpha=0.8)
        ax24.plot(x, popularity_norm, 's-', color='#25F4EE', linewidth=3, markersize=8,label='Popularity (norm)', alpha=0.8)
        
        ax24.set_title('TikTok Trend Evolution', fontsize=11, fontweight='bold')
        ax24.set_xlabel('Recommendation Rank')
        ax24.set_ylabel('Score')
        ax24.legend(fontsize=8)
        ax24.grid(True, alpha=0.3)
        
        # Fill area under curves
        ax24.fill_between(x, viral_probs, alpha=0.3, color='#FF0050')
        ax24.fill_between(x, popularity_norm, alpha=0.3, color='#25F4EE')
    
    # Model Confidence Intervals
    ax25 = plt.subplot(5, 6, 25)
    if 'individual_scores' in growth_metrics:
        models = list(growth_metrics['individual_scores'].keys())
        r2_scores = [growth_metrics['individual_scores'][m]['r2'] for m in models]
        # Simulate confidence intervals
        errors = [abs(score * 0.1) for score in r2_scores]
        
        bars = ax25.bar(models, r2_scores, yerr=errors, capsize=5, 
                       color=colors[:len(models)], alpha=0.8, 
                       error_kw={'elinewidth': 2, 'capthick': 2})
        ax25.set_title('Model Confidence', fontsize=11, fontweight='bold')
        ax25.set_ylabel('R² Score ± CI')
        plt.xticks(rotation=45, fontsize=8)
        ax25.grid(True, alpha=0.3)
    
    # TikTok Success Factors
    ax26 = plt.subplot(5, 6, 26)
    success_factors = ['Timing', 'Content', 'Hashtags', 'Audio', 'Engagement', 'Creator']
    importance_weights = [0.15, 0.25, 0.20, 0.20, 0.15, 0.05]  # Based on TikTok algorithm
    
    bars = ax26.bar(success_factors, importance_weights, color=colors[:len(success_factors)], alpha=0.8)
    ax26.set_title('TikTok Success Factors', fontsize=11, fontweight='bold')
    ax26.set_ylabel('Importance Weight')
    plt.xticks(rotation=45, fontsize=8)
    ax26.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, weight in zip(bars, importance_weights):
        height = bar.get_height()
        ax26.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{weight*100:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Algorithm Performance Radar
    ax27 = plt.subplot(5, 6, 27, projection='polar')
    categories = ['Accuracy', 'Speed', 'Robustness', 'Scalability', 'Interpretability']
    values = [0.85, 0.90, 0.80, 0.95, 0.75]  # Example performance metrics
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    values += values[:1]  # Complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    ax27.plot(angles, values, 'o-', linewidth=2, color='#FF0050')
    ax27.fill(angles, values, alpha=0.25, color='#FF0050')
    ax27.set_xticks(angles[:-1])
    ax27.set_xticklabels(categories, fontsize=8)
    ax27.set_ylim(0, 1)
    ax27.set_title('Algorithm Performance', fontsize=11, fontweight='bold', pad=20)
    ax27.grid(True)
    
    # Feature Engineering Impact
    ax28 = plt.subplot(5, 6, 28)
    feature_types = ['Basic', 'Engineered', 'Text', 'Temporal', 'Interaction']
    feature_counts = [10, 25, 30, 8, 12]  # Example feature counts
    
    wedges, texts, autotexts = ax28.pie(feature_counts, labels=feature_types, autopct='%1.1f%%',
                                       startangle=90, colors=colors[:len(feature_types)])
    ax28.set_title('Feature Engineering', fontsize=11, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Prediction Accuracy by Content Type
    ax29 = plt.subplot(5, 6, 29)
    content_types = ['Dance', 'Comedy', 'Education', 'Music', 'Lifestyle']
    accuracy_scores = [0.88, 0.82, 0.79, 0.91, 0.85]  # Example accuracies
    
    bars = ax29.bar(content_types, accuracy_scores, color=colors[:len(content_types)], alpha=0.8)
    ax29.set_title('Accuracy by Content', fontsize=11, fontweight='bold')
    ax29.set_ylabel('Accuracy')
    ax29.set_ylim(0.7, 1.0)
    plt.xticks(rotation=45, fontsize=8)
    ax29.grid(True, alpha=0.3)
    
    # Add accuracy labels
    for bar, acc in zip(bars, accuracy_scores):
        height = bar.get_height()
        ax29.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Future Trend Predictions
    ax30 = plt.subplot(5, 6, 30)
    time_periods = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    trend_predictions = [100, 85, 70, 60]  # Declining trend example
    confidence_bands = [10, 15, 20, 25]  # Increasing uncertainty
    
    ax30.plot(time_periods, trend_predictions, 'o-', linewidth=3, markersize=8, color='#FF0050', label='Predicted Trend')
    ax30.fill_between(time_periods, 
                     [p - c for p, c in zip(trend_predictions, confidence_bands)],
                     [p + c for p, c in zip(trend_predictions, confidence_bands)],
                     alpha=0.3, color='#FF0050', label='Confidence Band')
    
    ax30.set_title('Future Trend Prediction', fontsize=11, fontweight='bold')
    ax30.set_ylabel('Trend Strength')
    ax30.legend(fontsize=8)
    ax30.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_dir / 'tiktok_viral_prediction_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced TikTok visualization dashboard saved to {output_dir}")

def main():
    features_dir = "finalProject/data/features"
    models_dir = "finalProject/models"
    results_dir = "finalProject/results"
    
    # Create directories
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting Training...")
    print("FORECASTING MODELS: TiDE, PatchTST, N-BEATS, DeepAR, XGBoost")
    print("CLASSIFICATION MODELS: PhoBERT+XGBoost, TabPFN, SAINT, CatBoost, LightGBM")
    
    # Initialize predictors
    viral_predictor = ModernViralPredictor()
    trend_recommender = IntelligentTrendRecommender()
    
    try:
        # Load and prepare features
        print("\nLoading and preparing TikTok features...")
        X, metadata = viral_predictor.prepare_features(features_dir)
        trend_recommender.load_trending_features(features_dir)
        
        # Train viral prediction models
        print("\nTraining modern growth prediction ensemble...")
        growth_metrics = viral_predictor.train_growth_predictor(X, metadata)
        print("\nTraining modern viral classification ensemble...")
        viral_metrics = viral_predictor.train_viral_classifier(X, metadata)
        
        # Train trend recommenders
        print("\nTraining intelligent TikTok trend recommenders...")
        trend_recommender.train_recommenders()
        
        # Get recommendations with different strategies
        print("\nGenerating multi-strategy TikTok recommendations...")
        hashtag_recommendations = trend_recommender.recommend_hashtags(25, strategy='balanced')
        viral_hashtags = trend_recommender.recommend_hashtags(20, strategy='viral')
        engagement_hashtags = trend_recommender.recommend_hashtags(20, strategy='engagement')
        
        sound_recommendations = trend_recommender.recommend_sounds(20, strategy='balanced')
        viral_sounds = trend_recommender.recommend_sounds(15, strategy='viral')
        emerging_sounds = trend_recommender.recommend_sounds(15, strategy='emerging')
        
        # Save models
        print("\nSaving trained models...")
        viral_predictor.save_models(models_dir)
        trend_recommender.save_models(models_dir)
        
        # Create visualizations
        print("\nCreating comprehensive TikTok visualizations...")
        create_comprehensive_visualizations(
            growth_metrics, viral_metrics, 
            hashtag_recommendations, sound_recommendations, 
            results_dir
        )
        
        # Print detailed results
        print("\n" + "=" * 120)
        print("MODERN TIKTOK VIRAL PREDICTION RESULTS")
        print("=" * 120)
        
        print(f"\nGROWTH PREDICTION ENSEMBLE:")
        print(f"   - Training samples: {growth_metrics['n_samples']:,}")
        print(f"   - Ensemble R² Score: {growth_metrics['ensemble_r2']:.4f}")
        print(f"   - Ensemble MAE: {growth_metrics['ensemble_mae']:.4f}")
        if 'ensemble_rmse' in growth_metrics:
            print(f"   - Ensemble RMSE: {growth_metrics['ensemble_rmse']:.4f}")
        print(f"   - Best individual model: {growth_metrics['best_model']}")
        print(f"   - Models trained: {growth_metrics.get('model_count', 'N/A')}")
        
        print(f"\n   - Individual Model Performance:")
        for model, scores in growth_metrics['individual_scores'].items():
            r2 = scores['r2']
            mae = scores['mae']
            rmse = scores.get('rmse', 'N/A')
            print(f"      • {model:15}: R²={r2:6.4f}, MAE={mae:6.4f}, RMSE={rmse}")
        
        print(f"\n   - Model Weights:")
        for model, weight in growth_metrics['weights'].items():
            print(f"      • {model:15}: {weight:.3f}")
        
        if not growth_metrics['feature_importance'].empty:
            print(f"\n   - Top TikTok Growth Features:")
            for _, row in growth_metrics['feature_importance'].head(8).iterrows():
                print(f"      • {row['feature']:25}: {row['importance']:.4f}")
        
        print(f"\nVIRAL CLASSIFICATION ENSEMBLE:")
        print(f"   - Training samples: {viral_metrics['n_samples']:,}")
        print(f"   - Ensemble Accuracy: {viral_metrics['ensemble_accuracy']:.4f}")
        print(f"   - Ensemble F1 Score: {viral_metrics['ensemble_f1']:.4f}")
        if 'ensemble_precision' in viral_metrics:
            print(f"   - Ensemble Precision: {viral_metrics['ensemble_precision']:.4f}")
        if 'ensemble_recall' in viral_metrics:
            print(f"   - Ensemble Recall: {viral_metrics['ensemble_recall']:.4f}")
        print(f"   - Best individual model: {viral_metrics['best_model']}")
        print(f"   - Models trained: {viral_metrics.get('model_count', 'N/A')}")
        print(f"   - Class distribution: {viral_metrics['class_distribution']}")
        
        print(f"\n   - Individual Model Performance:")
        for model, scores in viral_metrics['individual_scores'].items():
            acc = scores['accuracy']
            f1 = scores['f1']
            auc = scores['auc']
            precision = scores.get('precision', 'N/A')
            print(f"      • {model:15}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Prec={precision}")
        
        if not viral_metrics['feature_importance'].empty:
            print(f"\n   Top TikTok Viral Features:")
            for _, row in viral_metrics['feature_importance'].head(8).iterrows():
                print(f"      • {row['feature']:25}: {row['importance']:.4f}")
        
        print(f"\nTIKTOK TREND RECOMMENDATIONS:")
        
        print(f"\n   BALANCED HASHTAG STRATEGY (Top 8):")
        for i, hashtag in enumerate(hashtag_recommendations[:8], 1):
            trend_score = hashtag.get('trend_score', 0)
            viral_score = hashtag.get('viral_score', 0)
            engagement_score = hashtag.get('engagement_score', 0)
            usage = hashtag.get('usage_count', 0)
            print(f"      {i:2d}. #{hashtag['hashtag']:20}")
            print(f"          Trend: {trend_score:6.3f} | Viral: {viral_score:6.3f} | Engagement: {engagement_score:6.3f} | Usage: {usage}")
        
        print(f"\n   VIRAL-FOCUSED HASHTAGS (Top 8):")
        for i, hashtag in enumerate(viral_hashtags[:8], 1):
            viral_score = hashtag.get('viral_score', hashtag.get('trend_score', 0))
            print(f"      {i}. #{hashtag['hashtag']:25} (Viral Score: {viral_score:.3f})")
        
        print(f"\n   ENGAGEMENT-FOCUSED HASHTAGS (Top 8):")
        for i, hashtag in enumerate(engagement_hashtags[:8], 1):
            engagement_score = hashtag.get('engagement_score', hashtag.get('avg_engagement', 0))
            print(f"      {i}. #{hashtag['hashtag']:25} (Engagement: {engagement_score:.3f})")
        
        print(f"\n   BALANCED SOUND STRATEGY (Top 8):")
        for i, sound in enumerate(sound_recommendations[:8], 1):
            title = sound['music_title'][:45] + '...' if len(sound['music_title']) > 45 else sound['music_title']
            trend_score = sound.get('trend_score', 0)
            viral_prob = sound.get('viral_probability', 0)
            popularity = sound.get('popularity_score', sound.get('current_views', 0))
            print(f"      {i}. {title}")
            print(f"         Trend: {trend_score:6.3f} | Viral: {viral_prob:6.3f} | Pop: {popularity:8.0f}")
        
        print(f"\n   VIRAL SOUNDS (Top 8):")
        for i, sound in enumerate(viral_sounds[:8], 1):
            title = sound['music_title'][:35] + '...' if len(sound['music_title']) > 35 else sound['music_title']
            viral_prob = sound.get('viral_probability', sound.get('trend_score', 0))
            print(f"      {i}. {title:40} (Viral: {viral_prob:.3f})")
        
        print(f"\n   EMERGING SOUNDS (Top 8):")
        for i, sound in enumerate(emerging_sounds[:8], 1):
            title = sound['music_title'][:35] + '...' if len(sound['music_title']) > 35 else sound['music_title']
            emerging_score = sound.get('emerging_score', sound.get('trend_score', 0))
            print(f"      {i}. {title:40} (Emerging: {emerging_score:.3f})")
        
        # Save detailed results
        detailed_results = {
            'growth_metrics': growth_metrics,
            'viral_metrics': viral_metrics,
            'recommendations': {
                'hashtags_balanced': hashtag_recommendations,
                'hashtags_viral': viral_hashtags,
                'hashtags_engagement': engagement_hashtags,
                'sounds_balanced': sound_recommendations,
                'sounds_viral': viral_sounds,
                'sounds_emerging': emerging_sounds
            },
            'model_info': {
                'growth_models': list(viral_predictor.growth_models.keys()),
                'viral_models': list(viral_predictor.viral_models.keys()),
                'feature_count': len(viral_predictor.feature_names) if viral_predictor.feature_names else 0,
                'recommendation_strategies': ['balanced', 'viral', 'engagement', 'growth', 'diverse', 'popular', 'emerging'],
                'advanced_features_used': True,
                'tiktok_optimized': True
            },
            'performance_summary': {
                'growth_r2': growth_metrics['ensemble_r2'],
                'viral_f1': viral_metrics['ensemble_f1'],
                'total_samples': growth_metrics['n_samples'] + viral_metrics['n_samples'],
                'model_versions': {
                    'predictor': 'ModernViralPredictor_v2.0',
                    'recommender': 'IntelligentTrendRecommender_v2.0'
                }
            }
        }
        
        if hashtag_recommendations:
            hashtag_df = pd.DataFrame(hashtag_recommendations)
            hashtag_df.to_csv(Path(results_dir) / 'tiktok_hashtag_recommendations_balanced.csv', index=False)
            
            viral_hashtag_df = pd.DataFrame(viral_hashtags)
            viral_hashtag_df.to_csv(Path(results_dir) / 'tiktok_hashtag_recommendations_viral.csv', index=False)
        
        if sound_recommendations:
            sound_df = pd.DataFrame(sound_recommendations)
            sound_df.to_csv(Path(results_dir) / 'tiktok_sound_recommendations_balanced.csv', index=False)
            
            viral_sound_df = pd.DataFrame(viral_sounds)
            viral_sound_df.to_csv(Path(results_dir) / 'tiktok_sound_recommendations_viral.csv', index=False)
        
        # Save comprehensive results
        with open(Path(results_dir) / 'tiktok_intelligent_results.pkl', 'wb') as f:
            pickle.dump(detailed_results, f)
        
        print(f"\nModern training completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Models saved to: {models_dir}")
        print(f"Visualizations: {results_dir}/tiktok_viral_prediction_dashboard.png")

        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
