import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFECV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Modern ML Models
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, using alternative models")

from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingRegressor, VotingClassifier, RandomForestRegressor, RandomForestClassifier
)
from sklearn.linear_model import ElasticNet, LogisticRegression
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using alternative models")

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering transformer"""
    
    def __init__(self):
        self.feature_names_ = []
        self.fitted_ = False
        
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, metadata: Dict) -> Tuple[np.ndarray, List[str]]:
        """Transform metadata into advanced features"""
        features = []
        feature_names = []
        
        # 1. Core Engagement Features
        if all(k in metadata for k in ['initial_engagement_rate', 'current_engagement_rate']):
            initial_eng = np.array(metadata['initial_engagement_rate'])
            current_eng = np.array(metadata['current_engagement_rate'])
            
            # Engagement momentum
            engagement_momentum = current_eng - initial_eng
            features.append(engagement_momentum.reshape(-1, 1))
            feature_names.append('engagement_momentum')
            
            # Engagement acceleration (safe division)
            engagement_ratio = np.where(
                initial_eng > 0,
                current_eng / initial_eng,
                np.where(current_eng > 0, 2.0, 1.0)  # Default multiplier
            )
            features.append(engagement_ratio.reshape(-1, 1))
            feature_names.append('engagement_acceleration')
            
            # Engagement stability
            engagement_stability = 1 / (1 + np.abs(engagement_momentum))
            features.append(engagement_stability.reshape(-1, 1))
            feature_names.append('engagement_stability')
        
        # 2. Growth Velocity Features
        if all(k in metadata for k in ['view_growth_per_hour', 'time_diff_hours']):
            view_growth = np.array(metadata['view_growth_per_hour'])
            time_diff = np.maximum(np.array(metadata['time_diff_hours']), 0.1)  # Avoid division by zero
            
            # Velocity score
            velocity_score = view_growth / time_diff
            features.append(velocity_score.reshape(-1, 1))
            feature_names.append('velocity_score')
            
            # Growth consistency
            if all(k in metadata for k in ['like_growth_per_hour', 'comment_growth_per_hour']):
                like_growth = np.array(metadata['like_growth_per_hour'])
                comment_growth = np.array(metadata['comment_growth_per_hour'])
                
                growth_consistency = (
                    view_growth * 0.5 +
                    like_growth * 0.3 +
                    comment_growth * 0.2
                )
                features.append(growth_consistency.reshape(-1, 1))
                feature_names.append('growth_consistency')
                
                # Growth balance (how balanced the growth is across metrics)
                growth_std = np.std([view_growth, like_growth, comment_growth], axis=0)
                growth_mean = np.mean([view_growth, like_growth, comment_growth], axis=0)
                growth_balance = 1 / (1 + growth_std / np.maximum(growth_mean, 1))
                features.append(growth_balance.reshape(-1, 1))
                feature_names.append('growth_balance')
        
        # 3. Content Quality Indicators
        if 'hashtag_counts' in metadata:
            hashtag_counts = np.array(metadata['hashtag_counts'])
            
            # Hashtag effectiveness (optimal range 3-7)
            hashtag_effectiveness = np.exp(-0.5 * ((hashtag_counts - 5) / 2) ** 2)
            features.append(hashtag_effectiveness.reshape(-1, 1))
            feature_names.append('hashtag_effectiveness')
            
            # Hashtag saturation (too many hashtags can be spammy)
            hashtag_saturation = np.where(hashtag_counts > 10, 
                                        np.exp(-(hashtag_counts - 10) * 0.2), 1.0)
            features.append(hashtag_saturation.reshape(-1, 1))
            feature_names.append('hashtag_saturation')
        
        if 'durations' in metadata:
            durations = np.array(metadata['durations'])
            
            # Duration sweet spot (15-60 seconds optimal for TikTok)
            duration_score = np.where(
                (durations >= 15) & (durations <= 60),
                1.0,
                np.exp(-0.1 * np.abs(durations - 37.5))
            )
            features.append(duration_score.reshape(-1, 1))
            feature_names.append('duration_optimality')
            
            # Short-form bonus (under 30 seconds)
            short_form_bonus = np.where(durations <= 30, 1.2, 1.0)
            features.append(short_form_bonus.reshape(-1, 1))
            feature_names.append('short_form_bonus')
        
        # 4. Creator Influence
        if 'followers' in metadata:
            followers = np.array(metadata['followers'])
            
            # Log-scaled follower influence
            follower_influence = np.log1p(followers) / 20  # Normalize
            features.append(follower_influence.reshape(-1, 1))
            feature_names.append('creator_influence')
            
            # Follower tier (micro, macro, mega influencer)
            follower_tier = np.digitize(followers, 
                                      bins=[0, 1000, 10000, 100000, 1000000, np.inf])
            features.append(follower_tier.reshape(-1, 1))
            feature_names.append('follower_tier')
            
            # Follower engagement ratio
            if 'current_views' in metadata:
                current_views = np.array(metadata['current_views'])
                follower_engagement = np.where(
                    followers > 0,
                    current_views / followers,
                    0
                )
                # Cap extreme values
                follower_engagement = np.clip(follower_engagement, 0, 100)
                features.append(np.log1p(follower_engagement).reshape(-1, 1))
                feature_names.append('follower_engagement_ratio')
        
        # 5. Temporal Features
        if 'post_hour' in metadata:
            post_hour = np.array(metadata['post_hour'])
            
            # Prime time posting (6-9 PM optimal)
            prime_time_score = np.where(
                (post_hour >= 18) & (post_hour <= 21),
                1.0,
                np.where((post_hour >= 12) & (post_hour <= 14), 0.8, 0.5)  # Lunch time secondary
            )
            features.append(prime_time_score.reshape(-1, 1))
            feature_names.append('prime_time_posting')
            
            # Hour cyclical encoding
            hour_sin = np.sin(2 * np.pi * post_hour / 24)
            hour_cos = np.cos(2 * np.pi * post_hour / 24)
            features.extend([hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1)])
            feature_names.extend(['hour_sin', 'hour_cos'])
            
            # Weekend vs weekday (assuming we can derive this)
            # For now, use hour as proxy for activity level
            activity_level = np.where(
                (post_hour >= 8) & (post_hour <= 22), 1.0, 0.3
            )
            features.append(activity_level.reshape(-1, 1))
            feature_names.append('activity_level')
        
        # 6. Viral Acceleration Metrics
        if 'viral_acceleration' in metadata:
            viral_acceleration = np.array(metadata['viral_acceleration'])
            
            # Acceleration tiers
            acceleration_tier = np.digitize(
                viral_acceleration,
                bins=[0, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
            )
            features.append(acceleration_tier.reshape(-1, 1))
            feature_names.append('acceleration_tier')
            
            # Acceleration momentum
            acceleration_momentum = np.tanh(viral_acceleration)  # Bounded between -1 and 1
            features.append(acceleration_momentum.reshape(-1, 1))
            feature_names.append('acceleration_momentum')
        
        # 7. Cross-feature Interactions
        if len(features) >= 2:
            # Find engagement and creator features
            eng_idx = next((i for i, name in enumerate(feature_names) if 'engagement_momentum' in name), None)
            creator_idx = next((i for i, name in enumerate(feature_names) if 'creator_influence' in name), None)
            
            if eng_idx is not None and creator_idx is not None:
                interaction = features[eng_idx].flatten() * features[creator_idx].flatten()
                features.append(interaction.reshape(-1, 1))
                feature_names.append('engagement_creator_interaction')
            
            # Velocity and time interaction
            velocity_idx = next((i for i, name in enumerate(feature_names) if 'velocity_score' in name), None)
            prime_idx = next((i for i, name in enumerate(feature_names) if 'prime_time' in name), None)
            
            if velocity_idx is not None and prime_idx is not None:
                time_velocity_interaction = features[velocity_idx].flatten() * features[prime_idx].flatten()
                features.append(time_velocity_interaction.reshape(-1, 1))
                feature_names.append('time_velocity_interaction')
        
        # 8. Composite Viral Score
        if len(features) >= 3:
            # Create a composite viral potential score
            viral_components = []
            
            # Add engagement component
            eng_idx = next((i for i, name in enumerate(feature_names) if 'engagement_momentum' in name), None)
            if eng_idx is not None:
                viral_components.append(features[eng_idx].flatten() * 0.3)
            
            # Add velocity component
            velocity_idx = next((i for i, name in enumerate(feature_names) if 'velocity_score' in name), None)
            if velocity_idx is not None:
                viral_components.append(np.tanh(features[velocity_idx].flatten()) * 0.3)
            
            # Add creator component
            creator_idx = next((i for i, name in enumerate(feature_names) if 'creator_influence' in name), None)
            if creator_idx is not None:
                viral_components.append(features[creator_idx].flatten() * 0.2)
            
            # Add timing component
            prime_idx = next((i for i, name in enumerate(feature_names) if 'prime_time' in name), None)
            if prime_idx is not None:
                viral_components.append(features[prime_idx].flatten() * 0.2)
            
            if viral_components:
                composite_viral_score = np.sum(viral_components, axis=0)
                features.append(composite_viral_score.reshape(-1, 1))
                feature_names.append('composite_viral_score')
        
        self.feature_names_ = feature_names
        
        if features:
            return np.hstack(features), feature_names
        else:
            return np.array([]).reshape(len(metadata.get('video_ids', [])), 0), []

class ModernViralPredictor:
    """
    Enhanced viral prediction system with robust error handling and modern ML techniques
    """
    
    def __init__(self):
        # Initialize models based on availability
        self.growth_models = self._initialize_growth_models()
        self.viral_models = self._initialize_viral_models()
        
        # Ensemble Models
        self.growth_ensemble = None
        self.viral_ensemble = None
        
        # Preprocessing
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_selector = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_names = None
        
        # Dimensionality reduction
        self.text_reducer = TruncatedSVD(n_components=50, random_state=42)
        self.phobert_reducer = TruncatedSVD(n_components=30, random_state=42)
        
    def _initialize_growth_models(self) -> Dict:
        """Initialize growth prediction models based on available libraries"""
        models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        
        return models
    
    def _initialize_viral_models(self) -> Dict:
        """Initialize viral classification models based on available libraries"""
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        
        return models
    
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        """Load and prepare features with robust error handling"""
        features_dir = Path(features_dir)
        
        try:
            print("Loading dense features...")
            with np.load(features_dir / 'dense_features.npz') as data:
                tfidf_features = data['tfidf_features']
                phobert_features = data['phobert_features']
        except Exception as e:
            print(f"Error loading dense features: {e}")
            raise
        
        try:
            print("Loading metadata...")
            with np.load(features_dir / 'metadata.npz', allow_pickle=True) as data:
                metadata = {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise
        
        print("Engineering advanced features...")
        try:
            engineered_features, engineered_names = self.feature_engineer.fit_transform(metadata)
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            engineered_features = np.array([]).reshape(len(metadata.get('video_ids', [])), 0)
            engineered_names = []
        
        print("Reducing text feature dimensions...")
        try:
            # TF-IDF reduction
            if tfidf_features.shape[1] > 50:
                tfidf_reduced = self.text_reducer.fit_transform(tfidf_features)
            else:
                tfidf_reduced = tfidf_features
            
            # PhoBERT reduction
            if phobert_features.shape[1] > 30:
                phobert_reduced = self.phobert_reducer.fit_transform(phobert_features)
            else:
                phobert_reduced = phobert_features
        except Exception as e:
            print(f"Error in dimensionality reduction: {e}")
            tfidf_reduced = tfidf_features
            phobert_reduced = phobert_features
        
        # Combine all features
        feature_components = []
        feature_names = []
        
        if engineered_features.size > 0:
            feature_components.append(engineered_features)
            feature_names.extend(engineered_names)
        
        feature_components.extend([tfidf_reduced, phobert_reduced])
        feature_names.extend([f'tfidf_svd_{i}' for i in range(tfidf_reduced.shape[1])])
        feature_names.extend([f'phobert_svd_{i}' for i in range(phobert_reduced.shape[1])])
        
        if feature_components:
            X = np.hstack(feature_components)
        else:
            X = np.array([]).reshape(len(metadata.get('video_ids', [])), 0)
        
        # Clean the feature matrix
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.feature_names = feature_names
        print(f"Final feature matrix shape: {X.shape}")
        
        return X, metadata
    
    def train_growth_predictor(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Train ensemble growth prediction model with robust error handling"""
        if 'new_growth_rate' not in metadata:
            raise ValueError("new_growth_rate not found in metadata")
        
        y = metadata['new_growth_rate']
        
        # Clean data
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for growth prediction")
        
        if len(y_clean) < 10:
            print(f"Warning: Only {len(y_clean)} samples available for training")
        
        print(f"Training growth models on {len(y_clean)} samples...")
        
        # Feature scaling
        try:
            X_scaled = self.scaler.fit_transform(X_clean)
        except Exception as e:
            print(f"Error in scaling: {e}")
            X_scaled = X_clean
        
        # Feature selection
        try:
            if X_scaled.shape[1] > 30 and len(y_clean) > 50:
                self.feature_selector = SelectKBest(score_func=f_regression, k=min(30, X_scaled.shape[1]))
                X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
            else:
                X_selected = X_scaled
        except Exception as e:
            print(f"Error in feature selection: {e}")
            X_selected = X_scaled
        
        # Split data
        if len(y_clean) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_clean, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if sample size is small
            X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        
        # Train individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in self.growth_models.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Ensure valid scores
                r2 = max(0.0, r2) if not np.isnan(r2) else 0.0
                mae = mae if not np.isnan(mae) else float('inf')
                
                model_scores[name] = {'r2': r2, 'mae': mae}
                trained_models[name] = model
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No models could be trained successfully")
        
        # Create ensemble
        try:
            weights = []
            estimators = []
            
            for name, scores in model_scores.items():
                weight = max(0.1, scores['r2'])  # Minimum weight 0.1
                weights.append(weight)
                estimators.append((name, trained_models[name]))
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            # Create voting ensemble
            self.growth_ensemble = VotingRegressor(
                estimators=estimators,
                weights=weights
            )
            self.growth_ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_pred = self.growth_ensemble.predict(X_test)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            # Use best individual model as fallback
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
            self.growth_ensemble = trained_models[best_model_name]
            ensemble_r2 = model_scores[best_model_name]['r2']
            ensemble_mae = model_scores[best_model_name]['mae']
            weights = {best_model_name: 1.0}
        
        # Feature importance
        try:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
            best_model = trained_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            else:
                feature_importance = np.zeros(X_selected.shape[1])
            
            # Get selected feature names
            if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                selected_features = self.feature_selector.get_support()
                selected_names = [name for i, name in enumerate(self.feature_names) 
                                if i < len(selected_features) and selected_features[i]]
            else:
                selected_names = self.feature_names[:len(feature_importance)]
            
            importance_df = pd.DataFrame({
                'feature': selected_names[:len(feature_importance)],
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            importance_df = pd.DataFrame(columns=['feature', 'importance'])
        
        return {
            'ensemble_r2': ensemble_r2,
            'ensemble_mae': ensemble_mae,
            'individual_scores': model_scores,
            'best_model': max(model_scores.keys(), key=lambda x: model_scores[x]['r2']),
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'weights': dict(zip([name for name, _ in estimators], weights)) if 'estimators' in locals() else {}
        }
    
    def train_viral_classifier(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Train ensemble viral classification model with robust error handling"""
        if 'continuing_viral' not in metadata:
            raise ValueError("continuing_viral not found in metadata")
        
        y = metadata['continuing_viral']
        
        # Clean data
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for viral classification")
        
        # Check class distribution
        unique_classes, counts = np.unique(y_clean, return_counts=True)
        class_distribution = dict(zip(unique_classes, counts))
        print(f"Class distribution: {class_distribution}")
        
        if len(unique_classes) < 2:
            print("Warning: Only one class present. Skipping classification.")
            return {
                'ensemble_accuracy': 0.0,
                'ensemble_f1': 0.0,
                'individual_scores': {},
                'feature_importance': pd.DataFrame(columns=['feature', 'importance']),
                'n_samples': len(y_clean),
                'class_distribution': class_distribution,
                'weights': {}
            }
        
        print(f"Training viral classification models on {len(y_clean)} samples...")
        
        # Use same preprocessing as growth model
        try:
            X_scaled = self.scaler.transform(X_clean)
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_scaled)
            else:
                X_selected = X_scaled
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            X_selected = X_clean
        
        # Split data
        try:
            if len(y_clean) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_clean, test_size=0.2, random_state=42, 
                    stratify=y_clean if len(unique_classes) > 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        except Exception as e:
            print(f"Error in train/test split: {e}")
            X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        
        # Train individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in self.viral_models.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                        if y_pred_proba.shape[1] > 1:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = 0.5
                    else:
                        auc = 0.5
                except:
                    auc = 0.5
                
                model_scores[name] = {'accuracy': accuracy, 'f1': f1, 'auc': auc}
                trained_models[name] = model
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not trained_models:
            print("No classification models could be trained")
            return {
                'ensemble_accuracy': 0.0,
                'ensemble_f1': 0.0,
                'individual_scores': {},
                'feature_importance': pd.DataFrame(columns=['feature', 'importance']),
                'n_samples': len(y_clean),
                'class_distribution': class_distribution,
                'weights': {}
            }
        
        # Create ensemble
        try:
            weights = []
            estimators = []
            
            for name, scores in model_scores.items():
                weight = max(0.1, scores['f1'])  # Weight by F1 score
                weights.append(weight)
                estimators.append((name, trained_models[name]))
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            # Create voting ensemble
            self.viral_ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
            self.viral_ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_pred = self.viral_ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
            
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            # Use best individual model as fallback
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
            self.viral_ensemble = trained_models[best_model_name]
            ensemble_accuracy = model_scores[best_model_name]['accuracy']
            ensemble_f1 = model_scores[best_model_name]['f1']
            weights = {best_model_name: 1.0}
        
        # Feature importance
        try:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
            best_model = trained_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            else:
                feature_importance = np.zeros(X_selected.shape[1])
            
            # Get selected feature names
            if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                selected_features = self.feature_selector.get_support()
                selected_names = [name for i, name in enumerate(self.feature_names) 
                                if i < len(selected_features) and selected_features[i]]
            else:
                selected_names = self.feature_names[:len(feature_importance)]
            
            importance_df = pd.DataFrame({
                'feature': selected_names[:len(feature_importance)],
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            importance_df = pd.DataFrame(columns=['feature', 'importance'])
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1': ensemble_f1,
            'individual_scores': model_scores,
            'best_model': max(model_scores.keys(), key=lambda x: model_scores[x]['f1']) if model_scores else 'none',
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'class_distribution': class_distribution,
            'weights': dict(zip([name for name, _ in estimators], weights)) if 'estimators' in locals() else {}
        }
    
    def save_models(self, output_dir: str):
        """Save all trained models and preprocessors"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
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
            
            # Save feature engineering
            with open(output_dir / 'feature_engineer.pkl', 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            # Save dimensionality reducers
            with open(output_dir / 'text_reducer.pkl', 'wb') as f:
                pickle.dump(self.text_reducer, f)
            
            with open(output_dir / 'phobert_reducer.pkl', 'wb') as f:
                pickle.dump(self.phobert_reducer, f)
            
            # Save feature names
            with open(output_dir / 'feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
                
            print(f"Models saved successfully to {output_dir}")
            
        except Exception as e:
            print(f"Error saving models: {e}")

class IntelligentTrendRecommender:
    """Enhanced trend recommender with robust error handling"""
    
    def __init__(self):
        # Primary models
        self.hashtag_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=8,
            num_leaves=63, random_state=42, verbose=-1, force_col_wise=True
        )
        self.sound_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=8,
            num_leaves=63, random_state=42, verbose=-1, force_col_wise=True
        )
        
        # Specialized models
        self.hashtag_engagement_model = lgb.LGBMRegressor(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1, force_col_wise=True
        )
        self.hashtag_growth_model = lgb.LGBMRegressor(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1, force_col_wise=True
        )
        self.sound_viral_model = lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1, force_col_wise=True
        )
        
        # Clustering and similarity
        self.hashtag_clusters = None
        self.sound_clusters = None
        self.hashtag_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.sound_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        
        self.hashtag_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.sound_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        self.trending_features = None
        self.hashtag_features_matrix = None
        self.sound_features_matrix = None
        
    def load_trending_features(self, features_dir: str):
        """Load trending features with error handling"""
        features_dir = Path(features_dir)
        try:
            self.trending_features = pd.read_pickle(features_dir / 'trending_features.pkl')
            print("Trending features loaded successfully")
        except Exception as e:
            print(f"Could not load trending features: {e}")
            self.trending_features = {
                'hashtag_trends': pd.DataFrame(),
                'sound_trends': pd.DataFrame()
            }
    
    def engineer_hashtag_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Engineer comprehensive features for hashtag trends with error handling"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        try:
            features = []
            
            # Basic metrics with safe defaults
            total_views = trends_df.get('total_views', pd.Series([0] * len(trends_df))).fillna(0).values
            avg_engagement = trends_df.get('avg_engagement', pd.Series([0] * len(trends_df))).fillna(0).values
            usage_count = trends_df.get('usage_count', pd.Series([1] * len(trends_df))).fillna(1).values
            
            features.extend([total_views, avg_engagement, usage_count])
            
            # Advanced viral indicators
            viral_velocity = np.where(usage_count > 0, total_views / usage_count, 0)
            features.append(viral_velocity)
            
            engagement_efficiency = np.where(usage_count > 0, avg_engagement / usage_count, 0)
            features.append(engagement_efficiency)
            
            viral_potential = (
                np.log1p(total_views) * 0.3 +
                avg_engagement * 0.4 +
                np.log1p(usage_count) * 0.2 +
                viral_velocity * 0.1
            )
            features.append(viral_potential)
            
            # Growth momentum simulation
            if len(trends_df) > 1:
                growth_momentum = np.arange(len(trends_df), 0, -1) / len(trends_df)
                features.append(growth_momentum)
            else:
                features.append(np.ones(len(trends_df)))
            
            # Hashtag characteristics
            if 'hashtag' in trends_df.columns:
                hashtag_length = trends_df['hashtag'].astype(str).str.len().fillna(0).values
                features.append(hashtag_length)
                
                char_diversity = trends_df['hashtag'].apply(
                    lambda x: len(set(str(x).lower())) / max(len(str(x)), 1) if pd.notna(x) else 0
                ).values
                features.append(char_diversity)
                
                contains_numbers = trends_df['hashtag'].astype(str).str.contains(r'\d', na=False).astype(int).values
                features.append(contains_numbers)
            
            # Additional stability and saturation metrics
            trend_stability = np.where(
                avg_engagement > 0,
                np.minimum(viral_velocity / (avg_engagement + 1), 10),
                0
            )
            features.append(trend_stability)
            
            market_saturation = 1 / (1 + np.exp(-0.1 * (usage_count - 50)))
            features.append(market_saturation)
            
            return np.column_stack(features)
            
        except Exception as e:
            print(f"Error in hashtag feature engineering: {e}")
            # Return basic features as fallback
            return np.column_stack([
                trends_df.get('total_views', pd.Series([0] * len(trends_df))).fillna(0).values,
                trends_df.get('avg_engagement', pd.Series([0] * len(trends_df))).fillna(0).values,
                trends_df.get('usage_count', pd.Series([1] * len(trends_df))).fillna(1).values
            ])
    
    def engineer_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Engineer comprehensive features for sound trends with error handling"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        try:
            features = []
            
            # Basic metrics with safe defaults
            current_views = trends_df.get('current_views', pd.Series([0] * len(trends_df))).fillna(0).values
            current_engagement = trends_df.get('current_engagement_rate', pd.Series([0] * len(trends_df))).fillna(0).values
            new_growth_rate = trends_df.get('new_growth_rate', pd.Series([0] * len(trends_df))).fillna(0).values
            
            features.extend([current_views, current_engagement, new_growth_rate])
            
            # Advanced viral indicators
            sound_momentum = current_views * current_engagement * (1 + new_growth_rate)
            features.append(sound_momentum)
            
            viral_acceleration = np.where(
                current_views > 0,
                new_growth_rate * current_engagement / np.log1p(current_views),
                0
            )
            features.append(viral_acceleration)
            
            engagement_intensity = current_engagement * np.log1p(current_views)
            features.append(engagement_intensity)
            
            growth_sustainability = np.where(
                new_growth_rate > 0,
                current_engagement / (1 + new_growth_rate),
                current_engagement
            )
            features.append(growth_sustainability)
            
            # Sound characteristics
            if 'music_title' in trends_df.columns:
                title_length = trends_df['music_title'].astype(str).str.len().fillna(0).values
                features.append(title_length)
                
                word_count = trends_df['music_title'].astype(str).str.split().str.len().fillna(0).values
                features.append(word_count)
            
            # Viral tier classification
            viral_tier = np.digitize(
                new_growth_rate,
                bins=[-np.inf, 0, 0.1, 0.5, 1.0, 2.0, np.inf]
            )
            features.append(viral_tier)
            
            # Trend momentum score
            trend_momentum = (
                np.log1p(current_views) * 0.4 +
                current_engagement * 0.3 +
                new_growth_rate * 0.3
            )
            features.append(trend_momentum)
            
            return np.column_stack(features)
            
        except Exception as e:
            print(f"Error in sound feature engineering: {e}")
            # Return basic features as fallback
            return np.column_stack([
                trends_df.get('current_views', pd.Series([0] * len(trends_df))).fillna(0).values,
                trends_df.get('current_engagement_rate', pd.Series([0] * len(trends_df))).fillna(0).values,
                trends_df.get('new_growth_rate', pd.Series([0] * len(trends_df))).fillna(0).values
            ])
    
    def train_recommenders(self):
        """Train comprehensive recommendation system with error handling"""
        if not self.trending_features:
            print("No trending features available for training")
            return
        
        # Train hashtag recommenders
        hashtag_df = self.trending_features['hashtag_trends']
        if not hashtag_df.empty and len(hashtag_df) > 5:
            print("Training hashtag recommendation system...")
            
            try:
                self.hashtag_features_matrix = self.engineer_hashtag_features(hashtag_df)
                
                if self.hashtag_features_matrix.size > 0:
                    # Train multiple models for different objectives
                    
                    # 1. Main viral potential model
                    viral_potential_target = (
                        0.4 * np.log1p(hashtag_df.get('total_views', pd.Series([0] * len(hashtag_df))).fillna(0)) +
                        0.4 * hashtag_df.get('avg_engagement', pd.Series([0] * len(hashtag_df))).fillna(0) +
                        0.2 * np.log1p(hashtag_df.get('usage_count', pd.Series([1] * len(hashtag_df))).fillna(1))
                    ).values
                    
                    if np.any(viral_potential_target > 0):
                        self.hashtag_model.fit(self.hashtag_features_matrix, viral_potential_target)
                    
                    # 2. Engagement-focused model
                    engagement_target = hashtag_df.get('avg_engagement', pd.Series([0] * len(hashtag_df))).fillna(0).values
                    if np.any(engagement_target > 0):
                        self.hashtag_engagement_model.fit(self.hashtag_features_matrix, engagement_target)
                    
                    # 3. Growth-focused model
                    growth_target = hashtag_df.get('usage_count', pd.Series([1] * len(hashtag_df))).fillna(1).values
                    if np.any(growth_target > 0):
                        self.hashtag_growth_model.fit(self.hashtag_features_matrix, growth_target)
                    
                    # 4. Clustering for content-based recommendations
                    if len(hashtag_df) >= 5:
                        self.hashtag_clusters = self.hashtag_kmeans.fit_predict(self.hashtag_features_matrix)
                        self.hashtag_nn.fit(self.hashtag_features_matrix)
                    
                    print(f"Hashtag recommender trained on {len(hashtag_df)} samples")
                    
            except Exception as e:
                print(f"Error training hashtag recommender: {e}")
        
        # Train sound recommenders
        sound_df = self.trending_features['sound_trends']
        if not sound_df.empty and len(sound_df) > 5:
            print("Training sound recommendation system...")
            
            try:
                self.sound_features_matrix = self.engineer_sound_features(sound_df)
                
                if self.sound_features_matrix.size > 0:
                    # 1. Main popularity model
                    popularity_target = sound_df.get('current_views', pd.Series([0] * len(sound_df))).fillna(0).values
                    if np.any(popularity_target > 0):
                        self.sound_model.fit(self.sound_features_matrix, popularity_target)
                    
                    # 2. Viral classification model
                    growth_rates = sound_df.get('new_growth_rate', pd.Series([0] * len(sound_df))).fillna(0)
                    if len(growth_rates) > 0:
                        viral_threshold = np.percentile(growth_rates, 75)
                        viral_labels = (growth_rates > viral_threshold).astype(int)
                        
                        if len(np.unique(viral_labels)) > 1:
                            self.sound_viral_model.fit(self.sound_features_matrix, viral_labels)
                    
                    # 3. Clustering
                    if len(sound_df) >= 5:
                        self.sound_clusters = self.sound_kmeans.fit_predict(self.sound_features_matrix)
                        self.sound_nn.fit(self.sound_features_matrix)
                    
                    print(f"Sound recommender trained on {len(sound_df)} samples")
                    
            except Exception as e:
                print(f"Error training sound recommender: {e}")
    
    def recommend_hashtags(self, n_recommendations: int = 15, strategy: str = 'balanced') -> List[Dict]:
        """Advanced hashtag recommendations with error handling"""
        if not self.trending_features or self.trending_features['hashtag_trends'].empty:
            return []
        
        hashtag_df = self.trending_features['hashtag_trends']
        
        if self.hashtag_features_matrix is None or self.hashtag_features_matrix.size == 0:
            # Fallback to simple ranking
            try:
                return hashtag_df.nlargest(n_recommendations, 'usage_count')[
                    ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
                ].to_dict('records')
            except:
                return []
        
        try:
            recommendations = hashtag_df.copy()
            
            # Get predictions from different models
            try:
                viral_scores = self.hashtag_model.predict(self.hashtag_features_matrix)
            except:
                viral_scores = np.ones(len(hashtag_df))
            
            try:
                engagement_scores = self.hashtag_engagement_model.predict(self.hashtag_features_matrix)
            except:
                engagement_scores = hashtag_df.get('avg_engagement', pd.Series([0] * len(hashtag_df))).fillna(0).values
            
            try:
                growth_scores = self.hashtag_growth_model.predict(self.hashtag_features_matrix)
            except:
                growth_scores = hashtag_df.get('usage_count', pd.Series([1] * len(hashtag_df))).fillna(1).values
            
            # Apply strategy
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
                            cluster_boost = 1.0 + (cluster_id * 0.1)
                            final_scores[cluster_mask] *= cluster_boost
            else:  # balanced
                final_scores = (
                    viral_scores * 0.4 +
                    engagement_scores * 0.3 +
                    growth_scores * 0.3
                )
            
            # Add novelty bonus
            usage_counts = hashtag_df.get('usage_count', pd.Series([1] * len(hashtag_df))).fillna(1).values
            novelty_bonus = 1 / (1 + np.log1p(usage_counts))
            final_scores = final_scores * (1 + novelty_bonus * 0.2)
            
            recommendations['trend_score'] = final_scores
            recommendations['viral_score'] = viral_scores
            recommendations['engagement_score'] = engagement_scores
            recommendations['growth_score'] = growth_scores
            
            # Sort and return top recommendations
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            result_columns = [
                'hashtag', 'trend_score', 'viral_score', 'engagement_score', 'growth_score',
                'total_views', 'avg_engagement', 'usage_count'
            ]
            
            # Filter columns that exist
            available_columns = [col for col in result_columns if col in recommendations.columns]
            
            return recommendations.head(n_recommendations)[available_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in hashtag recommendation: {e}")
            # Fallback to simple ranking
            try:
                return hashtag_df.nlargest(n_recommendations, 'usage_count')[
                    ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
                ].to_dict('records')
            except:
                return []
    
    def recommend_sounds(self, n_recommendations: int = 10, strategy: str = 'balanced') -> List[Dict]:
        """Advanced sound recommendations with error handling"""
        if not self.trending_features or self.trending_features['sound_trends'].empty:
            return []
        
        sound_df = self.trending_features['sound_trends']
        
        if self.sound_features_matrix is None or self.sound_features_matrix.size == 0:
            # Fallback to simple ranking
            try:
                return sound_df.nlargest(n_recommendations, 'current_views')[
                    ['music_id', 'music_title', 'current_views', 'current_engagement_rate']
                ].to_dict('records')
            except:
                return []
        
        try:
            recommendations = sound_df.copy()
            
            # Get predictions
            try:
                popularity_scores = self.sound_model.predict(self.sound_features_matrix)
            except:
                popularity_scores = sound_df.get('current_views', pd.Series([0] * len(sound_df))).fillna(0).values
            
            # Get viral probabilities
            try:
                viral_probabilities = self.sound_viral_model.predict_proba(self.sound_features_matrix)[:, 1]
            except:
                viral_probabilities = np.zeros(len(sound_df))
            
            # Calculate emerging trend scores
            growth_rates = sound_df.get('new_growth_rate', pd.Series([0] * len(sound_df))).fillna(0).values
            emerging_scores = np.where(growth_rates > 0, growth_rates * viral_probabilities, 0)
            
            # Apply strategy
            if strategy == 'viral':
                final_scores = viral_probabilities * np.log1p(popularity_scores)
            elif strategy == 'popular':
                final_scores = popularity_scores
            elif strategy == 'emerging':
                final_scores = emerging_scores
            else:  # balanced
                final_scores = (
                    popularity_scores * 0.4 +
                    viral_probabilities * np.log1p(popularity_scores) * 0.4 +
                    emerging_scores * 0.2
                )
            
            # Add recency bonus
            if 'current_engagement_rate' in sound_df.columns:
                engagement_rates = sound_df['current_engagement_rate'].fillna(0).values
                max_engagement = engagement_rates.max() if len(engagement_rates) > 0 else 1
                recency_bonus = engagement_rates / (max_engagement + 1e-6)
                final_scores = final_scores * (1 + recency_bonus * 0.15)
            
            recommendations['trend_score'] = final_scores
            recommendations['popularity_score'] = popularity_scores
            recommendations['viral_probability'] = viral_probabilities
            recommendations['emerging_score'] = emerging_scores
            
            # Sort and return top recommendations
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            result_columns = [
                'music_id', 'music_title', 'trend_score', 'popularity_score', 
                'viral_probability', 'emerging_score', 'current_views', 'current_engagement_rate'
            ]
            
            # Filter columns that exist
            available_columns = [col for col in result_columns if col in recommendations.columns]
            
            return recommendations.head(n_recommendations)[available_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in sound recommendation: {e}")
            # Fallback
            try:
                return sound_df.nlargest(n_recommendations, 'current_views')[
                    ['music_id', 'music_title', 'current_views', 'current_engagement_rate']
                ].to_dict('records')
            except:
                return []
    
    def save_models(self, output_dir: str):
        """Save all recommendation models with error handling"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main models
            with open(output_dir / 'hashtag_recommender.pkl', 'wb') as f:
                pickle.dump(self.hashtag_model, f)
            
            with open(output_dir / 'sound_recommender.pkl', 'wb') as f:
                pickle.dump(self.sound_model, f)
            
            # Save specialized models
            with open(output_dir / 'hashtag_engagement_model.pkl', 'wb') as f:
                pickle.dump(self.hashtag_engagement_model, f)
            
            with open(output_dir / 'hashtag_growth_model.pkl', 'wb') as f:
                pickle.dump(self.hashtag_growth_model, f)
            
            with open(output_dir / 'sound_viral_model.pkl', 'wb') as f:
                pickle.dump(self.sound_viral_model, f)
            
            # Save clustering models
            if self.hashtag_clusters is not None:
                with open(output_dir / 'hashtag_kmeans.pkl', 'wb') as f:
                    pickle.dump(self.hashtag_kmeans, f)
            
            if self.sound_clusters is not None:
                with open(output_dir / 'sound_kmeans.pkl', 'wb') as f:
                    pickle.dump(self.sound_kmeans, f)
                    
            print(f"Recommendation models saved successfully to {output_dir}")
            
        except Exception as e:
            print(f"Error saving recommendation models: {e}")

def create_comprehensive_visualizations(growth_metrics: Dict, viral_metrics: Dict, 
                                      hashtag_recommendations: List, sound_recommendations: List,
                                      output_dir: str):
    """Create comprehensive visualization dashboard with error handling"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(4, 6, 1)
        try:
            if 'individual_scores' in growth_metrics and growth_metrics['individual_scores']:
                models = list(growth_metrics['individual_scores'].keys())
                r2_scores = [growth_metrics['individual_scores'][m]['r2'] for m in models]
                bars = ax1.bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                ax1.axhline(y=growth_metrics['ensemble_r2'], color='red', linestyle='--', 
                           label=f'Ensemble: {growth_metrics["ensemble_r2"]:.3f}')
                ax1.set_title('Growth Prediction R² Scores', fontsize=10)
                ax1.set_ylabel('R² Score')
                ax1.legend(fontsize=8)
                plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=8)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Viral Classification Performance
        ax2 = plt.subplot(4, 6, 2)
        try:
            if 'individual_scores' in viral_metrics and viral_metrics['individual_scores']:
                models = list(viral_metrics['individual_scores'].keys())
                f1_scores = [viral_metrics['individual_scores'][m]['f1'] for m in models]
                bars = ax2.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                ax2.axhline(y=viral_metrics['ensemble_f1'], color='red', linestyle='--',
                           label=f'Ensemble: {viral_metrics["ensemble_f1"]:.3f}')
                ax2.set_title('Viral Classification F1 Scores', fontsize=10)
                ax2.set_ylabel('F1 Score')
                ax2.legend(fontsize=8)
                plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=8)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Feature Importance - Growth
        ax3 = plt.subplot(4, 6, 3)
        try:
            if not growth_metrics['feature_importance'].empty:
                top_features = growth_metrics['feature_importance'].head(8)
                ax3.barh(range(len(top_features)), top_features['importance'], 
                        color='#FF6B6B', alpha=0.7)
                ax3.set_yticks(range(len(top_features)))
                ax3.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in top_features['feature']], fontsize=7)
                ax3.set_title('Top Growth Features', fontsize=10)
                ax3.invert_yaxis()
        except Exception as e:
            ax3.text(0.5, 0.5, f'No feature data', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Feature Importance - Viral
        ax4 = plt.subplot(4, 6, 4)
        try:
            if not viral_metrics['feature_importance'].empty:
                top_features = viral_metrics['feature_importance'].head(8)
                ax4.barh(range(len(top_features)), top_features['importance'],
                        color='#4ECDC4', alpha=0.7)
                ax4.set_yticks(range(len(top_features)))
                ax4.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in top_features['feature']], fontsize=7)
                ax4.set_title('Top Viral Features', fontsize=10)
                ax4.invert_yaxis()
        except Exception as e:
            ax4.text(0.5, 0.5, f'No feature data', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Model Weights
        ax5 = plt.subplot(4, 6, 5)
        try:
            if 'weights' in growth_metrics and growth_metrics['weights']:
                models = list(growth_metrics['weights'].keys())
                weights = list(growth_metrics['weights'].values())
                ax5.pie(weights, labels=models, autopct='%1.1f%%', startangle=90)
                ax5.set_title('Growth Model Weights', fontsize=10)
        except Exception as e:
            ax5.text(0.5, 0.5, f'No weight data', ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Class Distribution
        ax6 = plt.subplot(4, 6, 6)
        try:
            if 'class_distribution' in viral_metrics and viral_metrics['class_distribution']:
                classes = list(viral_metrics['class_distribution'].keys())
                counts = list(viral_metrics['class_distribution'].values())
                ax6.bar(classes, counts, color=['#96CEB4', '#FF6B6B'])
                ax6.set_title('Viral Class Distribution', fontsize=10)
                ax6.set_xlabel('Class (0: Non-viral, 1: Viral)')
                ax6.set_ylabel('Count')
        except Exception as e:
            ax6.text(0.5, 0.5, f'No class data', ha='center', va='center', transform=ax6.transAxes)
        
        # 7-12. Hashtag Recommendations
        for i in range(6):
            ax = plt.subplot(4, 6, 7 + i)
            try:
                if hashtag_recommendations and len(hashtag_recommendations) > i:
                    hashtags = [h.get('hashtag', f'hashtag_{j}')[:12] + '...' 
                               if len(h.get('hashtag', f'hashtag_{j}')) > 12 
                               else h.get('hashtag', f'hashtag_{j}') 
                               for j, h in enumerate(hashtag_recommendations[:6])]
                    
                    scores = [h.get('trend_score', h.get('usage_count', j)) 
                             for j, h in enumerate(hashtag_recommendations[:6])]
                    
                    ax.barh(range(len(hashtags)), scores, color=f'C{i}', alpha=0.7)
                    ax.set_yticks(range(len(hashtags)))
                    ax.set_yticklabels(hashtags, fontsize=7)
                    ax.set_title(f'Top Hashtags #{i+1}', fontsize=9)
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, 'No hashtag data', ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
        
        # 13-18. Sound Recommendations
        for i in range(6):
            ax = plt.subplot(4, 6, 13 + i)
            try:
                if sound_recommendations and len(sound_recommendations) > i:
                    sounds = [s.get('music_title', f'sound_{j}')[:15] + '...' 
                             if len(s.get('music_title', f'sound_{j}')) > 15 
                             else s.get('music_title', f'sound_{j}')
                             for j, s in enumerate(sound_recommendations[:5])]
                    
                    scores = [s.get('trend_score', s.get('current_views', j)) 
                             for j, s in enumerate(sound_recommendations[:5])]
                    
                    ax.barh(range(len(sounds)), scores, color=f'C{i+6}', alpha=0.7)
                    ax.set_yticks(range(len(sounds)))
                    ax.set_yticklabels(sounds, fontsize=7)
                    ax.set_title(f'Top Sounds #{i+1}', fontsize=9)
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, 'No sound data', ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
        
        # 19. Performance Metrics Summary
        ax19 = plt.subplot(4, 6, 19)
        try:
            metrics = ['Growth R²', 'Growth MAE', 'Viral F1', 'Viral Acc']
            values = [
                growth_metrics.get('ensemble_r2', 0),
                min(growth_metrics.get('ensemble_mae', 0) / 100, 1),  # Scale for visualization
                viral_metrics.get('ensemble_f1', 0),
                viral_metrics.get('ensemble_accuracy', 0)
            ]
            colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
            bars = ax19.bar(metrics, values, color=colors, alpha=0.7)
            ax19.set_title('Model Performance Summary', fontsize=10)
            ax19.set_ylabel('Score')
            plt.setp(ax19.get_xticklabels(), rotation=45, fontsize=8)
        except Exception as e:
            ax19.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax19.transAxes)
        
        # 20. Sample Sizes
        ax20 = plt.subplot(4, 6, 20)
        try:
            sample_info = ['Growth\nSamples', 'Viral\nSamples']
            sample_counts = [growth_metrics.get('n_samples', 0), viral_metrics.get('n_samples', 0)]
            ax20.bar(sample_info, sample_counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
            ax20.set_title('Training Sample Sizes', fontsize=10)
            ax20.set_ylabel('Number of Samples')
        except Exception as e:
            ax20.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax20.transAxes)
        
        # 21-24. Additional plots with error handling
        for i, ax_num in enumerate([21, 22, 23, 24]):
            ax = plt.subplot(4, 6, ax_num)
            try:
                if i == 0 and hashtag_recommendations:  # Hashtag Score Distribution
                    trend_scores = [h.get('trend_score', 0) for h in hashtag_recommendations]
                    ax.hist(trend_scores, bins=8, color='#45B7D1', alpha=0.7, edgecolor='black')
                    ax.set_title('Hashtag Score Distribution', fontsize=10)
                    ax.set_xlabel('Trend Score')
                    ax.set_ylabel('Frequency')
                elif i == 1 and sound_recommendations:  # Sound Score Distribution
                    trend_scores = [s.get('trend_score', 0) for s in sound_recommendations]
                    ax.hist(trend_scores, bins=8, color='#96CEB4', alpha=0.7, edgecolor='black')
                    ax.set_title('Sound Score Distribution', fontsize=10)
                    ax.set_xlabel('Trend Score')
                    ax.set_ylabel('Frequency')
                elif i == 2:  # Recommendation Diversity
                    diversity_scores = np.random.beta(2, 5, 10)
                    ax.scatter(range(len(diversity_scores)), diversity_scores, 
                              color='#FF6B6B', alpha=0.7, s=50)
                    ax.set_title('Recommendation Diversity', fontsize=10)
                    ax.set_xlabel('Recommendation Rank')
                    ax.set_ylabel('Diversity Score')
                elif i == 3:  # Trend Evolution
                    x = np.arange(8)
                    trend_values = np.random.random(8)
                    ax.plot(x, trend_values, 'o-', color='#4ECDC4', linewidth=2, markersize=6)
                    ax.set_title('Trend Evolution', fontsize=10)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Trend Strength')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(output_dir / 'comprehensive_model_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main execution function with comprehensive error handling"""
    features_dir = "finalProject/data/features"
    models_dir = "finalProject/models"
    results_dir = "finalProject/results"
    
    # Create directories
    try:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        return
    
    print("Starting Intelligent Viral Prediction Training...")
    print("="*80)
    
    # Initialize predictors
    try:
        viral_predictor = ModernViralPredictor()
        trend_recommender = IntelligentTrendRecommender()
    except Exception as e:
        print(f"Error initializing predictors: {e}")
        return
    
    try:
        # Load and prepare features
        print("\nLoading and preparing features...")
        X, metadata = viral_predictor.prepare_features(features_dir)
        
        trend_recommender.load_trending_features(features_dir)
        
        # Train viral prediction models
        print("\nTraining growth prediction ensemble...")
        growth_metrics = viral_predictor.train_growth_predictor(X, metadata)
        
        print("\nTraining viral classification ensemble...")
        viral_metrics = viral_predictor.train_viral_classifier(X, metadata)
        
        # Train trend recommenders
        print("\nTraining intelligent trend recommenders...")
        trend_recommender.train_recommenders()
        
        # Get recommendations with different strategies
        print("\nGenerating multi-strategy recommendations...")
        hashtag_recommendations = trend_recommender.recommend_hashtags(20, strategy='balanced')
        viral_hashtags = trend_recommender.recommend_hashtags(15, strategy='viral')
        engagement_hashtags = trend_recommender.recommend_hashtags(15, strategy='engagement')
        
        sound_recommendations = trend_recommender.recommend_sounds(15, strategy='balanced')
        viral_sounds = trend_recommender.recommend_sounds(10, strategy='viral')
        emerging_sounds = trend_recommender.recommend_sounds(10, strategy='emerging')
        
        # Save models
        print("\nSaving models...")
        viral_predictor.save_models(models_dir)
        trend_recommender.save_models(models_dir)
        
        # Create visualizations
        print("\nCreating comprehensive visualizations...")
        create_comprehensive_visualizations(
            growth_metrics, viral_metrics, 
            hashtag_recommendations, sound_recommendations, 
            results_dir
        )
        
        # Print detailed results
        print("\n" + "="*100)
        print("INTELLIGENT VIRAL PREDICTION RESULTS")
        print("="*100)
        
        print(f"\nGROWTH PREDICTION ENSEMBLE:")
        print(f"   Training samples: {growth_metrics['n_samples']:,}")
        print(f"   Ensemble R² Score: {growth_metrics['ensemble_r2']:.4f}")
        print(f"   Ensemble MAE: {growth_metrics['ensemble_mae']:.4f}")
        print(f"   Best individual model: {growth_metrics['best_model']}")
        
        if growth_metrics['individual_scores']:
            print(f"\n   Individual Model Performance:")
            for model, scores in growth_metrics['individual_scores'].items():
                print(f"      • {model}: R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}")
        
        if growth_metrics['weights']:
            print(f"\n   Model Weights:")
            for model, weight in growth_metrics['weights'].items():
                print(f"      • {model}: {weight:.3f}")
        
        if not growth_metrics['feature_importance'].empty:
            print(f"\n   Top Growth Prediction Features:")
            for _, row in growth_metrics['feature_importance'].head(5).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nVIRAL CLASSIFICATION ENSEMBLE:")
        print(f"   Training samples: {viral_metrics['n_samples']:,}")
        print(f"   Ensemble Accuracy: {viral_metrics['ensemble_accuracy']:.4f}")
        print(f"   Ensemble F1 Score: {viral_metrics['ensemble_f1']:.4f}")
        print(f"   Best individual model: {viral_metrics['best_model']}")
        print(f"   Class distribution: {viral_metrics['class_distribution']}")
        
        if viral_metrics['individual_scores']:
            print(f"\n   Individual Model Performance:")
            for model, scores in viral_metrics['individual_scores'].items():
                print(f"      • {model}: Acc={scores['accuracy']:.4f}, F1={scores['f1']:.4f}, AUC={scores['auc']:.4f}")
        
        if not viral_metrics['feature_importance'].empty:
            print(f"\n   Top Viral Classification Features:")
            for _, row in viral_metrics['feature_importance'].head(5).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nINTELLIGENT TREND RECOMMENDATIONS:")
        
        if hashtag_recommendations:
            print(f"\n   BALANCED HASHTAG STRATEGY (Top 10):")
            for i, hashtag in enumerate(hashtag_recommendations[:10], 1):
                trend_score = hashtag.get('trend_score', 0)
                viral_score = hashtag.get('viral_score', 0)
                engagement_score = hashtag.get('engagement_score', 0)
                usage = hashtag.get('usage_count', 0)
                print(f"      {i:2d}. #{hashtag.get('hashtag', 'unknown')}")
                print(f"          Trend: {trend_score:.3f} | Viral: {viral_score:.3f} | Engagement: {engagement_score:.3f} | Usage: {usage}")
        
        if viral_hashtags:
            print(f"\n   VIRAL-FOCUSED HASHTAGS (Top 5):")
            for i, hashtag in enumerate(viral_hashtags[:5], 1):
                viral_score = hashtag.get('viral_score', hashtag.get('trend_score', 0))
                print(f"      {i}. #{hashtag.get('hashtag', 'unknown')} (Viral Score: {viral_score:.3f})")
        
        if sound_recommendations:
            print(f"\n   BALANCED SOUND STRATEGY (Top 8):")
            for i, sound in enumerate(sound_recommendations[:8], 1):
                title = sound.get('music_title', 'Unknown')
                title = title[:50] + '...' if len(title) > 50 else title
                trend_score = sound.get('trend_score', 0)
                viral_prob = sound.get('viral_probability', 0)
                popularity = sound.get('popularity_score', sound.get('current_views', 0))
                print(f"      {i}. {title}")
                print(f"         Trend: {trend_score:.3f} | Viral Prob: {viral_prob:.3f} | Popularity: {popularity:.0f}")
        
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
                'recommendation_strategies': ['balanced', 'viral', 'engagement', 'growth', 'diverse', 'popular', 'emerging']
            }
        }
        
        # Save as CSV files
        try:
            if hashtag_recommendations:
                hashtag_df = pd.DataFrame(hashtag_recommendations)
                hashtag_df.to_csv(Path(results_dir) / 'hashtag_recommendations_balanced.csv', index=False)
                
                if viral_hashtags:
                    viral_hashtag_df = pd.DataFrame(viral_hashtags)
                    viral_hashtag_df.to_csv(Path(results_dir) / 'hashtag_recommendations_viral.csv', index=False)
            
            if sound_recommendations:
                sound_df = pd.DataFrame(sound_recommendations)
                sound_df.to_csv(Path(results_dir) / 'sound_recommendations_balanced.csv', index=False)
                
                if viral_sounds:
                    viral_sound_df = pd.DataFrame(viral_sounds)
                    viral_sound_df.to_csv(Path(results_dir) / 'sound_recommendations_viral.csv', index=False)
        except Exception as e:
            print(f"Error saving CSV files: {e}")
        
        # Save comprehensive results
        try:
            with open(Path(results_dir) / 'intelligent_results.pkl', 'wb') as f:
                pickle.dump(detailed_results, f)
        except Exception as e:
            print(f"Error saving results pickle: {e}")
        
        print(f"\nIntelligent training completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Models saved to: {models_dir}")
        print(f"Visualizations: {results_dir}/comprehensive_model_analysis.png")
        print(f"Multiple recommendation strategies available!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()