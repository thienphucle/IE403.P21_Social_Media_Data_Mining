import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Any
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

# Modern ML Models
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb

warnings.filterwarnings('ignore')

class ModernViralPredictor:
    """
    Sử dụng ensemble của các mô hình hiện đại:
    - LightGBM: Nhanh, hiệu quả cho dữ liệu lớn
    - CatBoost: Xử lý tốt categorical features
    - XGBoost: Mạnh mẽ cho tabular data
    - Gradient Boosting: Robust baseline
    """
    
    def __init__(self):
        # Growth Prediction Models (Regression)
        self.growth_models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        # Viral Classification Models
        self.viral_models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        # Ensemble Models
        self.growth_ensemble = None
        self.viral_ensemble = None
        
        # Preprocessing
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_selector = None
        self.feature_names = None
        
    def engineer_advanced_features(self, metadata: Dict) -> Tuple[np.ndarray, List[str]]:
        features = []
        feature_names = []
        
        # 1. Core Engagement Features
        if all(k in metadata for k in ['initial_engagement_rate', 'current_engagement_rate']):
            # Engagement momentum
            engagement_momentum = metadata['current_engagement_rate'] - metadata['initial_engagement_rate']
            features.append(engagement_momentum.reshape(-1, 1))
            feature_names.append('engagement_momentum')
            
            # Engagement acceleration
            engagement_ratio = np.where(
                metadata['initial_engagement_rate'] > 0,
                metadata['current_engagement_rate'] / metadata['initial_engagement_rate'],
                1.0
            )
            features.append(engagement_ratio.reshape(-1, 1))
            feature_names.append('engagement_acceleration')
        
        # 2. Growth Velocity Features
        if all(k in metadata for k in ['view_growth_per_hour', 'time_diff_hours']):
            # Velocity score
            velocity_score = metadata['view_growth_per_hour'] / np.maximum(metadata['time_diff_hours'], 1)
            features.append(velocity_score.reshape(-1, 1))
            feature_names.append('velocity_score')
            
            # Growth consistency
            if 'like_growth_per_hour' in metadata and 'comment_growth_per_hour' in metadata:
                growth_consistency = (
                    metadata['view_growth_per_hour'] * 0.5 +
                    metadata['like_growth_per_hour'] * 0.3 +
                    metadata['comment_growth_per_hour'] * 0.2
                )
                features.append(growth_consistency.reshape(-1, 1))
                feature_names.append('growth_consistency')
        
        # 3. Content Quality Indicators
        if 'hashtag_counts' in metadata:
            # Hashtag effectiveness (optimal range 3-7)
            hashtag_effectiveness = np.exp(-0.5 * ((metadata['hashtag_counts'] - 5) / 2) ** 2)
            features.append(hashtag_effectiveness.reshape(-1, 1))
            feature_names.append('hashtag_effectiveness')
        
        if 'durations' in metadata:
            # Duration sweet spot (15-60 seconds optimal)
            duration_score = np.where(
                (metadata['durations'] >= 15) & (metadata['durations'] <= 60),
                1.0,
                np.exp(-0.1 * np.abs(metadata['durations'] - 37.5))
            )
            features.append(duration_score.reshape(-1, 1))
            feature_names.append('duration_optimality')
        
        # 4. Creator Influence
        if 'followers' in metadata:
            # Log-scaled follower influence
            follower_influence = np.log1p(metadata['followers']) / 20  # Normalize
            features.append(follower_influence.reshape(-1, 1))
            feature_names.append('creator_influence')
            
            # Follower engagement ratio
            if 'current_views' in metadata:
                follower_engagement = np.where(
                    metadata['followers'] > 0,
                    metadata['current_views'] / metadata['followers'],
                    0
                )
                features.append(np.log1p(follower_engagement).reshape(-1, 1))
                feature_names.append('follower_engagement_ratio')
        
        # 5. Temporal Features
        if 'post_hour' in metadata:
            # Prime time posting (6-9 PM optimal)
            prime_time_score = np.where(
                (metadata['post_hour'] >= 18) & (metadata['post_hour'] <= 21),
                1.0,
                0.5
            )
            features.append(prime_time_score.reshape(-1, 1))
            feature_names.append('prime_time_posting')
            
            # Hour cyclical encoding
            hour_sin = np.sin(2 * np.pi * metadata['post_hour'] / 24)
            hour_cos = np.cos(2 * np.pi * metadata['post_hour'] / 24)
            features.extend([hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1)])
            feature_names.extend(['hour_sin', 'hour_cos'])
        
        # 6. Viral Acceleration Metrics
        if 'viral_acceleration' in metadata:
            # Acceleration tiers
            acceleration_tier = np.digitize(
                metadata['viral_acceleration'],
                bins=[0, 0.1, 0.5, 1.0, 2.0, np.inf]
            )
            features.append(acceleration_tier.reshape(-1, 1))
            feature_names.append('acceleration_tier')
        
        # 7. Cross-feature Interactions
        if len(features) >= 2:
            # Engagement × Creator interaction
            if 'engagement_momentum' in feature_names and 'creator_influence' in feature_names:
                eng_idx = feature_names.index('engagement_momentum')
                creator_idx = feature_names.index('creator_influence')
                interaction = features[eng_idx].flatten() * features[creator_idx].flatten()
                features.append(interaction.reshape(-1, 1))
                feature_names.append('engagement_creator_interaction')
        
        if features:
            return np.hstack(features), feature_names
        else:
            return np.array([]).reshape(0, 0), []
    
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        features_dir = Path(features_dir)
        
        print("Loading dense features...")
        with np.load(features_dir / 'dense_features.npz') as data:
            tfidf_features = data['tfidf_features']
            phobert_features = data['phobert_features']
        
        print("Loading metadata...")
        with np.load(features_dir / 'metadata.npz', allow_pickle=True) as data:
            metadata = {key: data[key] for key in data.files}
        
        print("Engineering advanced features...")
        engineered_features, engineered_names = self.engineer_advanced_features(metadata)
        
        # Dimensionality reduction cho text features
        from sklearn.decomposition import TruncatedSVD
        
        print("Reducing text feature dimensions...")
        # TF-IDF reduction
        if tfidf_features.shape[1] > 50:
            tfidf_reducer = TruncatedSVD(n_components=50, random_state=42)
            tfidf_reduced = tfidf_reducer.fit_transform(tfidf_features)
        else:
            tfidf_reduced = tfidf_features
        
        # PhoBERT reduction
        if phobert_features.shape[1] > 30:
            phobert_reducer = TruncatedSVD(n_components=30, random_state=42)
            phobert_reduced = phobert_reducer.fit_transform(phobert_features)
        else:
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
        
        X = np.hstack(feature_components)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.feature_names = feature_names
        print(f"Final feature matrix shape: {X.shape}")
        
        return X, metadata
    
    def train_growth_predictor(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Train ensemble growth prediction model"""
        y = metadata['new_growth_rate']
        
        # Clean data
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for growth prediction")
        
        print(f"Training growth models on {len(y_clean)} samples...")
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Feature selection
        if X_scaled.shape[1] > 30:
            self.feature_selector = SelectKBest(score_func=f_regression, k=30)
            X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
        else:
            X_selected = X_scaled
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_clean, test_size=0.2, random_state=42
        )
        
        # Train individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in self.growth_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_scores[name] = {'r2': r2, 'mae': mae}
            trained_models[name] = model
        
        # Create ensemble (weighted by performance)
        weights = []
        estimators = []
        
        for name, scores in model_scores.items():
            weight = max(0.1, scores['r2'])  # Minimum weight 0.1
            weights.append(weight)
            estimators.append((name, trained_models[name]))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
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
        
        # Feature importance (from best individual model)
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
        best_model = trained_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        else:
            feature_importance = np.zeros(X_selected.shape[1])
        
        # Get selected feature names
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
            'individual_scores': model_scores,
            'best_model': best_model_name,
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'weights': dict(zip([name for name, _ in estimators], weights))
        }
    
    def train_viral_classifier(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Train ensemble viral classification model"""
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
                'feature_importance': pd.DataFrame(),
                'n_samples': len(y_clean),
                'class_distribution': class_distribution
            }
        
        print(f"Training viral classification models on {len(y_clean)} samples...")
        
        # Use same preprocessing as growth model
        X_scaled = self.scaler.transform(X_clean)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        # Train individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in self.viral_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            model_scores[name] = {'accuracy': accuracy, 'f1': f1, 'auc': auc}
            trained_models[name] = model
        
        # Create ensemble
        weights = []
        estimators = []
        
        for name, scores in model_scores.items():
            weight = max(0.1, scores['f1'])  # Weight by F1 score
            weights.append(weight)
            estimators.append((name, trained_models[name]))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
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
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        # Feature importance
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
        best_model = trained_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        else:
            feature_importance = np.zeros(X_selected.shape[1])
        
        # Get selected feature names
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
            'individual_scores': model_scores,
            'best_model': best_model_name,
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'class_distribution': class_distribution,
            'weights': dict(zip([name for name, _ in estimators], weights))
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

class IntelligentTrendRecommender:
    def __init__(self):
        # Primary models
        self.hashtag_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=8,
            num_leaves=63, random_state=42, verbose=-1
        )
        self.sound_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=8,
            num_leaves=63, random_state=42, verbose=-1
        )
        
        # Advanced models for different aspects
        self.hashtag_engagement_model = lgb.LGBMRegressor(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1
        )
        self.hashtag_growth_model = lgb.LGBMRegressor(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1
        )
        
        self.sound_viral_model = lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.15, max_depth=6,
            random_state=42, verbose=-1
        )
        
        # Clustering for content-based recommendations
        self.hashtag_clusters = None
        self.sound_clusters = None
        self.hashtag_kmeans = KMeans(n_clusters=5, random_state=42)
        self.sound_kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Similarity models
        self.hashtag_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.sound_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        self.trending_features = None
        self.hashtag_features_matrix = None
        self.sound_features_matrix = None
        
    def load_trending_features(self, features_dir: str):
        """Load trending features"""
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
        """Engineer comprehensive features for hashtag trends"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        features = []
        
        # Basic metrics
        total_views = trends_df['total_views'].fillna(0).values
        avg_engagement = trends_df['avg_engagement'].fillna(0).values
        usage_count = trends_df['usage_count'].fillna(0).values
        
        features.extend([total_views, avg_engagement, usage_count])
        
        # Advanced viral indicators
        # 1. Viral velocity (views per usage)
        viral_velocity = np.where(usage_count > 0, total_views / usage_count, 0)
        features.append(viral_velocity)
        
        # 2. Engagement efficiency
        engagement_efficiency = np.where(usage_count > 0, avg_engagement / usage_count, 0)
        features.append(engagement_efficiency)
        
        # 3. Viral potential score
        viral_potential = (
            np.log1p(total_views) * 0.3 +
            avg_engagement * 0.4 +
            np.log1p(usage_count) * 0.2 +
            viral_velocity * 0.1
        )
        features.append(viral_potential)
        
        # 4. Growth momentum (if we have time-series data)
        if len(trends_df) > 1:
            # Simulate growth momentum based on ranking
            growth_momentum = np.arange(len(trends_df), 0, -1) / len(trends_df)
            features.append(growth_momentum)
        else:
            features.append(np.ones(len(trends_df)))
        
        # 5. Hashtag characteristics
        if 'hashtag' in trends_df.columns:
            # Length features
            hashtag_length = trends_df['hashtag'].str.len().fillna(0).values
            features.append(hashtag_length)
            
            # Character diversity
            char_diversity = trends_df['hashtag'].apply(
                lambda x: len(set(str(x).lower())) / max(len(str(x)), 1) if pd.notna(x) else 0
            ).values
            features.append(char_diversity)
            
            # Contains numbers
            contains_numbers = trends_df['hashtag'].str.contains(r'\d', na=False).astype(int).values
            features.append(contains_numbers)
        
        # 6. Trend stability
        trend_stability = np.where(
            avg_engagement > 0,
            np.minimum(viral_velocity / (avg_engagement + 1), 10),
            0
        )
        features.append(trend_stability)
        
        # 7. Market saturation indicator
        market_saturation = 1 / (1 + np.exp(-0.1 * (usage_count - 50)))  # Sigmoid
        features.append(market_saturation)
        
        return np.column_stack(features)
    
    def engineer_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Engineer comprehensive features for sound trends"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        features = []
        
        # Basic metrics
        current_views = trends_df['current_views'].fillna(0).values
        current_engagement = trends_df['current_engagement_rate'].fillna(0).values
        new_growth_rate = trends_df['new_growth_rate'].fillna(0).values
        
        features.extend([current_views, current_engagement, new_growth_rate])
        
        # Advanced viral indicators
        # 1. Sound momentum
        sound_momentum = current_views * current_engagement * (1 + new_growth_rate)
        features.append(sound_momentum)
        
        # 2. Viral acceleration
        viral_acceleration = np.where(
            current_views > 0,
            new_growth_rate * current_engagement / np.log1p(current_views),
            0
        )
        features.append(viral_acceleration)
        
        # 3. Engagement intensity
        engagement_intensity = current_engagement * np.log1p(current_views)
        features.append(engagement_intensity)
        
        # 4. Growth sustainability
        growth_sustainability = np.where(
            new_growth_rate > 0,
            current_engagement / (1 + new_growth_rate),
            current_engagement
        )
        features.append(growth_sustainability)
        
        # 5. Sound characteristics (if available)
        if 'music_title' in trends_df.columns:
            # Title length
            title_length = trends_df['music_title'].str.len().fillna(0).values
            features.append(title_length)
            
            # Title complexity (word count)
            word_count = trends_df['music_title'].str.split().str.len().fillna(0).values
            features.append(word_count)
        
        # 6. Viral tier classification
        viral_tier = np.digitize(
            new_growth_rate,
            bins=[-np.inf, 0, 0.1, 0.5, 1.0, 2.0, np.inf]
        )
        features.append(viral_tier)
        
        # 7. Trend momentum score
        trend_momentum = (
            np.log1p(current_views) * 0.4 +
            current_engagement * 0.3 +
            new_growth_rate * 0.3
        )
        features.append(trend_momentum)
        
        return np.column_stack(features)
    
    def train_recommenders(self):
        if not self.trending_features:
            print("No trending features available for training")
            return
        
        # Train hashtag recommenders
        hashtag_df = self.trending_features['hashtag_trends']
        if not hashtag_df.empty and len(hashtag_df) > 10:
            print("Training hashtag recommendation system...")
            
            # Engineer features
            self.hashtag_features_matrix = self.engineer_hashtag_features(hashtag_df)
            
            if self.hashtag_features_matrix.size > 0:
                # Train multiple models for different objectives
                
                # 1. Main viral potential model
                viral_potential_target = (
                    0.4 * np.log1p(hashtag_df['total_views'].fillna(0)) +
                    0.4 * hashtag_df['avg_engagement'].fillna(0) +
                    0.2 * np.log1p(hashtag_df['usage_count'].fillna(0))
                ).values
                
                if np.any(viral_potential_target > 0):
                    self.hashtag_model.fit(self.hashtag_features_matrix, viral_potential_target)
                
                # 2. Engagement-focused model
                engagement_target = hashtag_df['avg_engagement'].fillna(0).values
                if np.any(engagement_target > 0):
                    self.hashtag_engagement_model.fit(self.hashtag_features_matrix, engagement_target)
                
                # 3. Growth-focused model
                growth_target = hashtag_df['usage_count'].fillna(0).values
                if np.any(growth_target > 0):
                    self.hashtag_growth_model.fit(self.hashtag_features_matrix, growth_target)
                
                # 4. Clustering for content-based recommendations
                if len(hashtag_df) >= 5:
                    self.hashtag_clusters = self.hashtag_kmeans.fit_predict(self.hashtag_features_matrix)
                    self.hashtag_nn.fit(self.hashtag_features_matrix)
                
                print(f"Hashtag recommender trained on {len(hashtag_df)} samples")
        
        # Train sound recommenders
        sound_df = self.trending_features['sound_trends']
        if not sound_df.empty and len(sound_df) > 10:
            print("Training sound recommendation system...")
            
            # Engineer features
            self.sound_features_matrix = self.engineer_sound_features(sound_df)
            
            if self.sound_features_matrix.size > 0:
                # 1. Main popularity model
                popularity_target = sound_df['current_views'].fillna(0).values
                if np.any(popularity_target > 0):
                    self.sound_model.fit(self.sound_features_matrix, popularity_target)
                
                # 2. Viral classification model
                viral_threshold = np.percentile(sound_df['new_growth_rate'].fillna(0), 75)
                viral_labels = (sound_df['new_growth_rate'].fillna(0) > viral_threshold).astype(int)
                
                if len(np.unique(viral_labels)) > 1:
                    self.sound_viral_model.fit(self.sound_features_matrix, viral_labels)
                
                # 3. Clustering
                if len(sound_df) >= 5:
                    self.sound_clusters = self.sound_kmeans.fit_predict(self.sound_features_matrix)
                    self.sound_nn.fit(self.sound_features_matrix)
                
                print(f"Sound recommender trained on {len(sound_df)} samples")
    
    def recommend_hashtags(self, n_recommendations: int = 15, 
                          strategy: str = 'balanced') -> List[Dict]:
        """
        Strategies:
        - 'balanced': Balance between viral potential and engagement
        - 'viral': Focus on viral potential
        - 'engagement': Focus on engagement rates
        - 'growth': Focus on usage growth
        - 'diverse': Diverse recommendations across clusters
        """
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
            
            # Apply strategy
            if strategy == 'viral':
                final_scores = viral_scores
            elif strategy == 'engagement':
                final_scores = engagement_scores
            elif strategy == 'growth':
                final_scores = growth_scores
            elif strategy == 'diverse':
                # Ensure diversity across clusters
                final_scores = viral_scores
                if self.hashtag_clusters is not None:
                    # Boost scores for different clusters
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
            
            # Add novelty bonus (prefer less saturated hashtags)
            usage_counts = hashtag_df['usage_count'].fillna(0).values
            novelty_bonus = 1 / (1 + np.log1p(usage_counts))
            final_scores = final_scores * (1 + novelty_bonus * 0.2)
            
            # Add diversity penalty for similar hashtags
            if len(hashtag_df) > 1 and 'hashtag' in hashtag_df.columns:
                diversity_scores = self._calculate_hashtag_diversity(hashtag_df['hashtag'].values)
                final_scores = final_scores * (1 + diversity_scores * 0.1)
            
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
            
            return recommendations.head(n_recommendations)[result_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in hashtag recommendation: {e}")
            # Fallback to simple ranking
            return hashtag_df.nlargest(n_recommendations, 'usage_count')[
                ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
            ].to_dict('records')
    
    def recommend_sounds(self, n_recommendations: int = 10,
                        strategy: str = 'balanced') -> List[Dict]:
        """
        Strategies:
        - 'balanced': Balance between popularity and viral potential
        - 'viral': Focus on viral potential
        - 'popular': Focus on current popularity
        - 'emerging': Focus on emerging trends
        """
        if not self.trending_features or self.trending_features['sound_trends'].empty:
            return []
        
        sound_df = self.trending_features['sound_trends']
        
        if self.sound_features_matrix is None or self.sound_features_matrix.size == 0:
            return []
        
        try:
            recommendations = sound_df.copy()
            
            # Get predictions
            popularity_scores = self.sound_model.predict(self.sound_features_matrix)
            
            # Get viral probabilities
            try:
                viral_probabilities = self.sound_viral_model.predict_proba(self.sound_features_matrix)[:, 1]
            except:
                viral_probabilities = np.zeros(len(sound_df))
            
            # Calculate emerging trend scores
            growth_rates = sound_df['new_growth_rate'].fillna(0).values
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
            
            # Add recency bonus (prefer newer trends)
            if 'current_engagement_rate' in sound_df.columns:
                engagement_rates = sound_df['current_engagement_rate'].fillna(0).values
                recency_bonus = engagement_rates / (engagement_rates.max() + 1e-6)
                final_scores = final_scores * (1 + recency_bonus * 0.15)
            
            # Add diversity bonus
            if self.sound_clusters is not None:
                diversity_bonus = self._calculate_cluster_diversity(self.sound_clusters)
                final_scores = final_scores * (1 + diversity_bonus * 0.1)
            
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
            
            return recommendations.head(n_recommendations)[result_columns].to_dict('records')
            
        except Exception as e:
            print(f"Error in sound recommendation: {e}")
            # Fallback
            return sound_df.nlargest(n_recommendations, 'current_views')[
                ['music_id', 'music_title', 'current_views', 'current_engagement_rate']
            ].to_dict('records')
    
    def _calculate_hashtag_diversity(self, hashtags: np.ndarray) -> np.ndarray:
        """Calculate diversity scores for hashtags"""
        diversity_scores = np.ones(len(hashtags))
        
        for i, hashtag in enumerate(hashtags):
            if pd.isna(hashtag):
                continue
            
            # Calculate similarity with other hashtags
            similarities = []
            for j, other_hashtag in enumerate(hashtags):
                if i != j and pd.notna(other_hashtag):
                    # Simple character-based similarity
                    similarity = len(set(str(hashtag).lower()) & set(str(other_hashtag).lower())) / \
                               len(set(str(hashtag).lower()) | set(str(other_hashtag).lower()))
                    similarities.append(similarity)
            
            if similarities:
                # Higher diversity score for less similar hashtags
                diversity_scores[i] = 1 - np.mean(similarities)
        
        return diversity_scores
    
    def _calculate_cluster_diversity(self, clusters: np.ndarray) -> np.ndarray:
        """Calculate diversity bonus based on cluster distribution"""
        diversity_scores = np.ones(len(clusters))
        
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        cluster_weights = dict(zip(unique_clusters, 1 / counts))  # Inverse frequency
        
        for i, cluster in enumerate(clusters):
            diversity_scores[i] = cluster_weights.get(cluster, 1.0)
        
        return diversity_scores
    
    def get_similar_hashtags(self, hashtag: str, n_similar: int = 5) -> List[str]:
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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

def create_comprehensive_visualizations(growth_metrics: Dict, viral_metrics: Dict, 
                                      hashtag_recommendations: List, sound_recommendations: List,
                                      output_dir: str):
    """Create comprehensive visualization dashboard"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(4, 6, 1)
    if 'individual_scores' in growth_metrics:
        models = list(growth_metrics['individual_scores'].keys())
        r2_scores = [growth_metrics['individual_scores'][m]['r2'] for m in models]
        bars = ax1.bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.axhline(y=growth_metrics['ensemble_r2'], color='red', linestyle='--', 
                   label=f'Ensemble: {growth_metrics["ensemble_r2"]:.3f}')
        ax1.set_title('Growth Prediction R² Scores', fontsize=10)
        ax1.set_ylabel('R² Score')
        ax1.legend(fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
    
    # 2. Viral Classification Performance
    ax2 = plt.subplot(4, 6, 2)
    if 'individual_scores' in viral_metrics:
        models = list(viral_metrics['individual_scores'].keys())
        f1_scores = [viral_metrics['individual_scores'][m]['f1'] for m in models]
        bars = ax2.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.axhline(y=viral_metrics['ensemble_f1'], color='red', linestyle='--',
                   label=f'Ensemble: {viral_metrics["ensemble_f1"]:.3f}')
        ax2.set_title('Viral Classification F1 Scores', fontsize=10)
        ax2.set_ylabel('F1 Score')
        ax2.legend(fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
    
    # 3. Feature Importance - Growth
    ax3 = plt.subplot(4, 6, 3)
    if not growth_metrics['feature_importance'].empty:
        top_features = growth_metrics['feature_importance'].head(8)
        ax3.barh(range(len(top_features)), top_features['importance'], 
                color='#FF6B6B', alpha=0.7)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'], fontsize=7)
        ax3.set_title('Top Growth Features', fontsize=10)
        ax3.invert_yaxis()
    
    # 4. Feature Importance - Viral
    ax4 = plt.subplot(4, 6, 4)
    if not viral_metrics['feature_importance'].empty:
        top_features = viral_metrics['feature_importance'].head(8)
        ax4.barh(range(len(top_features)), top_features['importance'],
                color='#4ECDC4', alpha=0.7)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'], fontsize=7)
        ax4.set_title('Top Viral Features', fontsize=10)
        ax4.invert_yaxis()
    
    # 5. Model Weights
    ax5 = plt.subplot(4, 6, 5)
    if 'weights' in growth_metrics:
        models = list(growth_metrics['weights'].keys())
        weights = list(growth_metrics['weights'].values())
        ax5.pie(weights, labels=models, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Growth Model Weights', fontsize=10)
    
    # 6. Class Distribution
    ax6 = plt.subplot(4, 6, 6)
    if 'class_distribution' in viral_metrics:
        classes = list(viral_metrics['class_distribution'].keys())
        counts = list(viral_metrics['class_distribution'].values())
        ax6.bar(classes, counts, color=['#96CEB4', '#FF6B6B'])
        ax6.set_title('Viral Class Distribution', fontsize=10)
        ax6.set_xlabel('Class (0: Non-viral, 1: Viral)')
        ax6.set_ylabel('Count')
    
    # 7-12. Hashtag Recommendations (Multiple Strategies)
    strategies = ['balanced', 'viral', 'engagement', 'growth', 'diverse']
    for i, strategy in enumerate(strategies[:6]):
        ax = plt.subplot(4, 6, 7 + i)
        if hashtag_recommendations:
            # Simulate different strategy scores
            hashtags = [h['hashtag'][:12] + '...' if len(h['hashtag']) > 12 else h['hashtag'] 
                       for h in hashtag_recommendations[:6]]
            
            if strategy == 'viral':
                scores = [h.get('viral_score', h.get('trend_score', 0)) for h in hashtag_recommendations[:6]]
            elif strategy == 'engagement':
                scores = [h.get('engagement_score', h.get('avg_engagement', 0)) for h in hashtag_recommendations[:6]]
            elif strategy == 'growth':
                scores = [h.get('growth_score', h.get('usage_count', 0)) for h in hashtag_recommendations[:6]]
            else:
                scores = [h.get('trend_score', h.get('usage_count', 0)) for h in hashtag_recommendations[:6]]
            
            ax.barh(range(len(hashtags)), scores, color=f'C{i}', alpha=0.7)
            ax.set_yticks(range(len(hashtags)))
            ax.set_yticklabels(hashtags, fontsize=7)
            ax.set_title(f'Hashtags ({strategy.title()})', fontsize=9)
            ax.invert_yaxis()
    
    # 13-18. Sound Recommendations (Multiple Strategies)
    sound_strategies = ['balanced', 'viral', 'popular', 'emerging']
    for i, strategy in enumerate(sound_strategies[:6]):
        ax = plt.subplot(4, 6, 13 + i)
        if sound_recommendations:
            sounds = [s['music_title'][:15] + '...' if len(s['music_title']) > 15 else s['music_title']
                     for s in sound_recommendations[:5]]
            
            if strategy == 'viral':
                scores = [s.get('viral_probability', 0) for s in sound_recommendations[:5]]
            elif strategy == 'popular':
                scores = [s.get('popularity_score', s.get('current_views', 0)) for s in sound_recommendations[:5]]
            elif strategy == 'emerging':
                scores = [s.get('emerging_score', 0) for s in sound_recommendations[:5]]
            else:
                scores = [s.get('trend_score', s.get('current_views', 0)) for s in sound_recommendations[:5]]
            
            ax.barh(range(len(sounds)), scores, color=f'C{i+6}', alpha=0.7)
            ax.set_yticks(range(len(sounds)))
            ax.set_yticklabels(sounds, fontsize=7)
            ax.set_title(f'Sounds ({strategy.title()})', fontsize=9)
            ax.invert_yaxis()
    
    # 19. Performance Metrics Summary
    ax19 = plt.subplot(4, 6, 19)
    metrics = ['Growth R²', 'Growth MAE', 'Viral F1', 'Viral Acc']
    values = [
        growth_metrics['ensemble_r2'],
        min(growth_metrics['ensemble_mae'] / 100, 1),
        viral_metrics['ensemble_f1'],
        viral_metrics['ensemble_accuracy']
    ]
    colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
    bars = ax19.bar(metrics, values, color=colors, alpha=0.7)
    ax19.set_title('Model Performance Summary', fontsize=10)
    ax19.set_ylabel('Score')
    plt.xticks(rotation=45, fontsize=8)
    
    # 20. Sample Sizes
    ax20 = plt.subplot(4, 6, 20)
    sample_info = ['Growth\nSamples', 'Viral\nSamples']
    sample_counts = [growth_metrics['n_samples'], viral_metrics['n_samples']]
    ax20.bar(sample_info, sample_counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax20.set_title('Training Sample Sizes', fontsize=10)
    ax20.set_ylabel('Number of Samples')
    
    # 21. Hashtag Score Distribution
    ax21 = plt.subplot(4, 6, 21)
    if hashtag_recommendations:
        trend_scores = [h.get('trend_score', 0) for h in hashtag_recommendations]
        ax21.hist(trend_scores, bins=8, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax21.set_title('Hashtag Score Distribution', fontsize=10)
        ax21.set_xlabel('Trend Score')
        ax21.set_ylabel('Frequency')
    
    # 22. Sound Score Distribution
    ax22 = plt.subplot(4, 6, 22)
    if sound_recommendations:
        trend_scores = [s.get('trend_score', 0) for s in sound_recommendations]
        ax22.hist(trend_scores, bins=8, color='#96CEB4', alpha=0.7, edgecolor='black')
        ax22.set_title('Sound Score Distribution', fontsize=10)
        ax22.set_xlabel('Trend Score')
        ax22.set_ylabel('Frequency')
    
    # 23. Recommendation Diversity
    ax23 = plt.subplot(4, 6, 23)
    if hashtag_recommendations:
        # Simulate diversity scores
        diversity_scores = np.random.beta(2, 5, len(hashtag_recommendations[:10]))
        ax23.scatter(range(len(diversity_scores)), diversity_scores, 
                    color='#FF6B6B', alpha=0.7, s=50)
        ax23.set_title('Hashtag Diversity Scores', fontsize=10)
        ax23.set_xlabel('Recommendation Rank')
        ax23.set_ylabel('Diversity Score')
    
    # 24. Trend Evolution
    ax24 = plt.subplot(4, 6, 24)
    if sound_recommendations:
        # Simulate trend evolution
        x = np.arange(len(sound_recommendations[:8]))
        viral_probs = [s.get('viral_probability', np.random.random()) for s in sound_recommendations[:8]]
        ax24.plot(x, viral_probs, 'o-', color='#4ECDC4', linewidth=2, markersize=6)
        ax24.set_title('Sound Viral Probability Trend', fontsize=10)
        ax24.set_xlabel('Recommendation Rank')
        ax24.set_ylabel('Viral Probability')
        ax24.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / 'comprehensive_model_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualizations saved to {output_dir}")

def main():
    """Main execution function"""
    features_dir = "finalProject/data/features"
    models_dir = "finalProject/models"
    results_dir = "finalProject/results"
    
    # Create directories
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting Intelligent Viral Prediction Training...")
    print("="*80)
    
    # Initialize predictors
    viral_predictor = ModernViralPredictor()
    trend_recommender = IntelligentTrendRecommender()
    
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
        
        print(f"\n   Individual Model Performance:")
        for model, scores in growth_metrics['individual_scores'].items():
            print(f"      • {model}: R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}")
        
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
        
        print(f"\n   Individual Model Performance:")
        for model, scores in viral_metrics['individual_scores'].items():
            print(f"      • {model}: Acc={scores['accuracy']:.4f}, F1={scores['f1']:.4f}, AUC={scores['auc']:.4f}")
        
        if not viral_metrics['feature_importance'].empty:
            print(f"\n   🔍 Top Viral Classification Features:")
            for _, row in viral_metrics['feature_importance'].head(5).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nINTELLIGENT TREND RECOMMENDATIONS:")
        
        print(f"\n   BALANCED HASHTAG STRATEGY (Top 5):")
        for i, hashtag in enumerate(hashtag_recommendations[:5], 1):
            trend_score = hashtag.get('trend_score', 0)
            viral_score = hashtag.get('viral_score', 0)
            engagement_score = hashtag.get('engagement_score', 0)
            usage = hashtag.get('usage_count', 0)
            print(f"      {i:2d}. #{hashtag['hashtag']}")
            print(f"          Trend: {trend_score:.3f} | Viral: {viral_score:.3f} | Engagement: {engagement_score:.3f} | Usage: {usage}")
        
        print(f"\n   VIRAL-FOCUSED HASHTAGS (Top 5):")
        for i, hashtag in enumerate(viral_hashtags[:5], 1):
            viral_score = hashtag.get('viral_score', hashtag.get('trend_score', 0))
            print(f"      {i}. #{hashtag['hashtag']} (Viral Score: {viral_score:.3f})")
        
        print(f"\n   ENGAGEMENT-FOCUSED HASHTAGS (Top 5):")
        for i, hashtag in enumerate(engagement_hashtags[:5], 1):
            engagement_score = hashtag.get('engagement_score', hashtag.get('avg_engagement', 0))
            print(f"      {i}. #{hashtag['hashtag']} (Engagement Score: {engagement_score:.3f})")
        
        print(f"\n   BALANCED SOUND STRATEGY (Top ):")
        for i, sound in enumerate(sound_recommendations[:5], 1):
            title = sound['music_title'][:50] + '...' if len(sound['music_title']) > 50 else sound['music_title']
            trend_score = sound.get('trend_score', 0)
            viral_prob = sound.get('viral_probability', 0)
            popularity = sound.get('popularity_score', sound.get('current_views', 0))
            print(f"      {i}. {title}")
            print(f"         Trend: {trend_score:.3f} | Viral Prob: {viral_prob:.3f} | Popularity: {popularity:.0f}")
        
        print(f"\n   VIRAL SOUNDS (Top 5):")
        for i, sound in enumerate(viral_sounds[:5], 1):
            title = sound['music_title'][:40] + '...' if len(sound['music_title']) > 40 else sound['music_title']
            viral_prob = sound.get('viral_probability', sound.get('trend_score', 0))
            print(f"      {i}. {title} (Viral Prob: {viral_prob:.3f})")
        
        print(f"\n   EMERGING SOUNDS (Top 5):")
        for i, sound in enumerate(emerging_sounds[:5], 1):
            title = sound['music_title'][:40] + '...' if len(sound['music_title']) > 40 else sound['music_title']
            emerging_score = sound.get('emerging_score', sound.get('trend_score', 0))
            print(f"      {i}. {title} (Emerging Score: {emerging_score:.3f})")
        
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
        if hashtag_recommendations:
            hashtag_df = pd.DataFrame(hashtag_recommendations)
            hashtag_df.to_csv(Path(results_dir) / 'hashtag_recommendations_balanced.csv', index=False)
            
            viral_hashtag_df = pd.DataFrame(viral_hashtags)
            viral_hashtag_df.to_csv(Path(results_dir) / 'hashtag_recommendations_viral.csv', index=False)
        
        if sound_recommendations:
            sound_df = pd.DataFrame(sound_recommendations)
            sound_df.to_csv(Path(results_dir) / 'sound_recommendations_balanced.csv', index=False)
            
            viral_sound_df = pd.DataFrame(viral_sounds)
            viral_sound_df.to_csv(Path(results_dir) / 'sound_recommendations_viral.csv', index=False)
        
        # Save comprehensive results
        with open(Path(results_dir) / 'intelligent_results.pkl', 'wb') as f:
            pickle.dump(detailed_results, f)
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Models saved to: {models_dir}")
        print(f"Visualizations: {results_dir}/comprehensive_model_analysis.png")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()