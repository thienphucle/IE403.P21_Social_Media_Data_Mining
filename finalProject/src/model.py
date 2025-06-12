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
        """Tạo các features nâng cao cho viral prediction"""
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
        """Load và chuẩn bị features với kỹ thuật nâng cao"""
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

class AdvancedTrendRecommender:
    """
    Sử dụng LightGBM cho trend recommendation
    - Nhanh và hiệu quả
    - Xử lý tốt categorical features
    - Built-in feature importance
    """
    
    def __init__(self):
        self.hashtag_model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.sound_model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.trending_features = None
        
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
        """Engineer advanced features for hashtag trends"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        features = []
        
        # Basic features
        features.append(trends_df['total_views'].fillna(0).values)
        features.append(trends_df['avg_engagement'].fillna(0).values)
        features.append(trends_df['usage_count'].fillna(0).values)
        
        # Advanced features
        # Viral potential score
        viral_potential = (
            np.log1p(trends_df['total_views'].fillna(0)) * 0.4 +
            trends_df['avg_engagement'].fillna(0) * 0.4 +
            np.log1p(trends_df['usage_count'].fillna(0)) * 0.2
        )
        features.append(viral_potential.values)
        
        # Engagement efficiency
        engagement_efficiency = np.where(
            trends_df['usage_count'] > 0,
            trends_df['avg_engagement'] / trends_df['usage_count'],
            0
        )
        features.append(engagement_efficiency)
        
        # Hashtag length (if available)
        if 'hashtag' in trends_df.columns:
            hashtag_length = trends_df['hashtag'].str.len().fillna(0)
            features.append(hashtag_length.values)
        
        return np.column_stack(features)
    
    def engineer_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Engineer advanced features for sound trends"""
        if trends_df.empty:
            return np.array([]).reshape(0, 0)
        
        features = []
        
        # Basic features
        for col in ['current_views', 'current_engagement_rate', 'new_growth_rate']:
            if col in trends_df.columns:
                features.append(trends_df[col].fillna(0).values)
            else:
                features.append(np.zeros(len(trends_df)))
        
        # Advanced features
        if len(features) >= 3:
            # Sound momentum
            momentum = features[0] * features[1] * features[2]  # views × engagement × growth
            features.append(momentum)
            
            # Viral coefficient
            viral_coeff = np.where(
                features[0] > 0,
                features[1] * features[2] / np.log1p(features[0]),
                0
            )
            features.append(viral_coeff)
        
        return np.column_stack(features)
    
    def train_recommenders(self):
        """Train advanced recommendation models"""
        if not self.trending_features:
            print("No trending features available for training")
            return
        
        # Train hashtag recommender
        hashtag_df = self.trending_features['hashtag_trends']
        if not hashtag_df.empty and len(hashtag_df) > 5:
            print("Training hashtag recommender...")
            hashtag_features = self.engineer_hashtag_features(hashtag_df)
            
            if hashtag_features.size > 0:
                # Target: weighted combination of metrics
                hashtag_target = (
                    0.4 * np.log1p(hashtag_df['total_views'].fillna(0)) +
                    0.4 * hashtag_df['avg_engagement'].fillna(0) +
                    0.2 * np.log1p(hashtag_df['usage_count'].fillna(0))
                ).values
                
                if np.any(hashtag_target > 0):
                    self.hashtag_model.fit(hashtag_features, hashtag_target)
                    print(f"Hashtag recommender trained on {len(hashtag_target)} samples")
        
        # Train sound recommender
        sound_df = self.trending_features['sound_trends']
        if not sound_df.empty and len(sound_df) > 5:
            print("Training sound recommender...")
            sound_features = self.engineer_sound_features(sound_df)
            
            if sound_features.size > 0:
                # Target: current views as proxy for popularity
                sound_target = sound_df['current_views'].fillna(0).values
                
                if np.any(sound_target > 0):
                    self.sound_model.fit(sound_features, sound_target)
                    print(f"Sound recommender trained on {len(sound_target)} samples")
    
    def recommend_hashtags(self, n_recommendations: int = 10) -> List[Dict]:
        """Get advanced hashtag recommendations"""
        if not self.trending_features or self.trending_features['hashtag_trends'].empty:
            return []
        
        hashtag_df = self.trending_features['hashtag_trends']
        features = self.engineer_hashtag_features(hashtag_df)
        
        if features.size == 0:
            return []
        
        try:
            predictions = self.hashtag_model.predict(features)
            recommendations = hashtag_df.copy()
            recommendations['trend_score'] = predictions
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            return recommendations.head(n_recommendations)[
                ['hashtag', 'trend_score', 'total_views', 'avg_engagement', 'usage_count']
            ].to_dict('records')
        except Exception as e:
            print(f"Error in hashtag recommendation: {e}")
            # Fallback
            return hashtag_df.nlargest(n_recommendations, 'usage_count')[
                ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
            ].to_dict('records')
    
    def recommend_sounds(self, n_recommendations: int = 10) -> List[Dict]:
        """Get advanced sound recommendations"""
        if not self.trending_features or self.trending_features['sound_trends'].empty:
            return []
        
        sound_df = self.trending_features['sound_trends']
        features = self.engineer_sound_features(sound_df)
        
        if features.size == 0:
            return []
        
        try:
            predictions = self.sound_model.predict(features)
            recommendations = sound_df.copy()
            recommendations['trend_score'] = predictions
            recommendations = recommendations.sort_values('trend_score', ascending=False)
            
            return recommendations.head(n_recommendations)[
                ['music_id', 'music_title', 'trend_score', 'current_views', 'current_engagement_rate']
            ].to_dict('records')
        except Exception as e:
            print(f"Error in sound recommendation: {e}")
            # Fallback
            return sound_df.nlargest(n_recommendations, 'current_views')[
                ['music_id', 'music_title', 'current_views', 'current_engagement_rate']
            ].to_dict('records')
    
    def save_models(self, output_dir: str):
        """Save recommendation models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'hashtag_recommender.pkl', 'wb') as f:
            pickle.dump(self.hashtag_model, f)
        
        with open(output_dir / 'sound_recommender.pkl', 'wb') as f:
            pickle.dump(self.sound_model, f)

def create_comprehensive_visualizations(growth_metrics: Dict, viral_metrics: Dict, 
                                      hashtag_recommendations: List, sound_recommendations: List,
                                      output_dir: str):
    """Create comprehensive visualization dashboard"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    if 'individual_scores' in growth_metrics:
        models = list(growth_metrics['individual_scores'].keys())
        r2_scores = [growth_metrics['individual_scores'][m]['r2'] for m in models]
        bars = ax1.bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.axhline(y=growth_metrics['ensemble_r2'], color='red', linestyle='--', 
                   label=f'Ensemble: {growth_metrics["ensemble_r2"]:.3f}')
        ax1.set_title('Growth Prediction R² Scores')
        ax1.set_ylabel('R² Score')
        ax1.legend()
        plt.xticks(rotation=45)
    
    # 2. Viral Classification Performance
    ax2 = plt.subplot(3, 4, 2)
    if 'individual_scores' in viral_metrics:
        models = list(viral_metrics['individual_scores'].keys())
        f1_scores = [viral_metrics['individual_scores'][m]['f1'] for m in models]
        bars = ax2.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.axhline(y=viral_metrics['ensemble_f1'], color='red', linestyle='--',
                   label=f'Ensemble: {viral_metrics["ensemble_f1"]:.3f}')
        ax2.set_title('Viral Classification F1 Scores')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        plt.xticks(rotation=45)
    
    # 3. Feature Importance - Growth
    ax3 = plt.subplot(3, 4, 3)
    if not growth_metrics['feature_importance'].empty:
        top_features = growth_metrics['feature_importance'].head(8)
        ax3.barh(range(len(top_features)), top_features['importance'], 
                color='#FF6B6B', alpha=0.7)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'], fontsize=8)
        ax3.set_title('Top Growth Features')
        ax3.invert_yaxis()
    
    # 4. Feature Importance - Viral
    ax4 = plt.subplot(3, 4, 4)
    if not viral_metrics['feature_importance'].empty:
        top_features = viral_metrics['feature_importance'].head(8)
        ax4.barh(range(len(top_features)), top_features['importance'],
                color='#4ECDC4', alpha=0.7)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'], fontsize=8)
        ax4.set_title('Top Viral Features')
        ax4.invert_yaxis()
    
    # 5. Model Weights
    ax5 = plt.subplot(3, 4, 5)
    if 'weights' in growth_metrics:
        models = list(growth_metrics['weights'].keys())
        weights = list(growth_metrics['weights'].values())
        ax5.pie(weights, labels=models, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Growth Model Weights')
    
    # 6. Class Distribution
    ax6 = plt.subplot(3, 4, 6)
    if 'class_distribution' in viral_metrics:
        classes = list(viral_metrics['class_distribution'].keys())
        counts = list(viral_metrics['class_distribution'].values())
        ax6.bar(classes, counts, color=['#96CEB4', '#FF6B6B'])
        ax6.set_title('Viral Class Distribution')
        ax6.set_xlabel('Class (0: Non-viral, 1: Viral)')
        ax6.set_ylabel('Count')
    
    # 7. Top Hashtags
    ax7 = plt.subplot(3, 4, 7)
    if hashtag_recommendations:
        hashtags = [h['hashtag'][:15] + '...' if len(h['hashtag']) > 15 else h['hashtag'] 
                   for h in hashtag_recommendations[:8]]
        scores = [h.get('trend_score', h.get('usage_count', 0)) for h in hashtag_recommendations[:8]]
        ax7.barh(range(len(hashtags)), scores, color='#45B7D1', alpha=0.7)
        ax7.set_yticks(range(len(hashtags)))
        ax7.set_yticklabels(hashtags, fontsize=8)
        ax7.set_title('Top Trending Hashtags')
        ax7.invert_yaxis()
    
    # 8. Top Sounds
    ax8 = plt.subplot(3, 4, 8)
    if sound_recommendations:
        sounds = [s['music_title'][:20] + '...' if len(s['music_title']) > 20 else s['music_title']
                 for s in sound_recommendations[:6]]
        scores = [s.get('trend_score', s.get('current_views', 0)) for s in sound_recommendations[:6]]
        ax8.barh(range(len(sounds)), scores, color='#96CEB4', alpha=0.7)
        ax8.set_yticks(range(len(sounds)))
        ax8.set_yticklabels(sounds, fontsize=8)
        ax8.set_title('Top Trending Sounds')
        ax8.invert_yaxis()
    
    # 9. Performance Metrics Summary
    ax9 = plt.subplot(3, 4, 9)
    metrics = ['Growth R²', 'Growth MAE', 'Viral F1', 'Viral Acc']
    values = [
        growth_metrics['ensemble_r2'],
        growth_metrics['ensemble_mae'] / 100,  # Scale for visualization
        viral_metrics['ensemble_f1'],
        viral_metrics['ensemble_accuracy']
    ]
    colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
    bars = ax9.bar(metrics, values, color=colors, alpha=0.7)
    ax9.set_title('Model Performance Summary')
    ax9.set_ylabel('Score')
    plt.xticks(rotation=45)
    
    # 10. Sample Sizes
    ax10 = plt.subplot(3, 4, 10)
    sample_info = ['Growth\nSamples', 'Viral\nSamples']
    sample_counts = [growth_metrics['n_samples'], viral_metrics['n_samples']]
    ax10.bar(sample_info, sample_counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax10.set_title('Training Sample Sizes')
    ax10.set_ylabel('Number of Samples')
    
    # 11. Hashtag Metrics Distribution
    ax11 = plt.subplot(3, 4, 11)
    if hashtag_recommendations:
        usage_counts = [h.get('usage_count', 0) for h in hashtag_recommendations[:10]]
        ax11.hist(usage_counts, bins=5, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax11.set_title('Hashtag Usage Distribution')
        ax11.set_xlabel('Usage Count')
        ax11.set_ylabel('Frequency')
    
    # 12. Sound Metrics Distribution
    ax12 = plt.subplot(3, 4, 12)
    if sound_recommendations:
        view_counts = [s.get('current_views', 0) for s in sound_recommendations[:10]]
        ax12.hist(view_counts, bins=5, color='#96CEB4', alpha=0.7, edgecolor='black')
        ax12.set_title('Sound Views Distribution')
        ax12.set_xlabel('Current Views')
        ax12.set_ylabel('Frequency')
    
    plt.tight_layout()
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
    
    print("Starting Modern Viral Prediction Training...")
    print("="*60)
    
    # Initialize predictors
    viral_predictor = ModernViralPredictor()
    trend_recommender = AdvancedTrendRecommender()
    
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
        print("\nTraining trend recommenders...")
        trend_recommender.train_recommenders()
        
        # Get recommendations
        hashtag_recommendations = trend_recommender.recommend_hashtags(15)
        sound_recommendations = trend_recommender.recommend_sounds(10)
        
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
        print("\n" + "="*80)
        print("MODERN VIRAL PREDICTION RESULTS")
        print("="*80)
        
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
            print(f"\n   Top Viral Classification Features:")
            for _, row in viral_metrics['feature_importance'].head(5).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nADVANCED TREND RECOMMENDATIONS:")
        print(f"\n   Top Trending Hashtags:")
        for i, hashtag in enumerate(hashtag_recommendations[:8], 1):
            score = hashtag.get('trend_score', hashtag.get('usage_count', 0))
            usage = hashtag.get('usage_count', 0)
            print(f"      {i}. #{hashtag['hashtag']} (Score: {score:.2f}, Usage: {usage})")
        
        print(f"\n   Top Trending Sounds:")
        for i, sound in enumerate(sound_recommendations[:5], 1):
            score = sound.get('trend_score', sound.get('current_views', 0))
            title = sound['music_title'][:40] + '...' if len(sound['music_title']) > 40 else sound['music_title']
            views = sound.get('current_views', 0)
            print(f"      {i}. {title}")
            print(f"         Score: {score:.2f}, Views: {views:,}")
        
        # Save results as CSV
        if hashtag_recommendations:
            hashtag_df = pd.DataFrame(hashtag_recommendations)
            hashtag_df.to_csv(Path(results_dir) / 'recommended_hashtags.csv', index=False)
        
        if sound_recommendations:
            sound_df = pd.DataFrame(sound_recommendations)
            sound_df.to_csv(Path(results_dir) / 'recommended_sounds.csv', index=False)
        
        # Save comprehensive results
        results_summary = {
            'growth_metrics': growth_metrics,
            'viral_metrics': viral_metrics,
            'hashtag_recommendations': hashtag_recommendations,
            'sound_recommendations': sound_recommendations,
            'model_info': {
                'growth_models': list(viral_predictor.growth_models.keys()),
                'viral_models': list(viral_predictor.viral_models.keys()),
                'feature_count': len(viral_predictor.feature_names) if viral_predictor.feature_names else 0
            }
        }
        
        with open(Path(results_dir) / 'comprehensive_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)
        
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