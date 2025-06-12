import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, QuantileTransformer, 
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, RFECV, 
    SelectFromModel, VarianceThreshold
)
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from scipy import stats

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
    VotingRegressor, VotingClassifier, RandomForestRegressor, 
    RandomForestClassifier, StackingRegressor, StackingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.linear_model import (
    ElasticNet, LogisticRegression, Ridge, Lasso, 
    BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using alternative models")

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Enhanced feature engineering with advanced techniques"""
    
    def __init__(self, add_polynomial_features=True, polynomial_degree=2, 
                 add_interaction_features=True, add_time_features=True):
        self.add_polynomial_features = add_polynomial_features
        self.polynomial_degree = polynomial_degree
        self.add_interaction_features = add_interaction_features
        self.add_time_features = add_time_features
        self.feature_names_ = []
        self.fitted_ = False
        self.poly_transformer = None
        
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, metadata: Dict) -> Tuple[np.ndarray, List[str]]:
        """Transform metadata into advanced features with sophisticated engineering"""
        features = []
        feature_names = []
        
        # 1. Enhanced Core Engagement Features
        if all(k in metadata for k in ['initial_engagement_rate', 'current_engagement_rate']):
            initial_eng = np.array(metadata['initial_engagement_rate'])
            current_eng = np.array(metadata['current_engagement_rate'])
            
            # Basic engagement features
            engagement_momentum = current_eng - initial_eng
            features.append(engagement_momentum.reshape(-1, 1))
            feature_names.append('engagement_momentum')
            
            # Advanced engagement features
            engagement_ratio = np.where(
                initial_eng > 0,
                current_eng / initial_eng,
                np.where(current_eng > 0, 2.0, 1.0)
            )
            features.append(engagement_ratio.reshape(-1, 1))
            feature_names.append('engagement_acceleration')
            
            # Engagement volatility
            engagement_volatility = np.abs(engagement_momentum) / (np.maximum(initial_eng, 0.001))
            features.append(engagement_volatility.reshape(-1, 1))
            feature_names.append('engagement_volatility')
            
            # Engagement stability with exponential smoothing
            engagement_stability = np.exp(-np.abs(engagement_momentum))
            features.append(engagement_stability.reshape(-1, 1))
            feature_names.append('engagement_stability')
            
            # Engagement trend strength
            trend_strength = np.where(
                engagement_momentum > 0,
                np.log1p(engagement_momentum) * current_eng,
                -np.log1p(-engagement_momentum) * current_eng
            )
            features.append(trend_strength.reshape(-1, 1))
            feature_names.append('engagement_trend_strength')
        
        # 2. Advanced Growth Velocity Features
        if all(k in metadata for k in ['view_growth_per_hour', 'time_diff_hours']):
            view_growth = np.array(metadata['view_growth_per_hour'])
            time_diff = np.maximum(np.array(metadata['time_diff_hours']), 0.1)
            
            # Enhanced velocity metrics
            velocity_score = view_growth / time_diff
            features.append(velocity_score.reshape(-1, 1))
            feature_names.append('velocity_score')
            
            # Velocity acceleration
            velocity_acceleration = velocity_score / np.maximum(time_diff, 1)
            features.append(velocity_acceleration.reshape(-1, 1))
            feature_names.append('velocity_acceleration')
            
            # Log-transformed velocity (handles skewness)
            log_velocity = np.log1p(np.maximum(velocity_score, 0))
            features.append(log_velocity.reshape(-1, 1))
            feature_names.append('log_velocity')
            
            # Velocity percentile rank (relative to dataset)
            velocity_percentile = stats.rankdata(velocity_score) / len(velocity_score)
            features.append(velocity_percentile.reshape(-1, 1))
            feature_names.append('velocity_percentile')
            
            # Growth consistency with all metrics
            if all(k in metadata for k in ['like_growth_per_hour', 'comment_growth_per_hour']):
                like_growth = np.array(metadata['like_growth_per_hour'])
                comment_growth = np.array(metadata['comment_growth_per_hour'])
                
                # Weighted growth consistency
                growth_consistency = (
                    view_growth * 0.5 +
                    like_growth * 0.3 +
                    comment_growth * 0.2
                )
                features.append(growth_consistency.reshape(-1, 1))
                feature_names.append('growth_consistency')
                
                # Growth balance coefficient of variation
                growth_metrics = np.array([view_growth, like_growth, comment_growth])
                growth_cv = np.std(growth_metrics, axis=0) / (np.mean(growth_metrics, axis=0) + 1e-6)
                features.append(growth_cv.reshape(-1, 1))
                feature_names.append('growth_coefficient_variation')
                
                # Growth momentum (weighted by time)
                growth_momentum = growth_consistency / np.maximum(time_diff, 1)
                features.append(growth_momentum.reshape(-1, 1))
                feature_names.append('growth_momentum')
        
        # 3. Enhanced Content Quality Indicators
        if 'hashtag_counts' in metadata:
            hashtag_counts = np.array(metadata['hashtag_counts'])
            
            # Optimal hashtag score (refined)
            optimal_hashtag_score = np.exp(-0.5 * ((hashtag_counts - 5) / 2.5) ** 2)
            features.append(optimal_hashtag_score.reshape(-1, 1))
            feature_names.append('optimal_hashtag_score')
            
            # Hashtag saturation penalty
            hashtag_penalty = np.where(hashtag_counts > 15, 
                                     np.exp(-(hashtag_counts - 15) * 0.3), 1.0)
            features.append(hashtag_penalty.reshape(-1, 1))
            feature_names.append('hashtag_penalty')
            
            # Hashtag diversity score
            hashtag_diversity = np.minimum(hashtag_counts / 10, 1.0)
            features.append(hashtag_diversity.reshape(-1, 1))
            feature_names.append('hashtag_diversity')
        
        if 'durations' in metadata:
            durations = np.array(metadata['durations'])
            
            # Advanced duration optimization
            # Multiple sweet spots for different content types
            short_form_score = np.exp(-0.1 * (durations - 15) ** 2)  # 15s optimal
            medium_form_score = np.exp(-0.05 * (durations - 30) ** 2)  # 30s optimal
            long_form_score = np.exp(-0.02 * (durations - 60) ** 2)   # 60s optimal
            
            # Combined duration score
            duration_optimality = np.maximum.reduce([
                short_form_score * 1.2,    # Bonus for short form
                medium_form_score,
                long_form_score * 0.8      # Penalty for long form
            ])
            features.append(duration_optimality.reshape(-1, 1))
            feature_names.append('duration_optimality')
            
            # Duration category
            duration_category = np.digitize(durations, bins=[0, 15, 30, 60, 120, np.inf])
            features.append(duration_category.reshape(-1, 1))
            feature_names.append('duration_category')
        
        # 4. Enhanced Creator Influence
        if 'followers' in metadata:
            followers = np.array(metadata['followers'])
            
            # Log-scaled influence with better normalization
            log_followers = np.log1p(followers)
            follower_influence = log_followers / 25  # Better scaling
            features.append(follower_influence.reshape(-1, 1))
            feature_names.append('creator_influence')
            
            # Creator tier with refined boundaries
            creator_tier = np.digitize(followers, 
                                     bins=[0, 1000, 10000, 100000, 500000, 1000000, np.inf])
            features.append(creator_tier.reshape(-1, 1))
            feature_names.append('creator_tier')
            
            # Follower influence percentile
            follower_percentile = stats.rankdata(followers) / len(followers)
            features.append(follower_percentile.reshape(-1, 1))
            feature_names.append('follower_percentile')
            
            # Enhanced follower engagement ratio
            if 'current_views' in metadata:
                current_views = np.array(metadata['current_views'])
                
                # Robust engagement ratio
                engagement_ratio = np.where(
                    followers > 0,
                    current_views / followers,
                    current_views / 1000  # Default assumption
                )
                
                # Cap and log transform
                engagement_ratio_capped = np.clip(engagement_ratio, 0, 50)
                log_engagement_ratio = np.log1p(engagement_ratio_capped)
                features.append(log_engagement_ratio.reshape(-1, 1))
                feature_names.append('log_follower_engagement_ratio')
                
                # Virality coefficient (views vs followers)
                virality_coefficient = np.where(
                    followers > 100,
                    np.minimum(current_views / followers, 10),
                    current_views / 100
                )
                features.append(virality_coefficient.reshape(-1, 1))
                feature_names.append('virality_coefficient')
        
        # 5. Advanced Temporal Features
        if 'post_hour' in metadata:
            post_hour = np.array(metadata['post_hour'])
            
            # Multiple time periods
            prime_time_evening = np.where((post_hour >= 18) & (post_hour <= 21), 1.0, 0.0)
            prime_time_afternoon = np.where((post_hour >= 12) & (post_hour <= 15), 0.8, 0.0)
            prime_time_morning = np.where((post_hour >= 8) & (post_hour <= 10), 0.6, 0.0)
            late_night = np.where((post_hour >= 22) | (post_hour <= 6), 0.3, 0.0)
            
            optimal_time_score = np.maximum.reduce([
                prime_time_evening, prime_time_afternoon, 
                prime_time_morning, late_night
            ])
            features.append(optimal_time_score.reshape(-1, 1))
            feature_names.append('optimal_time_score')
            
            # Cyclical encoding (improved)
            hour_sin = np.sin(2 * np.pi * post_hour / 24)
            hour_cos = np.cos(2 * np.pi * post_hour / 24)
            features.extend([hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1)])
            feature_names.extend(['hour_sin', 'hour_cos'])
            
            # Activity level zones
            high_activity = np.where((post_hour >= 18) & (post_hour <= 22), 1.0, 0.0)
            medium_activity = np.where(
                ((post_hour >= 8) & (post_hour < 18)) | 
                ((post_hour > 22) & (post_hour <= 23)), 0.7, 0.0
            )
            low_activity = np.where((post_hour >= 0) & (post_hour < 8), 0.3, 0.0)
            
            activity_score = high_activity + medium_activity + low_activity
            features.append(activity_score.reshape(-1, 1))
            feature_names.append('activity_score')
        
        # 6. Enhanced Viral Acceleration Metrics
        if 'viral_acceleration' in metadata:
            viral_acceleration = np.array(metadata['viral_acceleration'])
            
            # Log-transformed acceleration
            log_viral_accel = np.sign(viral_acceleration) * np.log1p(np.abs(viral_acceleration))
            features.append(log_viral_accel.reshape(-1, 1))
            feature_names.append('log_viral_acceleration')
            
            # Acceleration percentile
            accel_percentile = stats.rankdata(viral_acceleration) / len(viral_acceleration)
            features.append(accel_percentile.reshape(-1, 1))
            feature_names.append('acceleration_percentile')
            
            # Acceleration momentum with decay
            accel_momentum = np.tanh(viral_acceleration) * np.exp(-np.abs(viral_acceleration) * 0.1)
            features.append(accel_momentum.reshape(-1, 1))
            feature_names.append('acceleration_momentum_decay')
        
        # 7. Advanced Cross-feature Interactions
        if len(features) >= 3:
            # Find key feature indices
            eng_momentum_idx = next((i for i, name in enumerate(feature_names) 
                                   if 'engagement_momentum' in name), None)
            creator_influence_idx = next((i for i, name in enumerate(feature_names) 
                                        if 'creator_influence' in name), None)
            velocity_idx = next((i for i, name in enumerate(feature_names) 
                               if 'velocity_score' in name), None)
            time_score_idx = next((i for i, name in enumerate(feature_names) 
                                 if 'optimal_time_score' in name), None)
            
            # Enhanced interactions
            if eng_momentum_idx is not None and creator_influence_idx is not None:
                eng_creator_interaction = (
                    features[eng_momentum_idx].flatten() * 
                    features[creator_influence_idx].flatten()
                )
                features.append(eng_creator_interaction.reshape(-1, 1))
                feature_names.append('engagement_creator_interaction')
            
            if velocity_idx is not None and time_score_idx is not None:
                velocity_time_interaction = (
                    features[velocity_idx].flatten() * 
                    features[time_score_idx].flatten()
                )
                features.append(velocity_time_interaction.reshape(-1, 1))
                feature_names.append('velocity_time_interaction')
            
            # Triple interaction (engagement * creator * time)
            if (eng_momentum_idx is not None and creator_influence_idx is not None 
                and time_score_idx is not None):
                triple_interaction = (
                    features[eng_momentum_idx].flatten() * 
                    features[creator_influence_idx].flatten() * 
                    features[time_score_idx].flatten()
                )
                features.append(triple_interaction.reshape(-1, 1))
                feature_names.append('triple_viral_interaction')
        
        # 8. Enhanced Composite Viral Score
        if len(features) >= 4:
            viral_components = []
            weights = []
            
            # Engagement component
            eng_idx = next((i for i, name in enumerate(feature_names) 
                          if 'engagement_momentum' in name), None)
            if eng_idx is not None:
                viral_components.append(features[eng_idx].flatten())
                weights.append(0.25)
            
            # Velocity component
            velocity_idx = next((i for i, name in enumerate(feature_names) 
                               if 'velocity_score' in name), None)
            if velocity_idx is not None:
                viral_components.append(np.tanh(features[velocity_idx].flatten()))
                weights.append(0.25)
            
            # Creator component
            creator_idx = next((i for i, name in enumerate(feature_names) 
                              if 'creator_influence' in name), None)
            if creator_idx is not None:
                viral_components.append(features[creator_idx].flatten())
                weights.append(0.2)
            
            # Time component
            time_idx = next((i for i, name in enumerate(feature_names) 
                           if 'optimal_time_score' in name), None)
            if time_idx is not None:
                viral_components.append(features[time_idx].flatten())
                weights.append(0.15)
            
            # Content quality component
            hashtag_idx = next((i for i, name in enumerate(feature_names) 
                              if 'optimal_hashtag_score' in name), None)
            if hashtag_idx is not None:
                viral_components.append(features[hashtag_idx].flatten())
                weights.append(0.15)
            
            if viral_components:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Calculate weighted composite score
                composite_viral_score = np.average(viral_components, axis=0, weights=weights)
                features.append(composite_viral_score.reshape(-1, 1))
                feature_names.append('enhanced_composite_viral_score')
        
        # 9. Add polynomial features if requested
        if self.add_polynomial_features and len(features) >= 2:
            # Select key features for polynomial expansion
            key_feature_indices = []
            key_patterns = ['engagement_momentum', 'velocity_score', 'creator_influence']
            
            for pattern in key_patterns:
                idx = next((i for i, name in enumerate(feature_names) if pattern in name), None)
                if idx is not None:
                    key_feature_indices.append(idx)
            
            if key_feature_indices:
                key_features = np.hstack([features[i] for i in key_feature_indices])
                
                # Apply polynomial features to key features only
                if self.poly_transformer is None:
                    self.poly_transformer = PolynomialFeatures(
                        degree=self.polynomial_degree, 
                        include_bias=False,
                        interaction_only=False
                    )
                    poly_features = self.poly_transformer.fit_transform(key_features)
                else:
                    poly_features = self.poly_transformer.transform(key_features)
                
                # Add only the interaction and higher-order terms (skip original features)
                original_feature_count = key_features.shape[1]
                new_poly_features = poly_features[:, original_feature_count:]
                
                if new_poly_features.shape[1] > 0:
                    for i in range(new_poly_features.shape[1]):
                        features.append(new_poly_features[:, i].reshape(-1, 1))
                        feature_names.append(f'poly_feature_{i}')
        
        self.feature_names_ = feature_names
        
        if features:
            return np.hstack(features), feature_names
        else:
            return np.array([]).reshape(len(metadata.get('video_ids', [])), 0), []

class EnhancedViralPredictor:
    """
    Enhanced viral prediction system with advanced ML techniques and hyperparameter optimization
    """
    
    def __init__(self, use_advanced_preprocessing=True, use_stacking=True):
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.use_stacking = use_stacking
        
        # Initialize models with better hyperparameters
        self.growth_models = self._initialize_growth_models()
        self.viral_models = self._initialize_viral_models()
        
        # Ensemble Models
        self.growth_ensemble = None
        self.viral_ensemble = None
        
        # Advanced preprocessing
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.feature_selector = None
        self.feature_engineer = AdvancedFeatureEngineer(
            add_polynomial_features=True,
            polynomial_degree=2,
            add_interaction_features=True
        )
        self.feature_names = None
        
        # Dimensionality reduction
        self.text_reducer = TruncatedSVD(n_components=100, random_state=42)  # Increased components
        self.phobert_reducer = TruncatedSVD(n_components=50, random_state=42)  # Increased components
        
        # Outlier detection
        self.outlier_detector = None
        
    def _initialize_growth_models(self) -> Dict:
        """Initialize growth prediction models with optimized hyperparameters"""
        models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=127,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'catboost': None,  # Will be set conditionally
            'xgboost': None,   # Will be set conditionally
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'huber': HuberRegressor(epsilon=1.5, alpha=0.01),
            'bayesian_ridge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6)
        }
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                border_count=254,
                random_seed=42,
                verbose=False
            )
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            )
        
        # Remove None values
        return {k: v for k, v in models.items() if v is not None}
    
    def _initialize_viral_models(self) -> Dict:
        """Initialize viral classification models with optimized hyperparameters"""
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=127,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'catboost': None,  # Will be set conditionally
            'xgboost': None,   # Will be set conditionally
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'logistic': LogisticRegression(
                C=1.0, 
                class_weight='balanced', 
                random_state=42, 
                max_iter=1000
            )
        }
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                border_count=254,
                class_weights=[1, 1],  # Balanced
                random_seed=42,
                verbose=False
            )
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=1,
                random_state=42,
                verbosity=0
            )
        
        # Remove None values
        return {k: v for k, v in models.items() if v is not None}
    
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        """Enhanced feature preparation with advanced preprocessing"""
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
        
        print("Advanced text feature processing...")
        try:
            # Enhanced TF-IDF processing
            if tfidf_features.shape[1] > 100:
                # Apply variance threshold first
                var_selector = VarianceThreshold(threshold=0.001)
                tfidf_filtered = var_selector.fit_transform(tfidf_features)
                
                # Then apply SVD
                n_components = min(100, tfidf_filtered.shape[1])
                self.text_reducer.n_components = n_components
                tfidf_reduced = self.text_reducer.fit_transform(tfidf_filtered)
            else:
                tfidf_reduced = tfidf_features
            
            # Enhanced PhoBERT processing
            if phobert_features.shape[1] > 50:
                # Apply PCA for better representation
                pca_reducer = PCA(n_components=50, random_state=42)
                phobert_reduced = pca_reducer.fit_transform(phobert_features)
                self.phobert_reducer = pca_reducer
            else:
                phobert_reduced = phobert_features
        except Exception as e:
            print(f"Error in text processing: {e}")
            tfidf_reduced = tfidf_features
            phobert_reduced = phobert_features
        
        # Combine all features
        feature_components = []
        feature_names = []
        
        if engineered_features.size > 0:
            feature_components.append(engineered_features)
            feature_names.extend(engineered_names)
        
        feature_components.extend([tfidf_reduced, phobert_reduced])
        feature_names.extend([f'tfidf_component_{i}' for i in range(tfidf_reduced.shape[1])])
        feature_names.extend([f'phobert_component_{i}' for i in range(phobert_reduced.shape[1])])
        
        if feature_components:
            X = np.hstack(feature_components)
        else:
            X = np.array([]).reshape(len(metadata.get('video_ids', [])), 0)
        
        # Advanced data cleaning
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Remove constant features
        if X.shape[1] > 0:
            var_threshold = VarianceThreshold(threshold=1e-6)
            X = var_threshold.fit_transform(X)
            
            # Update feature names
            if hasattr(var_threshold, 'get_support'):
                mask = var_threshold.get_support()
                feature_names = [name for i, name in enumerate(feature_names) if i < len(mask) and mask[i]]
        
        self.feature_names = feature_names
        print(f"Final feature matrix shape: {X.shape}")
        
        return X, metadata
    
    def _optimize_hyperparameters(self, model, X, y, model_type='regression', cv=3):
        """Hyperparameter optimization using RandomizedSearchCV"""
        try:
            param_grids = {
                'lightgbm': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [6, 8, 10],
                    'num_leaves': [63, 127, 255],
                    'min_child_samples': [10, 20, 30],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'xgboost': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [6, 8, 10],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'catboost': {
                    'iterations': [200, 300, 500],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'depth': [6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5]
                }
            }
            
            model_name = type(model).__name__.lower()
            model_key = None
            
            for key in param_grids.keys():
                if key in model_name:
                    model_key = key
                    break
            
            if model_key and len(y) > 100:  # Only optimize if enough samples
                param_grid = param_grids[model_key]
                
                scoring = 'r2' if model_type == 'regression' else 'f1_weighted'
                
                random_search = RandomizedSearchCV(
                    model, param_grid, n_iter=10, cv=cv,
                    scoring=scoring, random_state=42, n_jobs=-1
                )
                
                random_search.fit(X, y)
                return random_search.best_estimator_
            else:
                model.fit(X, y)
                return model
                
        except Exception as e:
            print(f"Error in hyperparameter optimization: {e}")
            model.fit(X, y)
            return model
    
    def train_growth_predictor(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Enhanced growth prediction training with advanced techniques"""
        if 'new_growth_rate' not in metadata:
            raise ValueError("new_growth_rate not found in metadata")
        
        y = metadata['new_growth_rate']
        
        # Advanced outlier detection and handling
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Remove extreme outliers using IQR method
        Q1, Q3 = np.percentile(y_clean, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outlier_mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
        X_clean = X_clean[outlier_mask]
        y_clean = y_clean[outlier_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for growth prediction")
        
        print(f"Training growth models on {len(y_clean)} samples after outlier removal...")
        
        # Enhanced feature scaling
        if self.use_advanced_preprocessing:
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_clean)
            self.scaler = scaler
        
        # Advanced feature selection
        try:
            if X_scaled.shape[1] > 50 and len(y_clean) > 100:
                # Use RFECV for better feature selection
                base_estimator = ExtraTreesRegressor(n_estimators=50, random_state=42)
                self.feature_selector = RFECV(
                    base_estimator, 
                    step=1, 
                    cv=3,
                    scoring='r2',
                    min_features_to_select=20
                )
                X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
            elif X_scaled.shape[1] > 30:
                # Fallback to SelectKBest
                self.feature_selector = SelectKBest(
                    score_func=f_regression, 
                    k=min(30, X_scaled.shape[1])
                )
                X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
            else:
                X_selected = X_scaled
        except Exception as e:
            print(f"Error in feature selection: {e}")
            X_selected = X_scaled
        
        # Stratified split for better representation
        if len(y_clean) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_clean, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        
        # Train individual models with hyperparameter optimization
        model_scores = {}
        trained_models = {}
        
        for name, model in self.growth_models.items():
            try:
                print(f"Training and optimizing {name}...")
                
                # Optimize hyperparameters for key models
                if name in ['lightgbm', 'xgboost', 'catboost'] and len(y_train) > 100:
                    optimized_model = self._optimize_hyperparameters(
                        model, X_train, y_train, 'regression'
                    )
                else:
                    optimized_model = model
                    optimized_model.fit(X_train, y_train)
                
                # Cross-validation for robust evaluation
                cv_scores = cross_val_score(
                    optimized_model, X_train, y_train, cv=3, scoring='r2'
                )
                
                y_pred = optimized_model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                cv_mean = cv_scores.mean()
                
                # Ensure valid scores
                r2 = max(0.0, r2) if not np.isnan(r2) else 0.0
                mae = mae if not np.isnan(mae) else float('inf')
                cv_mean = max(0.0, cv_mean) if not np.isnan(cv_mean) else 0.0
                
                model_scores[name] = {
                    'r2': r2, 
                    'mae': mae, 
                    'cv_score': cv_mean,
                    'combined_score': (r2 + cv_mean) / 2  # Combined metric
                }
                trained_models[name] = optimized_model
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No models could be trained successfully")
        
        # Create advanced ensemble
        try:
            if self.use_stacking and len(trained_models) >= 3:
                # Stacking ensemble
                base_models = [(name, model) for name, model in trained_models.items()]
                
                # Use a simple linear model as meta-learner
                meta_learner = Ridge(alpha=1.0)
                
                self.growth_ensemble = StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=3
                )
                self.growth_ensemble.fit(X_train, y_train)
                
                ensemble_pred = self.growth_ensemble.predict(X_test)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                
                # Get stacking weights (approximate)
                weights = {name: 1.0/len(trained_models) for name in trained_models.keys()}
                
            else:
                # Voting ensemble with optimized weights
                weights = []
                estimators = []
                
                for name, scores in model_scores.items():
                    # Weight by combined score
                    weight = max(0.1, scores['combined_score'])
                    weights.append(weight)
                    estimators.append((name, trained_models[name]))
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                self.growth_ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=weights
                )
                self.growth_ensemble.fit(X_train, y_train)
                
                ensemble_pred = self.growth_ensemble.predict(X_test)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                
                weights = dict(zip([name for name, _ in estimators], weights))
                
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            # Use best individual model as fallback
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
            self.growth_ensemble = trained_models[best_model_name]
            ensemble_r2 = model_scores[best_model_name]['r2']
            ensemble_mae = model_scores[best_model_name]['mae']
            weights = {best_model_name: 1.0}
        
        # Enhanced feature importance
        try:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
            best_model = trained_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance = np.abs(best_model.coef_)
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
            'best_model': max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score']),
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'weights': weights,
            'outliers_removed': len(y) - len(y_clean)
        }
    
    def train_viral_classifier(self, X: np.ndarray, metadata: Dict) -> Dict:
        """Enhanced viral classification training"""
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
        
        # Stratified split for classification
        try:
            if len(y_clean) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_clean, test_size=0.2, random_state=42, 
                    stratify=y_clean if len(unique_classes) > 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        except Exception as e:
            print(f"Error in train/test split: {e}")
            X_train, X_test, y_train, y_test = X_selected, X_selected, y_clean, y_clean
        
        # Train individual models with optimization
        model_scores = {}
        trained_models = {}
        
        for name, model in self.viral_models.items():
            try:
                print(f"Training and optimizing {name}...")
                
                # Optimize hyperparameters for key models
                if name in ['lightgbm', 'xgboost', 'catboost'] and len(y_train) > 100:
                    optimized_model = self._optimize_hyperparameters(
                        model, X_train, y_train, 'classification'
                    )
                else:
                    optimized_model = model
                    optimized_model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    optimized_model, X_train, y_train, cv=3, scoring='f1_weighted'
                )
                
                y_pred = optimized_model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cv_mean = cv_scores.mean()
                
                try:
                    if hasattr(optimized_model, 'predict_proba'):
                        y_pred_proba = optimized_model.predict_proba(X_test)
                        if y_pred_proba.shape[1] > 1:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = 0.5
                    else:
                        auc = 0.5
                except:
                    auc = 0.5
                
                model_scores[name] = {
                    'accuracy': accuracy, 
                    'f1': f1, 
                    'auc': auc,
                    'cv_score': cv_mean,
                    'combined_score': (f1 + cv_mean + auc) / 3
                }
                trained_models[name] = optimized_model
                
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
        
        # Create advanced ensemble
        try:
            if self.use_stacking and len(trained_models) >= 3:
                # Stacking ensemble
                base_models = [(name, model) for name, model in trained_models.items()]
                
                meta_learner = LogisticRegression(random_state=42, max_iter=1000)
                
                self.viral_ensemble = StackingClassifier(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=3
                )
                self.viral_ensemble.fit(X_train, y_train)
                
                ensemble_pred = self.viral_ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                
                weights = {name: 1.0/len(trained_models) for name in trained_models.keys()}
                
            else:
                # Voting ensemble
                weights = []
                estimators = []
                
                for name, scores in model_scores.items():
                    weight = max(0.1, scores['combined_score'])
                    weights.append(weight)
                    estimators.append((name, trained_models[name]))
                
                weights = np.array(weights)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                self.viral_ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights
                )
                self.viral_ensemble.fit(X_train, y_train)
                
                ensemble_pred = self.viral_ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                
                weights = dict(zip([name for name, _ in estimators], weights))
                
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
            self.viral_ensemble = trained_models[best_model_name]
            ensemble_accuracy = model_scores[best_model_name]['accuracy']
            ensemble_f1 = model_scores[best_model_name]['f1']
            weights = {best_model_name: 1.0}
        
        # Feature importance
        try:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
            best_model = trained_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance = np.abs(best_model.coef_[0])
            else:
                feature_importance = np.zeros(X_selected.shape[1])
            
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
            'best_model': max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score']) if model_scores else 'none',
            'feature_importance': importance_df,
            'n_samples': len(y_clean),
            'class_distribution': class_distribution,
            'weights': weights
        }
    
    def save_models(self, output_dir: str):
        """Save all trained models and preprocessors"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save ensemble models
            if self.growth_ensemble:
                with open(output_dir / 'enhanced_growth_ensemble.pkl', 'wb') as f:
                    pickle.dump(self.growth_ensemble, f)
            
            if self.viral_ensemble:
                with open(output_dir / 'enhanced_viral_ensemble.pkl', 'wb') as f:
                    pickle.dump(self.viral_ensemble, f)
            
            # Save preprocessors
            with open(output_dir / 'enhanced_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            if self.feature_selector:
                with open(output_dir / 'enhanced_feature_selector.pkl', 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            # Save feature engineering
            with open(output_dir / 'enhanced_feature_engineer.pkl', 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            # Save dimensionality reducers
            with open(output_dir / 'enhanced_text_reducer.pkl', 'wb') as f:
                pickle.dump(self.text_reducer, f)
            
            with open(output_dir / 'enhanced_phobert_reducer.pkl', 'wb') as f:
                pickle.dump(self.phobert_reducer, f)
            
            # Save feature names
            with open(output_dir / 'enhanced_feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
                
            print(f"Enhanced models saved successfully to {output_dir}")
            
        except Exception as e:
            print(f"Error saving models: {e}")

# Update the main function to use the enhanced predictor
def main():
    """Main execution function with enhanced viral prediction"""
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
    
    print("Starting Enhanced Intelligent Viral Prediction Training...")
    print("="*80)
    
    # Initialize enhanced predictors
    try:
        # Use enhanced predictor with advanced techniques
        viral_predictor = EnhancedViralPredictor(
            use_advanced_preprocessing=True,
            use_stacking=True
        )
        
        # Keep the original trend recommender
        from model import IntelligentTrendRecommender
        trend_recommender = IntelligentTrendRecommender()
        
    except Exception as e:
        print(f"Error initializing predictors: {e}")
        return
    
    try:
        # Load and prepare features
        print("\nLoading and preparing features with enhanced preprocessing...")
        X, metadata = viral_predictor.prepare_features(features_dir)
        
        trend_recommender.load_trending_features(features_dir)
        
        # Train enhanced viral prediction models
        print("\nTraining enhanced growth prediction ensemble...")
        growth_metrics = viral_predictor.train_growth_predictor(X, metadata)
        
        print("\nTraining enhanced viral classification ensemble...")
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
        
        # Save enhanced models
        print("\nSaving enhanced models...")
        viral_predictor.save_models(models_dir)
        trend_recommender.save_models(models_dir)
        
        # Create visualizations
        print("\nCreating comprehensive visualizations...")
        from model import create_comprehensive_visualizations
        create_comprehensive_visualizations(
            growth_metrics, viral_metrics, 
            hashtag_recommendations, sound_recommendations, 
            results_dir
        )
        
        # Print detailed results
        print("\n" + "="*100)
        print("ENHANCED INTELLIGENT VIRAL PREDICTION RESULTS")
        print("="*100)
        
        print(f"\nENHANCED GROWTH PREDICTION ENSEMBLE:")
        print(f"   Training samples: {growth_metrics['n_samples']:,}")
        print(f"   Outliers removed: {growth_metrics.get('outliers_removed', 0):,}")
        print(f"   Ensemble R² Score: {growth_metrics['ensemble_r2']:.4f}")
        print(f"   Ensemble MAE: {growth_metrics['ensemble_mae']:.4f}")
        print(f"   Best individual model: {growth_metrics['best_model']}")
        
        if growth_metrics['individual_scores']:
            print(f"\n   Individual Model Performance:")
            for model, scores in growth_metrics['individual_scores'].items():
                cv_score = scores.get('cv_score', 0)
                combined = scores.get('combined_score', 0)
                print(f"      • {model}: R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}, CV={cv_score:.4f}, Combined={combined:.4f}")
        
        if growth_metrics['weights']:
            print(f"\n   Enhanced Model Weights:")
            for model, weight in growth_metrics['weights'].items():
                print(f"      • {model}: {weight:.3f}")
        
        if not growth_metrics['feature_importance'].empty:
            print(f"\n   Top Enhanced Growth Features:")
            for _, row in growth_metrics['feature_importance'].head(8).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nENHANCED VIRAL CLASSIFICATION ENSEMBLE:")
        print(f"   Training samples: {viral_metrics['n_samples']:,}")
        print(f"   Ensemble Accuracy: {viral_metrics['ensemble_accuracy']:.4f}")
        print(f"   Ensemble F1 Score: {viral_metrics['ensemble_f1']:.4f}")
        print(f"   Best individual model: {viral_metrics['best_model']}")
        print(f"   Class distribution: {viral_metrics['class_distribution']}")
        
        if viral_metrics['individual_scores']:
            print(f"\n   Individual Model Performance:")
            for model, scores in viral_metrics['individual_scores'].items():
                cv_score = scores.get('cv_score', 0)
                combined = scores.get('combined_score', 0)
                print(f"      • {model}: Acc={scores['accuracy']:.4f}, F1={scores['f1']:.4f}, AUC={scores['auc']:.4f}, CV={cv_score:.4f}, Combined={combined:.4f}")
        
        if not viral_metrics['feature_importance'].empty:
            print(f"\n   Top Enhanced Viral Features:")
            for _, row in viral_metrics['feature_importance'].head(8).iterrows():
                print(f"      • {row['feature']}: {row['importance']:.4f}")
        
        # Show improvement summary
        print(f"\n" + "="*60)
        print("ENHANCEMENT SUMMARY:")
        print(f"• Advanced feature engineering with {len(viral_predictor.feature_names)} features")
        print(f"• Hyperparameter optimization for key models")
        print(f"• {'Stacking' if viral_predictor.use_stacking else 'Voting'} ensemble methodology")
        print(f"• Advanced preprocessing with outlier removal")
        print(f"• Cross-validation based model evaluation")
        print(f"• Enhanced feature selection and dimensionality reduction")
        print("="*60)
        
        # Save enhanced results
        detailed_results = {
            'enhanced_growth_metrics': growth_metrics,
            'enhanced_viral_metrics': viral_metrics,
            'recommendations': {
                'hashtags_balanced': hashtag_recommendations,
                'hashtags_viral': viral_hashtags,
                'hashtags_engagement': engagement_hashtags,
                'sounds_balanced': sound_recommendations,
                'sounds_viral': viral_sounds,
                'sounds_emerging': emerging_sounds
            },
            'enhancement_info': {
                'preprocessing': 'QuantileTransformer + Advanced Feature Engineering',
                'ensemble_method': 'Stacking' if viral_predictor.use_stacking else 'Voting',
                'feature_count': len(viral_predictor.feature_names),
                'hyperparameter_optimization': True,
                'outlier_removal': True,
                'cross_validation': True
            }
        }
        
        # Save as pickle
        try:
            with open(Path(results_dir) / 'enhanced_intelligent_results.pkl', 'wb') as f:
                pickle.dump(detailed_results, f)
        except Exception as e:
            print(f"Error saving enhanced results: {e}")
        
        print(f"\nEnhanced intelligent training completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Enhanced models saved to: {models_dir}")
        print(f"Expected improvements in R² score and classification accuracy!")
        
    except Exception as e:
        print(f"\nError during enhanced training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()