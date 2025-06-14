import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import xgboost as xgb
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ViralPredictor:
    def __init__(self):
        self.growth_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.viral_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        """Load and prepare features for prediction"""
        features_dir = Path(features_dir)
        
        # Load dense features
        with np.load(features_dir / 'dense_features.npz') as data:
            tfidf_features = data['tfidf_features']
            phobert_features = data['phobert_features']
        
        # Load metadata
        with np.load(features_dir / 'metadata.npz', allow_pickle=True) as data:
            metadata = {key: data[key] for key in data.files}
        
        # Create feature matrix with proper handling of different feature types
        numerical_features = []
        feature_names = []
        
        # Add hashtag counts
        if 'hashtag_counts' in metadata:
            numerical_features.append(metadata['hashtag_counts'].reshape(-1, 1))
            feature_names.append('hashtag_count')
        
        # Add durations
        if 'durations' in metadata:
            numerical_features.append(metadata['durations'].reshape(-1, 1))
            feature_names.append('duration')
        
        # Add followers (log transform to handle large values)
        if 'followers' in metadata:
            followers = np.log1p(metadata['followers']).reshape(-1, 1)
            numerical_features.append(followers)
            feature_names.append('log_followers')
        
        # Add initial engagement metrics
        if 'initial_engagement_rate' in metadata:
            numerical_features.append(metadata['initial_engagement_rate'].reshape(-1, 1))
            feature_names.append('initial_engagement_rate')
        
        # Add current engagement metrics
        if 'current_engagement_rate' in metadata:
            numerical_features.append(metadata['current_engagement_rate'].reshape(-1, 1))
            feature_names.append('current_engagement_rate')
        
        # Add growth metrics
        if 'view_growth_per_hour' in metadata:
            numerical_features.append(metadata['view_growth_per_hour'].reshape(-1, 1))
            feature_names.append('view_growth_per_hour')
        
        if 'like_growth_per_hour' in metadata:
            numerical_features.append(metadata['like_growth_per_hour'].reshape(-1, 1))
            feature_names.append('like_growth_per_hour')
        
        if 'comment_growth_per_hour' in metadata:
            numerical_features.append(metadata['comment_growth_per_hour'].reshape(-1, 1))
            feature_names.append('comment_growth_per_hour')
        
        if 'share_growth_per_hour' in metadata:
            numerical_features.append(metadata['share_growth_per_hour'].reshape(-1, 1))
            feature_names.append('share_growth_per_hour')
        
        # Add viral acceleration
        if 'viral_acceleration' in metadata:
            numerical_features.append(metadata['viral_acceleration'].reshape(-1, 1))
            feature_names.append('viral_acceleration')
        
        # Add time features
        if 'post_hour' in metadata:
            numerical_features.append(metadata['post_hour'].reshape(-1, 1))
            feature_names.append('post_hour')
        
        if 'time_diff_hours' in metadata:
            numerical_features.append(metadata['time_diff_hours'].reshape(-1, 1))
            feature_names.append('time_diff_hours')
        
        # Combine all numerical features
        if numerical_features:
            numerical_matrix = np.hstack(numerical_features)
        else:
            numerical_matrix = np.empty((len(metadata['video_ids']), 0))
        
        # Reduce dimensionality of text features using PCA if needed
        from sklearn.decomposition import PCA
        
        # Reduce TF-IDF features
        if tfidf_features.shape[1] > 100:
            pca_tfidf = PCA(n_components=100, random_state=42)
            tfidf_reduced = pca_tfidf.fit_transform(tfidf_features)
        else:
            tfidf_reduced = tfidf_features
        
        # Reduce PhoBERT features
        if phobert_features.shape[1] > 50:
            pca_phobert = PCA(n_components=50, random_state=42)
            phobert_reduced = pca_phobert.fit_transform(phobert_features)
        else:
            phobert_reduced = phobert_features
        
        # Combine all features
        X = np.hstack([
            numerical_matrix,
            tfidf_reduced,
            phobert_reduced
        ])
        
        # Store feature names for interpretation
        tfidf_names = [f'tfidf_{i}' for i in range(tfidf_reduced.shape[1])]
        phobert_names = [f'phobert_{i}' for i in range(phobert_reduced.shape[1])]
        self.feature_names = feature_names + tfidf_names + phobert_names
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, metadata

    def train_growth_predictor(self, X: np.ndarray, metadata: Dict):
        """Train model to predict new growth rate"""
        # Use new_growth_rate as target
        y = metadata['new_growth_rate']
        
        # Remove any samples with invalid targets
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for growth prediction")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        self.growth_model.fit(X_train, y_train)
        
        y_pred = self.growth_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance analysis
        feature_importance = self.growth_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'top_features': importance_df.head(10),
            'n_samples': len(y_clean)
        }

    def train_viral_classifier(self, X: np.ndarray, metadata: Dict):
        """Train model to classify continuing viral videos"""
        # Use continuing_viral as target
        y = metadata['continuing_viral']
        
        # Remove any samples with invalid targets
        valid_mask = ~(np.isnan(y) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)
        
        if len(y_clean) == 0:
            raise ValueError("No valid samples for viral classification")
        
        # Check class distribution
        unique_classes, counts = np.unique(y_clean, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, counts))}")
        
        if len(unique_classes) < 2:
            print("Warning: Only one class present in data. Skipping classification.")
            return {
                'classification_report': {},
                'feature_importance': np.zeros(X_clean.shape[1]),
                'accuracy': 0.0,
                'n_samples': len(y_clean),
                'class_distribution': dict(zip(unique_classes, counts))
            }
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        self.viral_classifier.fit(X_train, y_train)
        
        y_pred = self.viral_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Feature importance analysis
        feature_importance = self.viral_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'classification_report': report,
            'feature_importance': feature_importance,
            'top_features': importance_df.head(10),
            'accuracy': accuracy,
            'n_samples': len(y_clean),
            'class_distribution': dict(zip(unique_classes, counts))
        }

    def save_models(self, output_dir: str):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'growth_model.pkl', 'wb') as f:
            pickle.dump(self.growth_model, f)
        
        with open(output_dir / 'viral_classifier.pkl', 'wb') as f:
            pickle.dump(self.viral_classifier, f)
        
        # Save feature names for later use
        with open(output_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)

class TrendRecommender:
    def __init__(self):
        self.hashtag_model = xgb.XGBRegressor(random_state=42)
        self.sound_model = xgb.XGBRegressor(random_state=42)
        self.trending_features = None
        
    def load_trending_features(self, features_dir: str):
        """Load trending features"""
        features_dir = Path(features_dir)
        self.trending_features = pd.read_pickle(features_dir / 'trending_features.pkl')
    
    def prepare_hashtag_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for hashtag prediction"""
        if trends_df.empty:
            return np.array([]).reshape(0, 3)
        
        features = []
        for col in ['total_views', 'avg_engagement', 'usage_count']:
            if col in trends_df.columns:
                values = trends_df[col].fillna(0).values
                features.append(values)
            else:
                features.append(np.zeros(len(trends_df)))
        
        return np.column_stack(features)
    
    def prepare_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for sound prediction"""
        if trends_df.empty:
            return np.array([]).reshape(0, 3)
        
        features = []
        # Map the actual column names from sound_trends
        column_mapping = {
            'current_views': 'current_views',
            'current_engagement_rate': 'current_engagement_rate', 
            'new_growth_rate': 'new_growth_rate'
        }
        
        for expected_col, actual_col in column_mapping.items():
            if actual_col in trends_df.columns:
                values = trends_df[actual_col].fillna(0).values
                features.append(values)
            else:
                features.append(np.zeros(len(trends_df)))
        
        return np.column_stack(features)
    
    def train_recommenders(self):
        """Train hashtag and sound recommenders"""
        if not self.trending_features:
            print("No trending features loaded. Skipping recommender training.")
            return
        
        # Train hashtag recommender
        hashtag_df = self.trending_features['hashtag_trends']
        if not hashtag_df.empty and len(hashtag_df) > 1:
            hashtag_features = self.prepare_hashtag_features(hashtag_df)
            if hashtag_features.size > 0:
                # Create target: combination of views and engagement
                hashtag_target = (hashtag_df['total_views'].fillna(0) * 
                                hashtag_df['avg_engagement'].fillna(0)).values
                
                if len(hashtag_target) > 0 and np.any(hashtag_target > 0):
                    self.hashtag_model.fit(hashtag_features, hashtag_target)
                    print(f"Trained hashtag recommender on {len(hashtag_target)} samples")
                else:
                    print("No valid hashtag targets for training")
            else:
                print("No hashtag features available for training")
        else:
            print("Insufficient hashtag data for training")
        
        # Train sound recommender
        sound_df = self.trending_features['sound_trends']
        if not sound_df.empty and len(sound_df) > 1:
            sound_features = self.prepare_sound_features(sound_df)
            if sound_features.size > 0:
                # Use current_views as target
                sound_target = sound_df['current_views'].fillna(0).values
                
                if len(sound_target) > 0 and np.any(sound_target > 0):
                    self.sound_model.fit(sound_features, sound_target)
                    print(f"Trained sound recommender on {len(sound_target)} samples")
                else:
                    print("No valid sound targets for training")
            else:
                print("No sound features available for training")
        else:
            print("Insufficient sound data for training")
    
    def recommend_hashtags(self, n_recommendations: int = 5) -> List[str]:
        """Recommend trending hashtags"""
        if not self.trending_features or self.trending_features['hashtag_trends'].empty:
            return []
        
        hashtag_df = self.trending_features['hashtag_trends']
        features = self.prepare_hashtag_features(hashtag_df)
        
        if features.size == 0:
            return []
        
        try:
            predictions = self.hashtag_model.predict(features)
            top_indices = np.argsort(predictions)[-n_recommendations:][::-1]
            return hashtag_df.iloc[top_indices]['hashtag'].tolist()
        except Exception as e:
            print(f"Error in hashtag recommendation: {e}")
            # Fallback to top hashtags by usage
            return hashtag_df.nlargest(n_recommendations, 'usage_count')['hashtag'].tolist()
    
    def recommend_sounds(self, n_recommendations: int = 5) -> List[Dict]:
        """Recommend trending sounds"""
        if not self.trending_features or self.trending_features['sound_trends'].empty:
            return []
        
        sound_df = self.trending_features['sound_trends']
        features = self.prepare_sound_features(sound_df)
        
        if features.size == 0:
            return []
        
        try:
            predictions = self.sound_model.predict(features)
            top_indices = np.argsort(predictions)[-n_recommendations:][::-1]
            result = sound_df.iloc[top_indices][['music_id', 'music_title']].to_dict('records')
            return result
        except Exception as e:
            print(f"Error in sound recommendation: {e}")
            # Fallback to top sounds by views
            return sound_df.nlargest(n_recommendations, 'current_views')[['music_id', 'music_title']].to_dict('records')

    def save_models(self, output_dir: str):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'hashtag_model.pkl', 'wb') as f:
            pickle.dump(self.hashtag_model, f)
        
        with open(output_dir / 'sound_model.pkl', 'wb') as f:
            pickle.dump(self.sound_model, f)

def create_visualizations(growth_metrics: Dict, viral_metrics: Dict, output_dir: str):
    """Create visualization plots for model performance"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot feature importance for growth model
    if 'top_features' in growth_metrics:
        plt.figure(figsize=(12, 8))
        top_features = growth_metrics['top_features']
        
        plt.subplot(2, 2, 1)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top Features for Growth Prediction')
        plt.gca().invert_yaxis()
    
    # Plot feature importance for viral classifier
    if 'top_features' in viral_metrics:
        top_features_viral = viral_metrics['top_features']
        
        plt.subplot(2, 2, 2)
        plt.barh(range(len(top_features_viral)), top_features_viral['importance'])
        plt.yticks(range(len(top_features_viral)), top_features_viral['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top Features for Viral Classification')
        plt.gca().invert_yaxis()
    
    # Plot model performance metrics
    plt.subplot(2, 2, 3)
    metrics = ['R² Score', 'MSE (scaled)']
    values = [growth_metrics['r2'], growth_metrics['mse'] / 1000]  # Scale MSE for visualization
    plt.bar(metrics, values)
    plt.title('Growth Model Performance')
    plt.ylabel('Score')
    
    plt.subplot(2, 2, 4)
    if 'accuracy' in viral_metrics:
        plt.bar(['Accuracy'], [viral_metrics['accuracy']])
        plt.title('Viral Classification Accuracy')
        plt.ylabel('Score')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    features_dir = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\data\features"
    models_dir = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\models"
    results_dir = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\model_results"
    
    # Create directories
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize predictors
    viral_predictor = ViralPredictor()
    trend_recommender = TrendRecommender()
    
    try:
        # Load and prepare features
        print("Loading features...")
        X, metadata = viral_predictor.prepare_features(features_dir)
        print(f"Loaded features with shape: {X.shape}")
        
        trend_recommender.load_trending_features(features_dir)
        
        # Train viral prediction models
        print("\nTraining growth prediction model...")
        growth_metrics = viral_predictor.train_growth_predictor(X, metadata)
        
        print("\nTraining viral classification model...")
        viral_metrics = viral_predictor.train_viral_classifier(X, metadata)
        
        # Train trend recommenders
        print("\nTraining trend recommenders...")
        trend_recommender.train_recommenders()
        
        # Save models
        print("\nSaving models...")
        viral_predictor.save_models(models_dir)
        trend_recommender.save_models(models_dir)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(growth_metrics, viral_metrics, results_dir)
        
        # Print results
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS")
        print("="*50)
        
        print(f"\nGrowth Prediction Model:")
        print(f"  Samples used: {growth_metrics['n_samples']}")
        print(f"  R² Score: {growth_metrics['r2']:.4f}")
        print(f"  MSE: {growth_metrics['mse']:.4f}")
        
        print(f"\nTop Growth Prediction Features:")
        for _, row in growth_metrics['top_features'].head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nViral Classification Model:")
        print(f"  Samples used: {viral_metrics['n_samples']}")
        print(f"  Accuracy: {viral_metrics['accuracy']:.4f}")
        print(f"  Class distribution: {viral_metrics['class_distribution']}")
        
        if viral_metrics['top_features'] is not None and not viral_metrics['top_features'].empty:
            print(f"\nTop Viral Classification Features:")
            for _, row in viral_metrics['top_features'].head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nTrend Recommendations:")
        hashtags = trend_recommender.recommend_hashtags()
        sounds = trend_recommender.recommend_sounds()
        
        print(f"  Top Trending Hashtags: {hashtags}")
        print(f"  Top Trending Sounds: {[s.get('music_title', 'Unknown') for s in sounds[:3]]}")
        
        # Save results summary
        results_summary = {
            'growth_metrics': growth_metrics,
            'viral_metrics': viral_metrics,
            'recommended_hashtags': hashtags,
            'recommended_sounds': sounds
        }
        
        with open(Path(results_dir) / 'training_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)
        
        print(f"\nResults saved to {results_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()