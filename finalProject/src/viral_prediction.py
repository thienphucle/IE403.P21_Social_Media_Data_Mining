import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

class ViralPredictor:
    def __init__(self):
        self.growth_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.viral_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_features(self, features_dir: str) -> Tuple[np.ndarray, Dict]:
        """Load and prepare features for prediction"""
        features_dir = Path(features_dir)
        
        # Load dense features
        with np.load(features_dir / 'dense_features.npz') as data:
            tfidf_features = data['tfidf_features']
            phobert_features = data['phobert_features']
        
        # Load metadata
        with np.load(features_dir / 'metadata.npz') as data:
            metadata = {key: data[key] for key in data.files}
        
        # Combine features
        X = np.hstack([
            tfidf_features,
            phobert_features,
            np.column_stack([
                metadata['hashtag_counts'],
                metadata['durations'],
                metadata['followers'],
                metadata['engagement_rates']
            ])
        ])
        
        return X, metadata

    def train_growth_predictor(self, X: np.ndarray, y: np.ndarray):
        """Train model to predict growth rate"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.growth_model.fit(X_train, y_train)
        
        y_pred = self.growth_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'feature_importance': self.growth_model.feature_importances_
        }

    def train_viral_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train model to classify viral videos"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.viral_classifier.fit(X_train, y_train)
        
        y_pred = self.viral_classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'classification_report': report,
            'feature_importance': self.viral_classifier.feature_importances_
        }

    def save_models(self, output_dir: str):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'growth_model.pkl', 'wb') as f:
            pickle.dump(self.growth_model, f)
        
        with open(output_dir / 'viral_classifier.pkl', 'wb') as f:
            pickle.dump(self.viral_classifier, f)

class TrendRecommender:
    def __init__(self):
        self.hashtag_model = xgb.XGBRegressor()
        self.sound_model = xgb.XGBRegressor()
        self.trending_features = None
        
    def load_trending_features(self, features_dir: str):
        """Load trending features"""
        features_dir = Path(features_dir)
        self.trending_features = pd.read_pickle(features_dir / 'trending_features.pkl')
    
    def prepare_hashtag_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for hashtag prediction"""
        return np.column_stack([
            trends_df['total_views'].values,
            trends_df['avg_engagement'].values,
            trends_df['usage_count'].values
        ])
    
    def prepare_sound_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for sound prediction"""
        return np.column_stack([
            trends_df['vid_nview'].values,
            trends_df['vid_nlike'].values,
            trends_df['vid_nshare'].values
        ])
    
    def train_recommenders(self):
        """Train hashtag and sound recommenders"""
        # Train hashtag recommender
        hashtag_features = self.prepare_hashtag_features(self.trending_features['hashtag_trends'])
        hashtag_target = self.trending_features['hashtag_trends']['total_views'] * \
                        self.trending_features['hashtag_trends']['avg_engagement']
        self.hashtag_model.fit(hashtag_features, hashtag_target)
        
        # Train sound recommender
        sound_features = self.prepare_sound_features(self.trending_features['sound_trends'])
        sound_target = self.trending_features['sound_trends']['music_nused'].values
        self.sound_model.fit(sound_features, sound_target)
    
    def recommend_hashtags(self, n_recommendations: int = 5) -> List[str]:
        """Recommend trending hashtags"""
        features = self.prepare_hashtag_features(self.trending_features['hashtag_trends'])
        predictions = self.hashtag_model.predict(features)
        top_indices = np.argsort(predictions)[-n_recommendations:]
        return self.trending_features['hashtag_trends'].iloc[top_indices]['hashtag'].tolist()
    
    def recommend_sounds(self, n_recommendations: int = 5) -> List[Dict]:
        """Recommend trending sounds"""
        features = self.prepare_sound_features(self.trending_features['sound_trends'])
        predictions = self.sound_model.predict(features)
        top_indices = np.argsort(predictions)[-n_recommendations:]
        return self.trending_features['sound_trends'].iloc[top_indices][['music_id', 'music_title']].to_dict('records')

    def save_models(self, output_dir: str):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'hashtag_model.pkl', 'wb') as f:
            pickle.dump(self.hashtag_model, f)
        
        with open(output_dir / 'sound_model.pkl', 'wb') as f:
            pickle.dump(self.sound_model, f)

def main():
    features_dir = "finalProject/data/features"
    models_dir = "finalProject/models"
    
    # Initialize predictors
    viral_predictor = ViralPredictor()
    trend_recommender = TrendRecommender()
    
    # Load and prepare features
    print("Loading features...")
    X, metadata = viral_predictor.prepare_features(features_dir)
    trend_recommender.load_trending_features(features_dir)
    
    # Train viral prediction models
    print("\nTraining viral prediction models...")
    growth_metrics = viral_predictor.train_growth_predictor(X, metadata['growth_rates'])
    viral_metrics = viral_predictor.train_viral_classifier(
        X, (metadata['engagement_rates'] > np.median(metadata['engagement_rates'])).astype(int)
    )
    
    # Train trend recommenders
    print("\nTraining trend recommenders...")
    trend_recommender.train_recommenders()
    
    # Save models
    print("\nSaving models...")
    viral_predictor.save_models(models_dir)
    trend_recommender.save_models(models_dir)
    
    # Print metrics and recommendations
    print("\nGrowth Prediction Metrics:")
    print(f"MSE: {growth_metrics['mse']:.4f}")
    print(f"R2 Score: {growth_metrics['r2']:.4f}")
    
    print("\nViral Classification Metrics:")
    print(viral_metrics['classification_report'])
    
    print("\nTop Trending Hashtags:")
    print(trend_recommender.recommend_hashtags())
    
    print("\nTop Trending Sounds:")
    print(trend_recommender.recommend_sounds())

if __name__ == "__main__":
    main()
