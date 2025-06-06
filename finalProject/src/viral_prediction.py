# viral_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
from typing import Dict, List, Tuple

class ViralPredictor:
    def __init__(self):
        self.growth_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.viral_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.hashtag_recommender = None
        self.sound_recommender = None

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        features = np.hstack([
            df['tfidf_features'].tolist(),
            df['phobert_features'].tolist(),
            df[['hashtag_count', 'vid_duration_sec', 'user_nfollower',
                'post_hour', 'engagement_rate']].values
        ])
        return features

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

    def predict_growth(self, X: np.ndarray) -> np.ndarray:
        """Predict growth rate for new videos"""
        return self.growth_model.predict(X)

    def predict_viral_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of video going viral"""
        return self.viral_classifier.predict_proba(X)

class TrendRecommender:
    def __init__(self):
        self.hashtag_model = xgb.XGBRegressor()
        self.sound_model = xgb.XGBRegressor()
        
    def prepare_trend_features(self, trends_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for trend prediction"""
        return np.column_stack([
            trends_df['total_views'].values,
            trends_df['avg_engagement'].values,
            trends_df['usage_count'].values
        ])
    
    def train_recommenders(self, df: pd.DataFrame):
        """Train hashtag and sound recommenders"""
        # Train hashtag recommender
        hashtag_features = self.prepare_trend_features(df['hashtag_trends'])
        hashtag_target = df['hashtag_trends']['total_views'] * \
                        df['hashtag_trends']['avg_engagement']
        self.hashtag_model.fit(hashtag_features, hashtag_target)
        
        # Train sound recommender
        sound_features = np.column_stack([
            df['sound_trends']['vid_nview'].values,
            df['sound_trends']['vid_nlike'].values,
            df['sound_trends']['vid_nshare'].values
        ])
        sound_target = df['sound_trends']['music_nused'].values
        self.sound_model.fit(sound_features, sound_target)
    
    def recommend_hashtags(self, n_recommendations: int = 5) -> List[str]:
        """Recommend trending hashtags"""
        predictions = self.hashtag_model.predict(self.prepare_trend_features(self.hashtag_trends))
        top_indices = np.argsort(predictions)[-n_recommendations:]
        return self.hashtag_trends.iloc[top_indices]['hashtag'].tolist()
    
    def recommend_sounds(self, n_recommendations: int = 5) -> List[Dict]:
        """Recommend trending sounds"""
        predictions = self.sound_model.predict(self.prepare_trend_features(self.sound_trends))
        top_indices = np.argsort(predictions)[-n_recommendations:]
        return self.sound_trends.iloc[top_indices][['music_id', 'music_title']].to_dict('records')

def main():
    # Load preprocessed data
    df = pd.read_csv("finalProject/data/processed_data.csv")
    
    # Initialize predictors
    viral_predictor = ViralPredictor()
    trend_recommender = TrendRecommender()
    
    # Prepare features
    X = viral_predictor.prepare_features(df)
    growth_target = df['growth_rate'].values
    viral_target = (df['engagement_rate'] > df['engagement_rate'].median()).astype(int)
    
    # Train models
    growth_metrics = viral_predictor.train_growth_predictor(X, growth_target)
    viral_metrics = viral_predictor.train_viral_classifier(X, viral_target)
    
    # Train trend recommenders
    trend_recommender.train_recommenders(df)
    
    # Save models and metrics
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
