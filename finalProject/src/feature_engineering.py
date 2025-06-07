import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def extract_phobert_features(texts, model_name="vinai/phobert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
    
    features = []
    batch_size = 32

    texts = [str(t) if not pd.isna(t) else "" for t in texts]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        features.extend(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return np.array(features)

def calculate_growth_rate(row):
    """Calculate video growth rate based on views and time"""
    hours_since_post = (pd.to_datetime(row['vid_scrapeTime']) - 
                       pd.to_datetime(row['vid_postTime'])).total_seconds() / 3600
    if hours_since_post > 0:
        return row['vid_nview'] / hours_since_post
    return 0

def calculate_engagement_rate(row):
    """Calculate engagement rate based on interactions"""
    if row['vid_nview'] > 0:
        total_engagement = (row['vid_nlike'] + row['vid_ncomment'] + 
                          row['vid_nshare'] + row['vid_nsave'])
        return (total_engagement / row['vid_nview']) * 100
    return 0

def calculate_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-series features for videos with multiple data points"""
    df_ts = df.copy()
    df_ts['vid_scrapeTime'] = pd.to_datetime(df_ts['vid_scrapeTime'])
    df_ts['vid_postTime'] = pd.to_datetime(df_ts['vid_postTime'])
    
    # Sort by video ID and scrape time
    df_ts = df_ts.sort_values(['vid_id', 'vid_scrapeTime'])
    
    # Calculate growth and engagement rates
    df_ts['growth_rate'] = df_ts.apply(calculate_growth_rate, axis=1)
    df_ts['engagement_rate'] = df_ts.apply(calculate_engagement_rate, axis=1)
    
    # Calculate time-series features for each video
    video_features = []
    
    for vid_id in df_ts['vid_id'].unique():
        video_data = df_ts[df_ts['vid_id'] == vid_id].sort_values('vid_scrapeTime')
        
        if len(video_data) >= 2:
            # Get first and latest measurements
            first_measurement = video_data.iloc[0]
            latest_measurement = video_data.iloc[-1]
            
            # Calculate growth metrics
            time_diff_hours = (latest_measurement['vid_scrapeTime'] - 
                             first_measurement['vid_scrapeTime']).total_seconds() / 3600
            
            if time_diff_hours > 0:
                view_growth = (latest_measurement['vid_nview'] - first_measurement['vid_nview']) / time_diff_hours
                like_growth = (latest_measurement['vid_nlike'] - first_measurement['vid_nlike']) / time_diff_hours
                comment_growth = (latest_measurement['vid_ncomment'] - first_measurement['vid_ncomment']) / time_diff_hours
                share_growth = (latest_measurement['vid_nshare'] - first_measurement['vid_nshare']) / time_diff_hours
                
                # Calculate growth rate change
                initial_growth_rate = first_measurement['growth_rate']
                current_growth_rate = latest_measurement['growth_rate']
                growth_rate_change = current_growth_rate - initial_growth_rate
                
                # Calculate engagement momentum
                initial_engagement = first_measurement['engagement_rate']
                current_engagement = latest_measurement['engagement_rate']
                engagement_momentum = current_engagement - initial_engagement
                
                video_features.append({
                    'vid_id': vid_id,
                    'user_name': latest_measurement['user_name'],
                    'vid_caption': latest_measurement['vid_caption'],
                    'vid_hashtags_normalized': latest_measurement['vid_hashtags_normalized'],
                    'hashtag_count': latest_measurement['hashtag_count'],
                    'vid_duration_sec': latest_measurement['vid_duration_sec'],
                    'user_nfollower': latest_measurement['user_nfollower'],
                    'music_id': latest_measurement['music_id'],
                    'music_title': latest_measurement['music_title'],
                    'initial_views': first_measurement['vid_nview'],
                    'current_views': latest_measurement['vid_nview'],
                    'initial_growth_rate': initial_growth_rate,
                    'current_growth_rate': current_growth_rate,
                    'new_growth_rate': growth_rate_change,
                    'view_growth_per_hour': view_growth,
                    'like_growth_per_hour': like_growth,
                    'comment_growth_per_hour': comment_growth,
                    'share_growth_per_hour': share_growth,
                    'initial_engagement_rate': initial_engagement,
                    'current_engagement_rate': current_engagement,
                    'engagement_momentum': engagement_momentum,
                    'time_diff_hours': time_diff_hours,
                    'viral_acceleration': view_growth / max(first_measurement['vid_nview'], 1),
                    'post_hour': latest_measurement['vid_postTime'].hour,
                    'post_day': latest_measurement['vid_postTime'].day_name()
                })
    
    return pd.DataFrame(video_features)

def find_optimal_viral_threshold(df: pd.DataFrame) -> float:
    """Find optimal threshold for continuing viral classification"""
    growth_rates = df['new_growth_rate'].dropna()
    
    # Try different percentiles as thresholds
    thresholds = np.percentile(growth_rates, [50, 60, 70, 75, 80, 85, 90, 95])
    
    # Calculate metrics for each threshold
    threshold_metrics = []
    
    for threshold in thresholds:
        continuing_viral = (growth_rates > threshold).sum()
        total_videos = len(growth_rates)
        viral_ratio = continuing_viral / total_videos
        
        # Calculate variance in growth rates above threshold
        above_threshold = growth_rates[growth_rates > threshold]
        variance = above_threshold.var() if len(above_threshold) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'viral_count': continuing_viral,
            'viral_ratio': viral_ratio,
            'variance': variance,
            'score': viral_ratio * (1 - variance / max(growth_rates.var(), 1))  # Balance ratio and consistency
        })
    
    # Find threshold with best score
    best_threshold = max(threshold_metrics, key=lambda x: x['score'])['threshold']
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(growth_rates, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {best_threshold:.2f}')
    plt.xlabel('New Growth Rate')
    plt.ylabel('Frequency')
    plt.title('Distribution of New Growth Rates')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    thresholds_plot = [m['threshold'] for m in threshold_metrics]
    scores = [m['score'] for m in threshold_metrics]
    plt.plot(thresholds_plot, scores, 'bo-')
    plt.axvline(best_threshold, color='red', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Optimization Score')
    
    plt.subplot(2, 2, 3)
    viral_ratios = [m['viral_ratio'] for m in threshold_metrics]
    plt.plot(thresholds_plot, viral_ratios, 'go-')
    plt.xlabel('Threshold')
    plt.ylabel('Viral Ratio')
    plt.title('Viral Video Ratio by Threshold')
    
    plt.subplot(2, 2, 4)
    variances = [m['variance'] for m in threshold_metrics]
    plt.plot(thresholds_plot, variances, 'mo-')
    plt.xlabel('Threshold')
    plt.ylabel('Variance')
    plt.title('Growth Rate Variance by Threshold')
    
    plt.tight_layout()
    plt.savefig('finalProject/results/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimal threshold for continuing viral classification: {best_threshold:.4f}")
    print(f"Videos classified as continuing viral: {(growth_rates > best_threshold).sum()}/{len(growth_rates)} ({(growth_rates > best_threshold).mean()*100:.1f}%)")
    
    return best_threshold

def extract_trending_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extract trending hashtags and sounds"""
    # Hashtag trends
    hashtag_stats = []
    for idx, row in df.iterrows():
        hashtags = row['vid_hashtags_normalized']
        if isinstance(hashtags, list):
            for tag in hashtags:
                hashtag_stats.append({
                    'hashtag': tag,
                    'views': row['current_views'],
                    'engagement': row['current_engagement_rate'],
                    'growth_momentum': row['new_growth_rate']
                })
    
    hashtag_df = pd.DataFrame(hashtag_stats)
    if not hashtag_df.empty:
        hashtag_trends = hashtag_df.groupby('hashtag').agg({
            'views': 'sum',
            'engagement': 'mean',
            'growth_momentum': 'mean'
        }).reset_index()
        hashtag_trends.columns = ['hashtag', 'total_views', 'avg_engagement', 'avg_growth_momentum']
        hashtag_trends['usage_count'] = hashtag_df.groupby('hashtag').size().values
    else:
        hashtag_trends = pd.DataFrame(columns=['hashtag', 'total_views', 'avg_engagement', 'avg_growth_momentum', 'usage_count'])
    
    # Sound trends
    sound_stats = df.groupby(['music_id', 'music_title']).agg({
        'current_views': 'sum',
        'current_engagement_rate': 'mean',
        'new_growth_rate': 'mean',
        'viral_acceleration': 'mean'
    }).reset_index()
    
    return {
        'hashtag_trends': hashtag_trends,
        'sound_trends': sound_stats
    }

def extract_features(df):
    print("Processing time-series data...")
    df_ts = calculate_time_series_features(df)
    
    print("Finding optimal viral threshold...")
    optimal_threshold = find_optimal_viral_threshold(df_ts)
    
    # Classify continuing viral videos
    df_ts['continuing_viral'] = (df_ts['new_growth_rate'] > optimal_threshold).astype(int)
    
    print("Extracting trending features...")
    trending_features = extract_trending_features(df_ts)
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Combine caption and hashtags for text features
    text_data = df_ts['vid_caption'].fillna('') + ' ' + df_ts['vid_hashtags_normalized'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else ''
    )
    text_features = tfidf.fit_transform(text_data)
    
    # Extract PhoBERT features
    print("Extracting PhoBERT embeddings...")
    phobert_features = extract_phobert_features(df_ts['vid_caption'].fillna(''))
    
    # Create comprehensive metadata dictionary
    metadata = {
        'video_ids': df_ts['vid_id'].values,
        'user_names': df_ts['user_name'].values,
        'hashtag_counts': df_ts['hashtag_count'].values,
        'durations': df_ts['vid_duration_sec'].values,
        'followers': df_ts['user_nfollower'].values,
        'initial_views': df_ts['initial_views'].values,
        'current_views': df_ts['current_views'].values,
        'initial_growth_rate': df_ts['initial_growth_rate'].values,
        'current_growth_rate': df_ts['current_growth_rate'].values,
        'new_growth_rate': df_ts['new_growth_rate'].values,
        'view_growth_per_hour': df_ts['view_growth_per_hour'].values,
        'like_growth_per_hour': df_ts['like_growth_per_hour'].values,
        'comment_growth_per_hour': df_ts['comment_growth_per_hour'].values,
        'share_growth_per_hour': df_ts['share_growth_per_hour'].values,
        'initial_engagement_rate': df_ts['initial_engagement_rate'].values,
        'current_engagement_rate': df_ts['current_engagement_rate'].values,
        'engagement_momentum': df_ts['engagement_momentum'].values,
        'viral_acceleration': df_ts['viral_acceleration'].values,
        'continuing_viral': df_ts['continuing_viral'].values,
        'time_diff_hours': df_ts['time_diff_hours'].values,
        'post_hour': df_ts['post_hour'].values,
        'optimal_threshold': optimal_threshold
    }
    
    return {
        'tfidf_features': text_features.toarray(),
        'phobert_features': phobert_features,
        'trending_features': trending_features,
        'tfidf_vectorizer': tfidf,
        'metadata': metadata,
        'processed_dataframe': df_ts
    }

def save_features(features, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dense features using numpy
    np.savez_compressed(
        output_dir / 'dense_features.npz',
        tfidf_features=features['tfidf_features'],
        phobert_features=features['phobert_features']
    )
    
    # Save metadata using numpy
    np.savez_compressed(
        output_dir / 'metadata.npz',
        **features['metadata']
    )
    
    # Save trending features
    pd.to_pickle(features['trending_features'], output_dir / 'trending_features.pkl')
    
    # Save the TF-IDF vectorizer
    with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(features['tfidf_vectorizer'], f)
    
    # Save processed dataframe for analysis
    features['processed_dataframe'].to_csv(output_dir / 'processed_timeseries_data.csv', index=False)
    
    print(f"Features saved to {output_dir}")
    print(f"Optimal viral threshold: {features['metadata']['optimal_threshold']:.4f}")
    print(f"Videos continuing to go viral: {features['metadata']['continuing_viral'].sum()}/{len(features['metadata']['continuing_viral'])}")

def main():
    infile = "final_filtered_preprocessed_data_nomissing.csv"
    output_dir = "finalProject/data/features"
    results_dir = "finalProject/FE_results"

    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading preprocessed data from {infile}...")
    df = pd.read_csv(infile)
    
    # Parse hashtags if they're stored as strings
    if 'vid_hashtags_normalized' in df.columns:
        df['vid_hashtags_normalized'] = df['vid_hashtags_normalized'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
    print("Extracting features...")
    features = extract_features(df)
    
    print(f"Saving features to {output_dir}...")
    save_features(features, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()