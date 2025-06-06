import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

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

def extract_trending_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extract trending hashtags and sounds"""
    # Hashtag trends
    hashtag_stats = []
    for idx, row in df.iterrows():
        hashtags = row['vid_hashtags_normalized']
        for tag in hashtags:
            hashtag_stats.append({
                'hashtag': tag,
                'views': row['vid_nview'],
                'engagement': calculate_engagement_rate(row),
                'timestamp': pd.to_datetime(row['vid_postTime'])
            })
    
    hashtag_df = pd.DataFrame(hashtag_stats)
    hashtag_trends = hashtag_df.groupby('hashtag').agg({
        'views': 'sum',
        'engagement': 'mean',
        'timestamp': 'count'
    }).reset_index()
    hashtag_trends.columns = ['hashtag', 'total_views', 'avg_engagement', 'usage_count']
    
    # Sound trends
    sound_stats = df.groupby(['music_id', 'music_title']).agg({
        'vid_nview': 'sum',
        'vid_nlike': 'sum',
        'vid_ncomment': 'sum',
        'vid_nshare': 'sum',
        'music_nused': 'first'
    }).reset_index()
    
    return {
        'hashtag_trends': hashtag_trends,
        'sound_trends': sound_stats
    }

def extract_features(df):
    df_features = df.copy()
    
    # Calculate growth and engagement rates
    df_features['growth_rate'] = df_features.apply(calculate_growth_rate, axis=1)
    df_features['engagement_rate'] = df_features.apply(calculate_engagement_rate, axis=1)
    
    # Extract trending features
    trending_features = extract_trending_features(df_features)
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=1000)
    text_features = tfidf.fit_transform(
        df_features['vid_caption_clean'] + ' ' + 
        df_features['vid_hashtags_normalized'].apply(lambda x: ' '.join(x))
    )
    
    # Extract PhoBERT features
    print("Extracting PhoBERT embeddings...")
    phobert_features = extract_phobert_features(df_features['vid_caption_clean'])
    
    # Create metadata dictionary
    metadata = {
        'video_ids': df_features['vid_id'].values,
        'post_times': df_features['vid_postTime'].values,
        'hashtag_counts': df_features['hashtag_count'].values,
        'durations': df_features['vid_duration_sec'].values,
        'followers': df_features['user_nfollower'].values,
        'views': df_features['vid_nview'].values,
        'likes': df_features['vid_nlike'].values,
        'comments': df_features['vid_ncomment'].values,
        'shares': df_features['vid_nshare'].values,
        'saves': df_features['vid_nsave'].values,
        'growth_rates': df_features['growth_rate'].values,
        'engagement_rates': df_features['engagement_rate'].values
    }
    
    return {
        'tfidf_features': text_features.toarray(),
        'phobert_features': phobert_features,
        'trending_features': trending_features,
        'tfidf_vectorizer': tfidf,
        'metadata': metadata
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

def main():
    infile = "finalProject/data/preprocessed_data.csv"
    output_dir = "finalProject/data/features"
    
    print(f"Loading preprocessed data from {infile}...")
    df = pd.read_csv(infile)
    
    print("Extracting features...")
    features = extract_features(df)
    
    print(f"Saving features to {output_dir}...")
    save_features(features, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
