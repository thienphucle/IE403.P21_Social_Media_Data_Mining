import pandas as pd
import numpy as np
from datetime import datetime
import emoji
import re
from dateutil import tz
from typing import Dict, List, Tuple
from pyvi import ViTokenizer

def remove_emoji(text):
    return emoji.replace_emoji(text, '')

# Load stopwords
STOPWORDS = "finalProject/data/vietnamese-stopwords-dash.txt"
with open(STOPWORDS, "r", encoding="utf-8") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    vietnamese_stopwords = set(stopwords)

def remove_stopwords(text):
    words = ViTokenizer.tokenize(text).split()
    return ' '.join([w for w in words if w.lower() not in vietnamese_stopwords])

def normalize_hashtags(hashtags):
    if pd.isna(hashtags) or hashtags == '':
        return []
    tags = hashtags.split(',')
    return [tag.strip().lower() for tag in tags]

def convert_to_vietnam_time(utc_time):
    from_zone = tz.tzutc()
    to_zone = tz.gettz('Asia/Ho_Chi_Minh')
    utc = pd.to_datetime(utc_time).replace(tzinfo=from_zone)
    return utc.astimezone(to_zone)

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

def calculate_viral_score(row):
    if row['vid_nview'] == 0 or pd.isna(row['vid_nview']):
        return 0

    # 1. Engagement (40% weight - all metrics equally weighted)
    engagement_metrics = [
        row['vid_nlike'] / row['vid_nview'],
        row['vid_ncomment'] / row['vid_nview'],
        row['vid_nshare'] / row['vid_nview'],
        row['vid_nsave'] / row['vid_nview']
    ]
    engagement_ratio = np.mean(engagement_metrics)
    engagement_score = min(40, engagement_ratio * 10_000)

    # 2. Creator Influence (20% weight)
    if row['user_nfollower'] <= 1000:  # Micro-influencers get bonus
        influence_score = min(20, (row['user_nfollower'] / 1000) * 10 + 10)
    else:
        influence_score = min(20, np.log10(row['user_nfollower']) * 5)

    # 3. Content Virality (30% weight)
    hashtag_count = len(row['vid_hashtags_normalized'])
    duration_score = 10 * (1 - min(1, row['vid_duration_sec'] / 60))  # Prefer shorter videos
    content_score = min(30, (hashtag_count * 5 + duration_score * 2))

    # 4. Velocity & Recency (10% weight)
    velocity = (row['vid_nview'] / max(1, row['vid_existtime_hrs'])) / 1000
    recency_score = 5 * (1 - min(1, row['vid_existtime_hrs'] / 72))  # Fresh = better
    velocity_score = min(10, velocity + recency_score)

    return min(100, engagement_score + influence_score + content_score + velocity_score)

def preprocess_tiktok_data(df):
    df_clean = df.copy()

    def convert_units(x):
        if isinstance(x, str):
            x = x.strip().upper().replace(' VIDEOS', '')
            try:
                if 'K' in x:
                    return float(x.replace('K', '').replace(',', '')) * 1_000
                elif 'M' in x:
                    return float(x.replace('M', '').replace(',', '')) * 1_000_000
                elif 'B' in x:
                    return float(x.replace('B', '').replace(',', '')) * 1_000_000_000
                else:
                    return float(x.replace(',', ''))
            except ValueError:
                return np.nan
        elif isinstance(x, (int, float)):
            return x
        return np.nan

    numeric_cols = ['user_nfollower', 'vid_nview', 'vid_nlike', 'vid_ncomment', 
                   'vid_nshare', 'vid_nsave', 'music_nused']

    for col in numeric_cols:
        df_clean[col] = df_clean[col].apply(convert_units)

    # Clean caption text
    df_clean['vid_desc_clean'] = df_clean['vid_caption'].fillna('')
    df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(remove_emoji)
    df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(remove_stopwords)

    # Process hashtags
    df_clean['vid_hashtags_normalized'] = df_clean['vid_hashtags'].apply(normalize_hashtags)
    df_clean['hashtag_count'] = df_clean['vid_hashtags_normalized'].apply(len)

    # Convert timestamps
    df_clean['vid_postTime'] = pd.to_datetime(df_clean['vid_postTime'])
    df_clean['vid_scrapeTime'] = pd.to_datetime(df_clean['vid_scrapeTime'])

    # Calculate growth and engagement metrics
    df_clean['growth_rate'] = df_clean.apply(calculate_growth_rate, axis=1)
    df_clean['engagement_rate'] = df_clean.apply(calculate_engagement_rate, axis=1)
    
    # Extract time-based features
    df_clean['post_hour'] = df_clean['vid_postTime'].dt.hour
    df_clean['post_day'] = df_clean['vid_postTime'].dt.day_name()
    df_clean['video_age_hours'] = (df_clean['vid_scrapeTime'] - 
                                  df_clean['vid_postTime']).dt.total_seconds() / 3600

    # Process duration
    def duration_to_seconds(duration):
        if isinstance(duration, str) and ':' in duration:
            parts = duration.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return np.nan

    df_clean['vid_duration_sec'] = df_clean['vid_duration'].apply(duration_to_seconds)

    # Extract trending features
    trending_features = extract_trending_features(df_clean)
    df_clean['hashtag_trends'] = trending_features['hashtag_trends']
    df_clean['sound_trends'] = trending_features['sound_trends']

    # Calculate viral score
    df_clean['viral_score'] = df_clean.apply(calculate_viral_score, axis=1)

    return df_clean

def main():
    infile = "finalProject/data/raw_data.csv"
    outfile = "finalProject/data/processedd_data.csv"

    print(f"Loading data from {infile}...")
    df = pd.read_csv(infile)

    print("Preprocessing data...")
    df_processed = preprocess_tiktok_data(df)

    print(f"Saving processed data to {outfile}...")
    df_processed.to_csv(outfile, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
