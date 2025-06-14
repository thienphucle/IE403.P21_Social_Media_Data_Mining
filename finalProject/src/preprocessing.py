import pandas as pd
import numpy as np
from datetime import datetime
import emoji
import re
from dateutil import tz
from pyvi import ViTokenizer

def remove_emoji(text):
    return emoji.replace_emoji(text, '')

# Load stopwords
STOPWORDS = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\data\vietnamese-stopwords-dash.txt"
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

    # Clean text data
    df_clean['vid_desc_clean'] = df_clean['vid_caption'].fillna('')
    df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(remove_emoji)
    df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(remove_stopwords)

    # Process hashtags
    df_clean['vid_hashtags_normalized'] = df_clean['vid_hashtags'].apply(normalize_hashtags)
    df_clean['hashtag_count'] = df_clean['vid_hashtags_normalized'].apply(len)

    # Convert timestamps
    df_clean['vid_postTime'] = df_clean['vid_postTime'].apply(convert_to_vietnam_time)
    df_clean['vid_scrapeTime'] = df_clean['vid_scrapeTime'].apply(convert_to_vietnam_time)

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

    # Fill missing values
    df_clean['vid_hashtags'] = df_clean['vid_hashtags'].fillna('')
    df_clean['music_title'] = df_clean['music_title'].fillna('Unknown')
    df_clean['music_authorName'] = df_clean['music_authorName'].fillna('Unknown')

    # Time-based features
    df_clean['vid_existtime_hrs'] = (df_clean['vid_scrapeTime'] - df_clean['vid_postTime']).dt.total_seconds() / 3600
    df_clean['post_hour'] = df_clean['vid_postTime'].dt.hour
    df_clean['post_day'] = df_clean['vid_postTime'].dt.day_name()

    # Calculate viral score
    df_clean['viral_score'] = df_clean.apply(calculate_viral_score, axis=1)

    return df_clean

def main():
    infile = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\full_recrawl_final.csv"
    outfile = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\full_recrawl_final.csv"

    print(f"Loading data from {infile}...")
    df = pd.read_csv(infile)

    print("Preprocessing data...")
    df_processed = preprocess_tiktok_data(df)

    print(f"Saving preprocessed data to {outfile}...")
    df_processed.to_csv(outfile, index=False)
    print("Done.")

if __name__ == "__main__":
    main()