
import pandas as pd
import numpy as np
from datetime import datetime

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

    numeric_cols = ['user_followers', 'vid_nview', 'vid_nlike', 'vid_ncomment', 
                   'vid_nshare', 'vid_nsave', 'music_nused']

    for col in numeric_cols:
        df_clean[col] = df_clean[col].apply(convert_units)

    df_clean['vid_postTime'] = pd.to_datetime(df_clean['vid_postTime'])
    df_clean['vid_scrapeTime'] = pd.to_datetime(df_clean['vid_scrapeTime'])
    df_clean['user_followers'] = df_clean['user_followers'].round(0)
    df_clean['music_nused'] = df_clean['music_nused'].round(0)


    def duration_to_seconds(duration):
        if isinstance(duration, str) and ':' in duration:
            parts = duration.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return np.nan

    df_clean['vid_duration_sec'] = df_clean['vid_duration'].apply(duration_to_seconds)

    df_clean['vid_hashtags'] = df_clean['vid_hashtags'].fillna('')
    df_clean['music_title'] = df_clean['music_title'].fillna('Unknown')
    df_clean['music_authorName'] = df_clean['music_authorName'].fillna('Unknown')


    df_clean['vid_existtime_hrs'] = (df_clean['vid_scrapeTime'] - df_clean['vid_postTime']).dt.total_seconds() / 3600

    def calculate_viral_score(row):
        if row['vid_nview'] == 0 or pd.isna(row['vid_nview']):
            return 0

        engagement = (
            0.4 * (row['vid_nlike'] / row['vid_nview']) +
            0.3 * (row['vid_ncomment'] / row['vid_nview']) +
            0.2 * (row['vid_nshare'] / row['vid_nview']) +
            0.1 * (row['vid_nsave'] / row['vid_nview'])
        ) * 40

        influence = min(20, np.log10(row['user_followers'] + 1) * 4)

        hashtag_count = len(row['vid_hashtags'].split(',')) if row['vid_hashtags'] else 0
        duration_score = max(0, 10 - (row['vid_duration_sec'] / 10))
        content = min(20, (hashtag_count * 2) + duration_score)

        recency = max(0, 20 - (row['vid_existtime_hrs'] / 24)) if row['vid_existtime_hrs'] > 0 else 0
        velocity = min(20, (row['vid_nview'] / row['vid_existtime_hrs']) / 1000) if row['vid_existtime_hrs'] > 0 else 0

        total_score = engagement + influence + content + (recency * 0.5 + velocity * 0.5)
        return min(100, max(0, total_score))

    df_clean['viral_score'] = df_clean.apply(calculate_viral_score, axis=1)

    return df_clean

def main():
    infile = "tiktok_feed_combined4.csv"
    outfile = "processed_tiktok_data.csv"

    print(f"Loading data from {infile}...")
    df = pd.read_csv(infile)

    print("Preprocessing data...")
    df_processed = preprocess_tiktok_data(df)

    print(f"Saving processed data to {outfile}...")
    df_processed.to_csv(outfile, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
