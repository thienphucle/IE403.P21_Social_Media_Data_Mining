import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import timedelta

def load_data(preprocessed_file: str) -> pd.DataFrame:
    return pd.read_csv(preprocessed_file, parse_dates=['vid_postTime', 'vid_scrapeTime'])

def analyze_before_after(df: pd.DataFrame, viral_threshold: float = 70) -> Dict[str, float]:
    # Identify viral videos
    viral_videos = df[df['viral_score'] >= viral_threshold]
    
    results = {}
    
    for viral_video in viral_videos.itertuples():
        # Get videos with same hashtags
        related_videos = df[
            df['vid_hashtags_normalized'].apply(
                lambda x: bool(set(eval(x)) & set(eval(viral_video.vid_hashtags_normalized)))
            )
        ]
        
        # Split into before and after viral video post
        before_videos = related_videos[
            related_videos['vid_postTime'] < viral_video.vid_postTime
        ]
        after_videos = related_videos[
            related_videos['vid_postTime'] > viral_video.vid_postTime
        ]
        
        # Calculate metrics
        if not before_videos.empty and not after_videos.empty:
            results[viral_video.vid_id] = {
                'before_avg_views': before_videos['vid_nview'].mean(),
                'after_avg_views': after_videos['vid_nview'].mean(),
                'before_avg_engagement': (
                    before_videos['vid_nlike'] + 
                    before_videos['vid_ncomment'] + 
                    before_videos['vid_nshare']
                ).mean(),
                'after_avg_engagement': (
                    after_videos['vid_nlike'] + 
                    after_videos['vid_ncomment'] + 
                    after_videos['vid_nshare']
                ).mean(),
                'hashtag_usage_before': len(before_videos),
                'hashtag_usage_after': len(after_videos)
            }
    
    return results

def calculate_impact_metrics(analysis_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not analysis_results:
        return {}
    
    metrics = {
        'avg_view_increase': [],
        'avg_engagement_increase': [],
        'avg_hashtag_usage_increase': []
    }
    
    for video_results in analysis_results.values():
        metrics['avg_view_increase'].append(
            (video_results['after_avg_views'] - video_results['before_avg_views']) / 
            video_results['before_avg_views'] * 100
        )
        
        metrics['avg_engagement_increase'].append(
            (video_results['after_avg_engagement'] - video_results['before_avg_engagement']) / 
            video_results['before_avg_engagement'] * 100
        )
        
        metrics['avg_hashtag_usage_increase'].append(
            (video_results['hashtag_usage_after'] - video_results['hashtag_usage_before']) / 
            video_results['hashtag_usage_before'] * 100
        )
    
    return {
        metric: np.mean(values) for metric, values in metrics.items()
    }

def plot_impact_analysis(analysis_results: Dict[str, Dict[str, float]], output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    metrics = calculate_impact_metrics(analysis_results)
    
    # Plot impact metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Average Impact of Viral Videos')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage Increase (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'viral_impact_metrics.png')
    plt.close()

def main():
    # Load data
    df = load_data('finalProject/data/preprocessed_data.csv')
    
    # Analyze before-after impact
    print("Analyzing viral video impact...")
    analysis_results = analyze_before_after(df)
    
    # Calculate and print impact metrics
    metrics = calculate_impact_metrics(analysis_results)
    print("\nImpact Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    plot_impact_analysis(analysis_results, 'finalProject/results')
    print("Done.")