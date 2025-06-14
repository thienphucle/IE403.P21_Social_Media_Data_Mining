import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


# load data va chuyen dinh dang datatime
def load_data(preprocessed_file: str, user_videos_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both preprocessed data and user videos data"""
    df_main = pd.read_csv(preprocessed_file, parse_dates=['vid_postTime', 'vid_scrapeTime'])
    df_user_videos = pd.read_csv(user_videos_file, parse_dates=['vid_postTime', 'vid_scrapeTime'])
    return df_main, df_user_videos

# Chon cac vids viral voi thresold > 80 
def identify_viral_videos(df: pd.DataFrame, viral_threshold: float = 80) -> pd.DataFrame:
    """Identify viral videos based on viral score threshold"""
    return df[df['viral_score'] >= viral_threshold].copy()

# phan  tich anh huong hoat dong nguoi dung 
def analyze_user_activity_impact(df_main: pd.DataFrame, df_user_videos: pd.DataFrame, 
                               viral_threshold: float = 80) -> Dict[str, Dict]:
    """
    Analyze the impact of viral videos on user's subsequent content performance
    """
    # Get viral videos from main dataset
    viral_videos = identify_viral_videos(df_main, viral_threshold)
    
    results = {}
    
    for _, viral_video in viral_videos.iterrows():
        user_name = viral_video['user_name']
        viral_video_id = str(viral_video['vid_id'])
        viral_post_time = viral_video['vid_postTime']
        
        # Get all videos from this user
        user_videos = df_user_videos[df_user_videos['user_name'] == user_name].copy()
        
        # Bỏ những người dùng it hơn 5 vids 
        if len(user_videos) < 5:  # Need sufficient data
            continue
            
        # Sort by post time
        user_videos = user_videos.sort_values('vid_postTime')
        
        # Find videos before and after the viral video
        before_videos = user_videos[user_videos['vid_postTime'] < viral_post_time]
        after_videos = user_videos[user_videos['vid_postTime'] > viral_post_time]
        
        # Ensure we have videos both before and after
        if len(before_videos) == 0 or len(after_videos) == 0:
            continue
        
        # Calculate metrics for before and after periods
        before_metrics = calculate_period_metrics(before_videos)
        after_metrics = calculate_period_metrics(after_videos)
        
        # Calculate temporal patterns
        before_temporal = analyze_temporal_patterns(before_videos, viral_post_time, 'before')
        after_temporal = analyze_temporal_patterns(after_videos, viral_post_time, 'after')
        
        # Calculate content strategy changes
        content_changes = analyze_content_strategy_changes(before_videos, after_videos)
        
        results[viral_video_id] = {
            'user_name': user_name,
            'viral_video': {
                'vid_id': viral_video_id,
                'viral_score': viral_video['viral_score'],
                'vid_nview': viral_video['vid_nview'],
                'vid_nlike': viral_video['vid_nlike'],
                'post_time': viral_post_time
            },
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'before_temporal': before_temporal,
            'after_temporal': after_temporal,
            'content_changes': content_changes,
            'impact_metrics': calculate_impact_metrics(before_metrics, after_metrics)
        }
    
    return results

def calculate_period_metrics(videos: pd.DataFrame) -> Dict[str, float]:
    """Calculate aggregated metrics for a period of videos"""
    if len(videos) == 0:
        return {}
    
    # Fillna(0)
    # Chuẩn hóa các số liệu 
    def safe_convert(series):
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    views = safe_convert(videos['vid_nview'])
    likes = safe_convert(videos['vid_nlike'])
    comments = safe_convert(videos['vid_ncomment'])
    shares = safe_convert(videos['vid_nshare'])
    saves = safe_convert(videos['vid_nsave'])

    
    # Calculate engagement rates
    # Engagement Rate= (Lượt xem + Tổng tương tác) ×100
    # Nếu views = 0 thì gán Engagement Rate = 0 
    total_engagement = likes + comments + shares + saves
    engagement_rates = np.where(views > 0, (total_engagement / views) * 100, 0)
    
    # Calculate posting frequency
    # nếu chỉ có 1 vids thì frequency được gán = 0 
    if len(videos) > 1:
        time_span = (videos['vid_postTime'].max() - videos['vid_postTime'].min()).total_seconds() / (24 * 3600)  # days
        posting_frequency = len(videos) / max(time_span, 1)
    else:
        posting_frequency = 0
    
    return {
        'avg_views': views.mean(),
        'median_views': views.median(),
        'max_views': views.max(),
        'avg_likes': likes.mean(),
        'avg_comments': comments.mean(),
        'avg_shares': shares.mean(),
        'avg_saves': saves.mean(),
        'avg_engagement_rate': engagement_rates.mean(),
        'median_engagement_rate': np.median(engagement_rates),
        'total_videos': len(videos),
        'posting_frequency': posting_frequency,
        'view_consistency': views.std() / max(views.mean(), 1),  # Coefficient of variation
        'engagement_consistency': np.std(engagement_rates) / max(np.mean(engagement_rates), 1) 
    }

def analyze_temporal_patterns(videos: pd.DataFrame, viral_time: pd.Timestamp, period: str) -> Dict[str, float]:
    """Analyze temporal patterns in video performance"""
    if len(videos) == 0:
        return {}
    
    videos = videos.copy()
    
    # Calculate time differences from viral video
    # khoảng cách thời gian giữa vids đầu và vids sau so vói thoi gian viral  theo ngày 
    if period == 'before':
        videos['days_to_viral'] = (viral_time - videos['vid_postTime']).dt.total_seconds() / (24 * 3600)
    else:  # after
        videos['days_from_viral'] = (videos['vid_postTime'] - viral_time).dt.total_seconds() / (24 * 3600)
    
    # Convert views to numeric
    views = pd.to_numeric(videos['vid_nview'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # Analyze performance trends
    if len(videos) > 2:
        # Calculate correlation between time and performance
        # Càng gần viral_time: view cao hơn 
        time_col = 'days_to_viral' if period == 'before' else 'days_from_viral'
        time_correlation = np.corrcoef(videos[time_col], views)[0, 1] if not np.isnan(views).all() else 0
        
        # Calculate performance momentum (trend)
        # Lấy 3 vids mới nhất và 3 vids cũ nhất 
        if len(videos) >= 3:
            recent_videos = videos.nlargest(3, 'vid_postTime') if period == 'after' else videos.nsmallest(3, 'vid_postTime')
            older_videos = videos.nsmallest(3, 'vid_postTime') if period == 'after' else videos.nlargest(3, 'vid_postTime')
            
            recent_avg = pd.to_numeric(recent_videos['vid_nview'].astype(str).str.replace(',', ''), errors='coerce').mean()
            older_avg = pd.to_numeric(older_videos['vid_nview'].astype(str).str.replace(',', ''), errors='coerce').mean()
            
            # mức độ tăng trưởng view 
            momentum = (recent_avg - older_avg) / max(older_avg, 1) * 100
        else:
            momentum = 0
    else:
        time_correlation = 0
        momentum = 0
    
    return {
        'time_correlation': time_correlation,
        'performance_momentum': momentum,
        'avg_time_gap': videos['vid_postTime'].diff().dt.total_seconds().mean() / (24 * 3600) if len(videos) > 1 else 0
    }

def analyze_content_strategy_changes(before_videos: pd.DataFrame, after_videos: pd.DataFrame) -> Dict[str, float]:
    """Analyze changes in content strategy before and after viral video"""
    
    def extract_hashtag_stats(videos):
        if len(videos) == 0:
            return {'avg_hashtags': 0, 'unique_hashtags': 0, 'hashtag_diversity': 0}
        
        all_hashtags = []
        hashtag_counts = []
        
        for _, video in videos.iterrows():
            hashtags = video.get('vid_hashtags_normalized', '[]')
            if isinstance(hashtags, str):
                try:
                    hashtags = eval(hashtags) if hashtags.startswith('[') else []
                except:
                    hashtags = []
            elif not isinstance(hashtags, list):
                hashtags = []
            
            hashtag_counts.append(len(hashtags))
            all_hashtags.extend(hashtags)
        
        unique_hashtags = len(set(all_hashtags))
        avg_hashtags = np.mean(hashtag_counts) if hashtag_counts else 0
        hashtag_diversity = unique_hashtags / max(len(all_hashtags), 1)
        
        return {
            'avg_hashtags': avg_hashtags,
            'unique_hashtags': unique_hashtags,
            'hashtag_diversity': hashtag_diversity
        }
    
    def extract_duration_stats(videos):
        if len(videos) == 0:
            return {'avg_duration': 0}
        
        durations = []
        for _, video in videos.iterrows():
            duration_sec = video.get('vid_duration_sec', 0)
            if pd.notna(duration_sec):
                durations.append(duration_sec)
        
        return {'avg_duration': np.mean(durations) if durations else 0}
    
    before_hashtags = extract_hashtag_stats(before_videos)
    after_hashtags = extract_hashtag_stats(after_videos)
    
    before_duration = extract_duration_stats(before_videos)
    after_duration = extract_duration_stats(after_videos)
    
    return {
        'hashtag_usage_change': after_hashtags['avg_hashtags'] - before_hashtags['avg_hashtags'],
        'hashtag_diversity_change': after_hashtags['hashtag_diversity'] - before_hashtags['hashtag_diversity'],
        'duration_change': after_duration['avg_duration'] - before_duration['avg_duration'],
        'posting_frequency_change': after_videos.shape[0] - before_videos.shape[0]  # Simple count difference
    }

def calculate_impact_metrics(before_metrics: Dict, after_metrics: Dict) -> Dict[str, float]:
    """Calculate the impact of viral video on user's subsequent performance"""
    if not before_metrics or not after_metrics:
        return {}
    
    def safe_percentage_change(before_val, after_val):
        if before_val == 0:
            return 100 if after_val > 0 else 0
        return ((after_val - before_val) / before_val) * 100
    
    return {
        'view_change_pct': safe_percentage_change(before_metrics['avg_views'], after_metrics['avg_views']),
        'like_change_pct': safe_percentage_change(before_metrics['avg_likes'], after_metrics['avg_likes']),
        'engagement_rate_change_pct': safe_percentage_change(before_metrics['avg_engagement_rate'], after_metrics['avg_engagement_rate']),
        'posting_frequency_change_pct': safe_percentage_change(before_metrics['posting_frequency'], after_metrics['posting_frequency']),
        'consistency_change': after_metrics['view_consistency'] - before_metrics['view_consistency']
    }

def calculate_aggregate_impact(analysis_results: Dict) -> Dict[str, float]:
    """Calculate aggregate impact metrics across all users"""
    if not analysis_results:
        return {}
    
    metrics = {
        'view_changes': [],
        'like_changes': [],
        'engagement_changes': [],
        'frequency_changes': [],
        'consistency_changes': []
    }
    
    for result in analysis_results.values():
        impact = result.get('impact_metrics', {})
        if impact:
            metrics['view_changes'].append(impact.get('view_change_pct', 0))
            metrics['like_changes'].append(impact.get('like_change_pct', 0))
            metrics['engagement_changes'].append(impact.get('engagement_rate_change_pct', 0))
            metrics['frequency_changes'].append(impact.get('posting_frequency_change_pct', 0))
            metrics['consistency_changes'].append(impact.get('consistency_change', 0))
    
    return {
        'avg_view_change': np.mean(metrics['view_changes']) if metrics['view_changes'] else 0,
        'median_view_change': np.median(metrics['view_changes']) if metrics['view_changes'] else 0,
        'avg_engagement_change': np.mean(metrics['engagement_changes']) if metrics['engagement_changes'] else 0,
        'avg_frequency_change': np.mean(metrics['frequency_changes']) if metrics['frequency_changes'] else 0,
        'users_with_improved_views': sum(1 for x in metrics['view_changes'] if x > 0),
        'users_with_improved_engagement': sum(1 for x in metrics['engagement_changes'] if x > 0),
        'total_users_analyzed': len(metrics['view_changes'])
    }

def create_impact_visualizations(analysis_results: Dict, aggregate_metrics: Dict, output_dir: str):
    """Create comprehensive visualizations of viral video impact"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Impact Distribution
    plt.subplot(3, 4, 1)
    view_changes = [result['impact_metrics'].get('view_change_pct', 0) 
                   for result in analysis_results.values() 
                   if result.get('impact_metrics')]
    
    plt.hist(view_changes, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('View Change (%)')
    plt.ylabel('Number of Users')
    plt.title('Distribution of View Changes\nAfter Viral Video')
    
    # 2. Engagement Impact
    plt.subplot(3, 4, 2)
    engagement_changes = [result['impact_metrics'].get('engagement_rate_change_pct', 0) 
                         for result in analysis_results.values() 
                         if result.get('impact_metrics')]
    
    plt.hist(engagement_changes, bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Engagement Rate Change (%)')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Engagement Changes\nAfter Viral Video')
    
    # 3. Before vs After Performance Comparison
    plt.subplot(3, 4, 3)
    before_views = [result['before_metrics'].get('avg_views', 0) 
                   for result in analysis_results.values() 
                   if result.get('before_metrics')]
    after_views = [result['after_metrics'].get('avg_views', 0) 
                  for result in analysis_results.values() 
                  if result.get('after_metrics')]
    
    plt.scatter(before_views, after_views, alpha=0.6)
    max_val = max(max(before_views) if before_views else 0, max(after_views) if after_views else 0)
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    plt.xlabel('Average Views Before Viral')
    plt.ylabel('Average Views After Viral')
    plt.title('Before vs After Performance')
    plt.xscale('log')
    plt.yscale('log')
    
    # 4. Posting Frequency Changes
    plt.subplot(3, 4, 4)
    freq_changes = [result['impact_metrics'].get('posting_frequency_change_pct', 0) 
                   for result in analysis_results.values() 
                   if result.get('impact_metrics')]
    
    plt.hist(freq_changes, bins=15, alpha=0.7, edgecolor='black', color='green')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Posting Frequency Change (%)')
    plt.ylabel('Number of Users')
    plt.title('Changes in Posting Frequency')
    
    # 5. Content Strategy Changes
    plt.subplot(3, 4, 5)
    hashtag_changes = [result['content_changes'].get('hashtag_usage_change', 0) 
                      for result in analysis_results.values() 
                      if result.get('content_changes')]
    duration_changes = [result['content_changes'].get('duration_change', 0) 
                       for result in analysis_results.values() 
                       if result.get('content_changes')]
    
    plt.scatter(hashtag_changes, duration_changes, alpha=0.6, color='purple')
    plt.xlabel('Change in Hashtag Usage')
    plt.ylabel('Change in Video Duration (sec)')
    plt.title('Content Strategy Changes')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 6. Success Rate by User Follower Count
    plt.subplot(3, 4, 6)
    follower_counts = []
    success_rates = []
    
    for result in analysis_results.values():
        if result.get('before_metrics') and result.get('impact_metrics'):
            # Get follower count from viral video info
            followers = result['viral_video'].get('user_nfollower', 0)
            view_improvement = result['impact_metrics'].get('view_change_pct', 0) > 0
            
            follower_counts.append(followers)
            success_rates.append(1 if view_improvement else 0)
    
    if follower_counts:
        min_f, max_f = min(follower_counts), max(follower_counts)

        if min_f == max_f:
            bins = [min_f - 1, min_f + 1]  # Tạo 1 bin duy nhất đủ bao hết giá trị
        else:
            bins = np.logspace(np.log10(min_f), np.log10(max_f), 5)

        digitized = np.digitize(follower_counts, bins, right=True)

        bin_success_rates = []
        bin_labels = []

        for i in range(1, len(bins)):
            indices = [j for j, d in enumerate(digitized) if d == i]
            if indices:
                success_rate = np.mean([success_rates[j] for j in indices])
                bin_success_rates.append(success_rate)
                bin_labels.append(f'{int(bins[i-1])}-{int(bins[i])}')

        if bin_success_rates:
            plt.bar(range(len(bin_success_rates)), bin_success_rates, alpha=0.7)
            plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
            plt.ylabel('Success Rate')
            plt.xlabel('Follower Count Range')
            plt.title('Success Rate by Follower Count')
        else:
            plt.text(0.5, 0.5, 'No data available for bins', ha='center', va='center')
            plt.axis('off')

    
    # 7. Temporal Patterns
    plt.subplot(3, 4, 7)
    before_momentum = [result['before_temporal'].get('performance_momentum', 0) 
                      for result in analysis_results.values() 
                      if result.get('before_temporal')]
    after_momentum = [result['after_temporal'].get('performance_momentum', 0) 
                     for result in analysis_results.values() 
                     if result.get('after_temporal')]
    
    plt.scatter(before_momentum, after_momentum, alpha=0.6, color='red')
    max_momentum = max(max(before_momentum) if before_momentum else 0, 
                      max(after_momentum) if after_momentum else 0)
    min_momentum = min(min(before_momentum) if before_momentum else 0, 
                      min(after_momentum) if after_momentum else 0)
    plt.plot([min_momentum, max_momentum], [min_momentum, max_momentum], 'k--', alpha=0.7)
    plt.xlabel('Performance Momentum Before')
    plt.ylabel('Performance Momentum After')
    plt.title('Performance Momentum Changes')
    
    """
    # 8. Summary Statistics
    plt.subplot(3, 4, 8)
    plt.axis('off')

    summary_lines = [
        "VIRAL VIDEO IMPACT SUMMARY",
        "",
        f"Total Users Analyzed: {aggregate_metrics.get('total_users_analyzed', 0)}",
        f"Average View Change: {aggregate_metrics.get('avg_view_change', 0):.1f}%",
        f"Users with Improved Views: {aggregate_metrics.get('users_with_improved_views', 0)}",
        f"Average Engagement Change: {aggregate_metrics.get('avg_engagement_change', 0):.1f}%",
        f"Users with Improved Engagement: {aggregate_metrics.get('users_with_improved_engagement', 0)}",
        f"Average Frequency Change: {aggregate_metrics.get('avg_frequency_change', 0):.1f}%"
    ]
    summary_text = "\n".join(summary_lines)

    plt.text(0.05, 0.95, summary_text,
         transform=plt.gca().transAxes,
         fontsize=12, fontweight='bold', va='top', fontfamily='monospace')
    """ 

    # Subplot 8: Summary bar chart + text
    plt.subplot(3, 4, 8)

    # Dữ liệu
    metrics = ['View Change', 'Engagement', 'Frequency']
    values = [
        aggregate_metrics.get('avg_view_change', 0),
        aggregate_metrics.get('avg_engagement_change', 0),
        aggregate_metrics.get('avg_frequency_change', 0)
    ]
    colors = ['skyblue', 'salmon', 'lightgreen']

    # Vẽ bar chart
    bars = plt.bar(metrics, values, color=colors)

    # Ghi giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    #   Thêm thông tin tổng hợp trong biểu đồ (góc dưới hoặc trái)
    text_summary = (
        f"Total Users: {aggregate_metrics.get('total_users_analyzed', 0)}\n"
        f"↑ Views: {aggregate_metrics.get('users_with_improved_views', 0)}\n"
        f"↑ Engagement: {aggregate_metrics.get('users_with_improved_engagement', 0)}"
    )

    # Vị trí text: bạn có thể thay đổi x và y cho phù hợp
    plt.text(-0.4, max(values) * 0.6, text_summary,
         fontsize=9, fontfamily='monospace',
         bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.4'))

    # Cài đặt trục
    plt.title("VIRAL IMPACT SUMMARY", fontsize=11)
    plt.ylabel("Change (%)")
    plt.ylim(0, max(values) * 1.3)



 
    # 9-12. Individual User Case Studies (top 4 performers)
    top_performers = sorted(analysis_results.items(), 
                           key=lambda x: x[1]['impact_metrics'].get('view_change_pct', 0), 
                           reverse=True)[:4]
    
    for i, (vid_id, result) in enumerate(top_performers):
        plt.subplot(3, 4, 9 + i)
        
        # Create a simple before/after comparison
        metrics = ['Views', 'Likes', 'Engagement Rate']
        before_vals = [
            result['before_metrics'].get('avg_views', 0),
            result['before_metrics'].get('avg_likes', 0),
            result['before_metrics'].get('avg_engagement_rate', 0)
        ]
        after_vals = [
            result['after_metrics'].get('avg_views', 0),
            result['after_metrics'].get('avg_likes', 0),
            result['after_metrics'].get('avg_engagement_rate', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, before_vals, width, label='Before', alpha=0.7)
        plt.bar(x + width/2, after_vals, width, label='After', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'User: {result["user_name"][:10]}...\nView Change: {result["impact_metrics"].get("view_change_pct", 0):.1f}%')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viral_impact_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate detailed timeline plot for top users
    create_timeline_analysis(analysis_results, output_dir)

def create_timeline_analysis(analysis_results: Dict, output_dir: Path):
    """Create detailed timeline analysis for top performing users"""
    top_users = sorted(analysis_results.items(), 
                      key=lambda x: x[1]['impact_metrics'].get('view_change_pct', 0), 
                      reverse=True)[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (vid_id, result) in enumerate(top_users):
        ax = axes[i]
        
        # Extract timeline data (this would need actual video data with timestamps)
        user_name = result['user_name']
        viral_time = result['viral_video']['post_time']
        
        # Create a conceptual timeline visualization
        before_metrics = result['before_metrics']
        after_metrics = result['after_metrics']
        
        # Simulate timeline data points
        timeline_points = [-30, -20, -10, 0, 10, 20, 30]  # Days relative to viral video
        performance_trend = [
            before_metrics.get('avg_views', 0) * 0.8,
            before_metrics.get('avg_views', 0) * 0.9,
            before_metrics.get('avg_views', 0),
            result['viral_video']['vid_nview'],  # Viral video performance
            after_metrics.get('avg_views', 0) * 1.2,
            after_metrics.get('avg_views', 0) * 1.1,
            after_metrics.get('avg_views', 0)
        ]
        
        ax.plot(timeline_points, performance_trend, 'o-', linewidth=2, markersize=6)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Viral Video')
        ax.set_xlabel('Days from Viral Video')
        ax.set_ylabel('Average Views')
        ax.set_title(f'{user_name[:15]}...\nImpact: {result["impact_metrics"].get("view_change_pct", 0):.1f}%')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'viral_impact_timeline_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(analysis_results: Dict, aggregate_metrics: Dict, output_dir: str):
    """Save detailed analysis results to CSV files"""
    output_dir = Path(output_dir)
    
    # Create summary DataFrame
    summary_data = []
    for vid_id, result in analysis_results.items():
        summary_data.append({
            'viral_video_id': vid_id,
            'user_name': result['user_name'],
            'viral_score': result['viral_video']['viral_score'],
            'viral_views': result['viral_video']['vid_nview'],
            'before_avg_views': result['before_metrics'].get('avg_views', 0),
            'after_avg_views': result['after_metrics'].get('avg_views', 0),
            'view_change_pct': result['impact_metrics'].get('view_change_pct', 0),
            'before_avg_engagement': result['before_metrics'].get('avg_engagement_rate', 0),
            'after_avg_engagement': result['after_metrics'].get('avg_engagement_rate', 0),
            'engagement_change_pct': result['impact_metrics'].get('engagement_rate_change_pct', 0),
            'posting_frequency_change': result['impact_metrics'].get('posting_frequency_change_pct', 0),
            'hashtag_usage_change': result['content_changes'].get('hashtag_usage_change', 0),
            'duration_change': result['content_changes'].get('duration_change', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'viral_impact_summary.csv', index=False)
    
    # Save aggregate metrics
    with open(output_dir / 'aggregate_impact_metrics.txt', 'w') as f:
        f.write("VIRAL VIDEO IMPACT ANALYSIS - AGGREGATE METRICS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in aggregate_metrics.items():
            f.write(f"{key}: {value}\n")

def main():
    # File paths
    preprocessed_file = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\results\filtered_user.csv"
    user_videos_file = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\full_recrawl_final.csv"  # This should be the combined user videos file
    output_dir = r"D:\UIT\DS200\IE403\IE403.P21_Social_Media_Data_Mining\finalProject\analysis_results"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    try:
        df_main, df_user_videos = load_data(preprocessed_file, user_videos_file)
        print(f"Loaded {len(df_main)} main videos and {len(df_user_videos)} user videos")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure both preprocessed_data.csv and user videos CSV files exist")
        return
    
    print("Analyzing viral video impact on user activities...")
    analysis_results = analyze_user_activity_impact(df_main, df_user_videos, viral_threshold=80)
    
    if not analysis_results:
        print("No viral videos found with sufficient user data for analysis")
        return
    
    print(f"Analyzed {len(analysis_results)} viral videos and their impact on user activities")
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_impact(analysis_results)
    
    # Print summary results
    print("\n" + "="*60)
    print("VIRAL VIDEO IMPACT ANALYSIS RESULTS")
    print("="*60)
    print(f"Total users analyzed: {aggregate_metrics.get('total_users_analyzed', 0)}")
    print(f"Average view change: {aggregate_metrics.get('avg_view_change', 0):.1f}%")
    print(f"Users with improved views: {aggregate_metrics.get('users_with_improved_views', 0)}")
    print(f"Average engagement change: {aggregate_metrics.get('avg_engagement_change', 0):.1f}%")
    print(f"Users with improved engagement: {aggregate_metrics.get('users_with_improved_engagement', 0)}")
    
    # Create visualizations
    print("\nGenerating comprehensive visualizations...")
    create_impact_visualizations(analysis_results, aggregate_metrics, output_dir)
    
    # Save detailed results
    print("Saving detailed results...")
    save_detailed_results(analysis_results, aggregate_metrics, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print("Generated files:")
    print("- viral_impact_comprehensive_analysis.png")
    print("- viral_impact_timeline_analysis.png") 
    print("- viral_impact_summary.csv")
    print("- aggregate_impact_metrics.txt")

if __name__ == "__main__":
    main()