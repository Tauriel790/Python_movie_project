"""
Analysis module for movie dataset

This module contains functions to perform statistical analysis
on the movie dataset to identify blockbuster patterns

Author: Legolas
Date: 2024
"""

import pandas as pd
import numpy as np


def analyze_budget_revenue_correlation(df):
    """
    Analyze correlation between budget and revenue
    
    Examines the relationship between production budget and box office
    revenue, breaking down performance by budget category (low, mid, high).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and revenue columns
    
    Returns:
    --------
    dict
        Dictionary containing correlation coefficient and category statistics
    """
    clean_data = df[['budget', 'revenue']].dropna()
    correlation = clean_data['budget'].corr(clean_data['revenue'])

    # Budget categories
    low_budget = clean_data[clean_data['budget'] < 1e6]
    mid_budget = clean_data[(clean_data['budget'] >= 1e6) & (clean_data['budget'] < 5e7)]
    high_budget = clean_data[clean_data['budget'] >= 5e7]

    results = {
        'correlation': correlation,
        'low_budget_count': len(low_budget),
        'low_budget_avg_revenue': low_budget['revenue'].mean() / 1e6,
        'mid_budget_count': len(mid_budget),
        'mid_budget_avg_revenue': mid_budget['revenue'].mean() / 1e6,
        'high_budget_count': len(high_budget),
        'high_budget_avg_revenue': high_budget['revenue'].mean() / 1e6
    }

    return results


def analyze_profitability_distribution(df):
    """
    Analyze distribution of profitable vs unprofitable films
    
    Calculates the proportion of films that generate positive returns
    (revenue > budget) vs those that lose money.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and profit columns
    
    Returns:
    --------
    dict
        Dictionary containing profitability counts and percentages
    """
    clean_profit_data = df[['budget', 'profit']].dropna()
    profitability_counts = clean_profit_data['profit'].apply(
        lambda x: 'Profitable' if x > 0 else 'Unprofitable'
    ).value_counts()

    total = profitability_counts.sum()
    profitable_count = profitability_counts.get('Profitable', 0)
    unprofitable_count = profitability_counts.get('Unprofitable', 0)

    return {
        'total': total,
        'profitable': profitable_count,
        'unprofitable': unprofitable_count,
        'profitable_pct': (profitable_count / total) * 100
    }


def analyze_rating_revenue_correlation(df):
    """
    Analyze correlation between ratings and revenue
    
    Examines whether critical acclaim (vote_average) predicts
    commercial success (revenue).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with vote_average and revenue columns
    
    Returns:
    --------
    dict
        Dictionary containing correlation coefficient and sample size
    """
    rating_revenue = df[['vote_average', 'revenue']].dropna()
    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(rating_revenue)}


def analyze_runtime_revenue_correlation(df):
    """
    Analyze correlation between runtime and revenue
    
    Examines whether movie length influences box office performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with runtime and revenue columns
    
    Returns:
    --------
    dict
        Dictionary containing correlation coefficient and sample size
    """
    runtime_revenue = df[['runtime', 'revenue']].dropna()
    correlation = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(runtime_revenue)}


def analyze_popularity_revenue_correlation(df):
    """
    Analyze correlation between popularity and revenue
    
    Examines whether pre-release buzz and marketing visibility
    (popularity score) predicts box office success.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with popularity and revenue columns
    
    Returns:
    --------
    dict
        Dictionary containing correlation coefficient and sample size
    """
    popularity_revenue = df[['popularity', 'revenue']].dropna()
    correlation = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(popularity_revenue)}


def analyze_genre_performance(df):
    """
    Analyze top performing genres by revenue
    
    Identifies which film genres generate the highest average revenue,
    filtering for genres with sufficient sample size (≥50 films).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with primary_genre and revenue columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with top 10 genres sorted by average revenue
    """
    genre_data = df[['primary_genre', 'revenue']].dropna()
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    genre_stats.columns = ['genre', 'avg_revenue', 'count']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    genre_stats = genre_stats[genre_stats['count'] >= 50]
    genre_stats = genre_stats.sort_values('avg_revenue', ascending=False).head(10)

    return genre_stats


def analyze_temporal_trends(df):
    """
    Analyze revenue trends by decade
    
    Examines how average film revenue has evolved over time,
    filtering for decades with sufficient sample size (≥50 films).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with decade and revenue columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with decade statistics sorted chronologically
    """
    decade_data = df[['decade', 'revenue']].dropna()
    decade_stats = decade_data.groupby('decade').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    decade_stats.columns = ['decade', 'avg_revenue', 'count']
    decade_stats['avg_revenue'] = decade_stats['avg_revenue'] / 1e6
    decade_stats = decade_stats[decade_stats['count'] >= 50]

    return decade_stats


def analyze_seasonal_patterns(df):
    """
    Analyze revenue patterns by release season
    
    Examines whether release timing (Winter, Spring, Summer, Fall)
    influences box office performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with release_season and revenue columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with seasonal statistics
    """
    season_data = df[['release_season', 'revenue']].dropna()
    season_stats = season_data.groupby('release_season').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    season_stats.columns = ['season', 'avg_revenue', 'count']
    season_stats['avg_revenue'] = season_stats['avg_revenue'] / 1e6

    return season_stats


def print_analysis_summary(df):
    """
    Print comprehensive analysis summary
    
    Generates a formatted report of all key statistical findings,
    including correlations, profitability, genre performance, and
    temporal patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fully processed dataframe with all features
    
    Returns:
    --------
    None
        Prints formatted summary to console
    """
    print("\n" + "="*80)
    print("BLOCKBUSTER ANALYSIS SUMMARY")
    print("="*80)

    # Budget vs Revenue
    budget_results = analyze_budget_revenue_correlation(df)
    print(f"\n1. BUDGET VS REVENUE")
    print(f"   Correlation: {budget_results['correlation']:.3f}")
    print(f"   Low budget (<$1M): {budget_results['low_budget_count']} films, "
          f"avg revenue ${budget_results['low_budget_avg_revenue']:.1f}M")
    print(f"   Mid budget ($1M-$50M): {budget_results['mid_budget_count']} films, "
          f"avg revenue ${budget_results['mid_budget_avg_revenue']:.1f}M")
    print(f"   High budget (>$50M): {budget_results['high_budget_count']} films, "
          f"avg revenue ${budget_results['high_budget_avg_revenue']:.1f}M")
    
    # Profitability
    profit_results = analyze_profitability_distribution(df)
    print(f"\n2. PROFITABILITY")
    print(f"   Total films: {profit_results['total']}")
    print(f"   Profitable: {profit_results['profitable']} ({profit_results['profitable_pct']:.1f}%)")
    print(f"   Unprofitable: {profit_results['unprofitable']}")

    # Correlations
    rating_results = analyze_rating_revenue_correlation(df)
    runtime_results = analyze_runtime_revenue_correlation(df)
    popularity_results = analyze_popularity_revenue_correlation(df)

    print(f"\n3. KEY CORRELATIONS")
    print(f"   Rating vs Revenue: {rating_results['correlation']:.3f}")
    print(f"   Runtime vs Revenue: {runtime_results['correlation']:.3f}")
    print(f"   Popularity vs Revenue: {popularity_results['correlation']:.3f}")

    # Top genres
    genre_stats = analyze_genre_performance(df)
    print(f"\n4. TOP 5 GENRES BY REVENUE")
    for idx, row in genre_stats.head().iterrows():
        print(f"   {row['genre']}: ${row['avg_revenue']:.1f}M (n={int(row['count'])})")

    # Seasonal patterns
    season_stats = analyze_seasonal_patterns(df)
    print(f"\n5. SEASONAL PATTERNS")
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_stats_ordered = season_stats.set_index('season').reindex(season_order).reset_index()
    for idx, row in season_stats_ordered.iterrows():
        print(f"   {row['season']}: ${row['avg_revenue']:.1f}M")

    print("\n" + "="*80 + "\n")