"""Analysis module for movie dataset

This module contains functions to perform statistical analysis
on the movie dataset to identify blockbuster patterns
"""

import pandas as pd
import numpy as np

def analyze_budget_revenue_correlation(df):
    """Analyze correlation between budget and revenue"""
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
        'mid_budget_avg_revenue': mid_budget['revenue'].mean() /1e6,
        'high_budget_count': len(high_budget),
        'high_budget_avg_revenue': high_budget['revenue'].mean() / 1e6
    }

    return results

def analyze_profitability_distribution(df):
    """Analyze distribution of profitable vs unprofitable films"""
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
    """Analyze correlation between ratings and revenue"""
    rating_revenue = df[['vote_average', 'revenue']].dropna()
    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(rating_revenue)}

def analyze_runtime_revenue_correlation(df):
    """Analyze correlation between runtime and revenue"""
    runtime_revenue = df[['runtime', 'revenue']].dropna()
    correlation = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(runtime_revenue)}

def analyze_popularity_revenue_correlation(df):
    """Analyze correlation between popularity and revenue"""
    popularity_revenue = df[['popularity', 'revenue']].dropna()
    correlation = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])

    return {'correlation': correlation, 'sample_size': len(popularity_revenue)}

def analyze_genre_performance(df):
    """Analyze top performing genres by revenue"""
    genre_data = df[['primary_genre', 'revenue']].dropna()
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    genre_stats.columns = ['genre', 'avg_revenue', 'count']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    genre_stats = genre_stats[genre_stats['count'] >= 50]
    genre_stats = genre_stats.sort_values('avg_revenue', ascending = False).head(10)

    return genre_stats

def analyze_temporal_trends(df):
    """Analyze revenue trends by decade"""
    decade_data = df[['decade', 'revenue']].dropna()
    decade_stats = decade_data.groupby('decade').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    decade_stats.columns = ['decade', 'avg_revenue', 'count']
    decade_stats['avg_revenue'] = decade_stats['avg_revenue'] / 1e6
    decade_stats = decade_stats[decade_stats['count'] >= 50]

    return decade_stats

def analyze_seasonal_patterns(df):
    """Analyze revenue patterns by release season"""
    season_data = df[['release_season', 'revenue']].dropna()
    season_stats = season_data.groupby('release_season').agg({
        'revenue': ['mean', 'count']
    }).reset_index()

    season_stats.columns = ['season', 'avg_revenue', 'count']
    season_stats['avg_revenue'] = season_stats['avg_revenue'] / 1e6

    return season_stats

def print_analysis_summary(df):
    """Print comprehensive analysis summary"""
    print("BLOCKBUSTER ANALYSIS SUMMARY")
    print("=" * 80)

    # Budget vs Revenue
    budget_results = analyze_budget_revenue_correlation(df)
    print(f"\n1. BUDGET VS REVENUE")
    print(f"Correlation: {budget_results['correlation']:.3f}")
    print(f"Low budget (<$1M): {budget_results['low_budget_count']} films, "
          f"avg revenue ${budget_results['low_budget_avg_revenue']:.1f}M")
    print(f"Mid budget ($1M-$50M): {budget_results['mid_budget_count']} films, "
          f"avg revenue ${budget_results['mid_budget_avg_revenue']:.1f}M")
    print(f"High budget (>$50M): {budget_results['high_budget_count']} films, "
          f"avg revenue ${budget_results['high_budget_avg_revenue']:.1f}M")
    
    # Profitability
    profit_results = analyze_profitability_distribution(df)
    print(f"\n2. PROFITABILITY")
    print(f"Total films: {profit_results['total']}")
    print(f"Profitable: {profit_results['profitable']} ({profit_results['profitable_pct']:.1f}%)")
    print(f"Unprofitable: {profit_results['unprofitable']}")

    # Correlations
    rating_results = analyze_rating_revenue_correlation(df)
    runtime_results = analyze_runtime_revenue_correlation(df)
    popularity_results = analyze_popularity_revenue_correlation(df)

    print(f"\n3. KEY CORRELATIONS")
    print(f"Rating vs Revenue: {rating_results['correlation']:.3f}")
    print(f"Runtime vs Revenue: {runtime_results['correlation']:.3f}")
    print(f"Popularity vs Revenue: {popularity_results['correlation']:.3f}")

    # Top genres
    genre_stats = analyze_genre_performance(df)
    print(f"\n4. TOP 5 GENRES BY REVENUE")
    for idx, row in genre_stats.head().iterrows():
        print(f"{row['genre']}: ${row['avg_revenue']:.1f}M (n={int(row['count'])})")

    # Seasonal patterns
    season_stats = analyze_seasonal_patterns(df)
    print(f"\n5. SEASONAL PATTERNS")
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_stats_ordered = season_stats.set_index('season').reindex(season_order).reset_index()
    for idx, row in season_stats_ordered.iterrows():
        print(f"{row['season']}: ${row['avg_revenue']:.1f}M")

    print("\n" + "=" * 80 + "\n")