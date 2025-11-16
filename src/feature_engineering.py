"""
Feature engineering module for movie dataset

This module creates derived features for analysis including
financial metrics, temporal features and blockbuster indicators
"""

import pandas as pd

def add_financial_metrics(df):
    """Calculate profit and profitability indicators"""
    df = df.copy()

    df_financial = df[df[['budget', 'revenue']].notna().all(axis = 1)].copy()
    df_financial['profit'] = df_financial['revenue'] - df_financial['budget']
    df_financial['is_profitable'] = df_financial['profit'] > 0

    df = df.merge(
        df_financial[['title', 'release_date', 'profit', 'is_profitable']],
        on = ['title', 'release_date'],
        how = 'left'
    )

    print(f"Financial metrics added: {df_financial['is_profitable'].sum()} profitable movies out of {len(df_financial)}")
    return df

def add_temporal_features(df):
    """Add decade and season features"""
    df = df.copy()

    # Add decade
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['decade'] = (df['release_year'] // 10) * 10

    # Add season
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['release_month'] = df['release_date'].dt.month
    df['release_season'] = df['release_month'].map(season_map)

    print("Temporal features added: decade and release_season")

    return df

def add_blockbuster_indicator(df, percentile = 0.90):
    """Identify blockbuster movies based on revenue percentile"""
    df = df.copy()

    revenue_with_data = df[df['revenue'].notna()].copy()
    blockbuster_threshold = revenue_with_data['revenue'].quantile(percentile)
    revenue_with_data['is_blockbuster'] = revenue_with_data['revenue'] >= blockbuster_threshold

    df = df.merge(
        revenue_with_data[['title', 'release_date', 'is_blockbuster']],
        on = ['title', 'release_date'],
        how = 'left'
    )

    print(f"Blockbuster indicator added (90th percentile threshold: ${blockbuster_threshold:,.0f})")
    print(f"Number of blockbusters identified: {revenue_with_data['is_blockbuster'].sum()}")

    return df

def engineer_features(df):
    """Execute complete feature engineering pipeline"""
    print("\nFeature Engineering:")
    print("-" * 80)

    print("Adding financial metrics...")
    df = add_financial_metrics(df)

    print("Adding temporal features...")
    df = add_temporal_features(df)

    print("Adding blockbuster indicator...")
    df = add_blockbuster_indicator(df)

    print("Feature engineering complete")

    return df