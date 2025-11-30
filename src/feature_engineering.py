"""
Feature engineering module for movie dataset

This module creates derived features for analysis including
financial metrics, temporal features and blockbuster indicators

"""

import pandas as pd


def add_financial_metrics(df):
    """
    Calculate profit and profitability indicators
    
    Creates profit and is_profitable columns for movies with complete
    budget and revenue data. These metrics are essential for understanding
    which films generate returns on investment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and revenue columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with profit and is_profitable columns added
    """
    # Create financial subset with complete data
    df_financial = df[df[['budget', 'revenue']].notna().all(axis=1)].copy()
    df_financial['profit'] = df_financial['revenue'] - df_financial['budget']
    df_financial['is_profitable'] = df_financial['profit'] > 0

    # Merge back to main dataframe
    df = df.merge(
        df_financial[['title', 'release_date', 'profit', 'is_profitable']],
        on=['title', 'release_date'],
        how='left'
    )

    print(f"Financial metrics added: {df_financial['is_profitable'].sum()} profitable movies out of {len(df_financial)}")
    return df


def add_temporal_features(df):
    """
    Add decade and season features for temporal analysis
    
    Creates:
    - decade: Groups release years into decades (e.g., 1990, 2000, 2010)
    - release_season: Maps release month to season (Winter, Spring, Summer, Fall)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with release_date and release_year columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with decade and release_season columns added
    """
    # Ensure release_date is datetime
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    # Add decade
    df['decade'] = (df['release_year'] // 10) * 10

    # Add season mapping
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


def add_blockbuster_indicator(df, percentile=0.90):
    """
    Identify blockbuster movies based on revenue percentile
    
    Defines blockbusters as movies in the top 10% (by default) of revenue.
    This threshold-based approach captures the "tentpole" films that dominate
    box office performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with revenue column
    percentile : float, default=0.90
        Percentile threshold for blockbuster classification (0.90 = top 10%)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with is_blockbuster column added
    """
    # Create revenue subset with complete data
    revenue_with_data = df[df['revenue'].notna()].copy()
    blockbuster_threshold = revenue_with_data['revenue'].quantile(percentile)
    revenue_with_data['is_blockbuster'] = revenue_with_data['revenue'] >= blockbuster_threshold

    # Merge back to main dataframe
    df = df.merge(
        revenue_with_data[['title', 'release_date', 'is_blockbuster']],
        on=['title', 'release_date'],
        how='left'
    )

    print(f"Blockbuster indicator added (90th percentile threshold: ${blockbuster_threshold:,.0f})")
    print(f"Number of blockbusters identified: {revenue_with_data['is_blockbuster'].sum()}")

    return df


def engineer_features(df):
    """
    Execute complete feature engineering pipeline
    
    This is the main function that orchestrates all feature engineering
    steps in the correct order. It creates:
    1. Financial metrics (profit, profitability)
    2. Temporal features (decade, season)
    3. Blockbuster indicators (90th percentile classification)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe from data_cleaning module
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features added
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    print("\n1. Adding financial metrics...")
    df = add_financial_metrics(df)

    print("\n2. Adding temporal features...")
    df = add_temporal_features(df)

    print("\n3. Adding blockbuster indicator...")
    df = add_blockbuster_indicator(df)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Final shape: {df.shape}")
    print(f"New columns added: profit, is_profitable, decade, release_month, release_season, is_blockbuster")
    print("="*80 + "\n")

    return df