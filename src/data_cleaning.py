"""
Data cleaning module for movie dataset analysis

This module contains functions to load, clean, and prepare the movie
metadata dataset for the analysis that will be conducted about the 
Blockbuster formula.

"""
import pandas as pd
import numpy as np
from ast import literal_eval


def load_data(filepath):
    """
    Load the metadata from the CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing movie metadata
    
    Returns:
    --------
    pd.DataFrame
        Raw movie metadata dataframe
    """
    return pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')


def clean_column_names(df):
    """
    Strip whitespace from column names
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potentially messy column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned column names
    """
    df.columns = df.columns.str.strip()
    return df


def drop_irrelevant_columns(df):
    """
    Remove columns not needed for blockbuster analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with only relevant columns retained
    """
    columns_to_drop = [
        'adult', 'belongs_to_collection', 'homepage', 'poster_path',
        'tagline', 'video', 'spoken_languages', 'backdrop_path', 
        'imdb_id', 'original_title', 'overview', 'status'
    ]
    return df.drop(columns=columns_to_drop, errors='ignore')


def clean_dates_and_years(df):
    """
    Parse release dates and filter unrealistic years
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw date strings
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with parsed dates and filtered years (1900-2025)
    """
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['title', 'release_date'])
    df['release_year'] = df['release_date'].dt.year
    # Make a copy here to avoid SettingWithCopyWarning on subsequent operations
    df = df[(df['release_year'] >= 1900) & (df['release_year'] <= 2025)].copy()
    return df


def convert_numeric_columns(df):
    """
    Convert specified columns to numeric type
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns to convert
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with numeric columns properly typed
    """
    numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average',
                       'vote_count', 'popularity']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies
    
    Strategy:
    - Runtime: Replace zeros with NaN, impute with median
    - Budget/Revenue: Replace zeros with NaN, keep NaN for flexibility
    - Filter extreme values for data quality
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values properly handled
    """
    # Replace runtime zeros with NaN
    df['runtime'] = df['runtime'].replace(0, np.nan)

    # Impute safe columns with median
    safe_impute = ['runtime', 'popularity', 'vote_average', 'vote_count']
    for col in safe_impute:
        df[col] = df[col].fillna(df[col].median())

    # Filter extreme runtime values (only reasonable movie lengths: 40-300 minutes)
    df = df[(df['runtime'] >= 40) & (df['runtime'] <= 300)].copy()

    # Replace budget and revenue zeros with NaN
    df[['budget', 'revenue']] = df[['budget', 'revenue']].replace(0, np.nan)

    # Remove unrealistic values (< $1000)
    df = df[(df['budget'].isna()) | (df['budget'] >= 1000)]
    df = df[(df['revenue'].isna()) | (df['revenue'] >= 1000)]

    return df


def remove_duplicates(df):
    """
    Remove duplicate entries from the dataset
    
    First removes exact duplicates, then removes duplicates based on
    title and release_date combination (for remakes/re-releases)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe potentially containing duplicates
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    df = df.drop_duplicates(keep='first')
    df = df.drop_duplicates(subset=['title', 'release_date'], keep='first')
    return df


def filter_vote_count(df, min_votes=100):
    """
    Filter movies by minimum vote count for reliable ratings
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with vote_count column
    min_votes : int, default=100
        Minimum number of votes required
    
    Returns:
    --------
    pd.DataFrame
        Dataframe containing only movies with sufficient votes
    """
    return df[df['vote_count'] >= min_votes].copy()


def parse_json_columns(df): 
    """
    Parse JSON-formatted columns into Python lists
    
    Handles columns that contain JSON-like strings representing lists
    of dictionaries (e.g., genres, production companies)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with JSON-formatted string columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with JSON columns parsed into Python lists
    """
    json_columns = ['genres', 'production_companies', 'production_countries']

    for col in json_columns:
        if col in df.columns:
            s = df[col].fillna('[]').astype(str)
            looks_like_list = s.str.strip().str.startswith('[') & s.str.strip().str.endswith(']')
            s = s.where(looks_like_list, '[]')
            parsed = s.apply(lambda x: literal_eval(x) if x.strip().startswith('[') else [])
            df[col] = parsed.apply(lambda v: v if isinstance(v, list) else [])

    return df 


def extract_names_from_json(value):
    """
    Extract names from JSON list of dictionaries
    
    Helper function to extract 'name' fields from list of dicts
    and return as comma-separated string
    
    Parameters:
    -----------
    value : list or other
        List of dictionaries containing 'name' keys
    
    Returns:
    --------
    str or pd.NA
        Comma-separated string of names, or pd.NA if empty
    """
    if not isinstance(value, list) or len(value) == 0:
        return pd.NA
    names = [str(item.get('name', '')) for item in value
             if isinstance(item, dict) and 'name' in item]
    
    return ', '.join(names) if names else pd.NA


def create_readable_columns(df):
    """
    Create human-readable string columns from JSON data
    
    Extracts names from JSON columns and creates:
    - genres_str: Comma-separated genre names
    - companies_str: Comma-separated production company names
    - countries_str: Comma-separated production country names
    - primary_genre: First genre listed (for categorical analysis)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with parsed JSON columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with readable string columns and JSON columns removed
    """
    df['genres_str'] = df['genres'].apply(extract_names_from_json)
    df['companies_str'] = df['production_companies'].apply(extract_names_from_json)
    df['countries_str'] = df['production_countries'].apply(extract_names_from_json)
    df['primary_genre'] = df['genres_str'].apply(
        lambda x: x.split(', ')[0] if pd.notna(x) else 'Unknown'
    )

    # Drop original JSON columns (keeping only readable string versions)
    json_columns = ['genres', 'production_companies', 'production_countries']
    df = df.drop(columns=json_columns, errors='ignore')

    return df


def clean_data(filepath):
    """
    Execute complete data cleaning pipeline
    
    This is the main function that orchestrates all cleaning steps
    in the correct order to produce a cleaned dataset ready for
    feature engineering and analysis.
    
    Pipeline steps:
    1. Load raw data
    2. Clean column names
    3. Drop irrelevant columns
    4. Clean dates and filter years
    5. Convert numeric columns
    6. Handle missing values
    7. Remove duplicates
    8. Filter by vote count
    9. Parse JSON columns
    10. Create readable columns
    
    Parameters:
    -----------
    filepath : str
        Path to the raw movie metadata CSV file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering
    """
    print("Loading data...")
    df = load_data(filepath)

    print("Cleaning column names...")
    df = clean_column_names(df)

    print("Dropping irrelevant columns...")
    df = drop_irrelevant_columns(df)

    print("Cleaning dates and years...")
    df = clean_dates_and_years(df)

    print("Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Removing duplicates...")
    df = remove_duplicates(df)

    print("Filtering by vote count...")
    df = filter_vote_count(df)

    print("Parsing JSON columns...")
    df = parse_json_columns(df)

    print("Creating readable columns...")
    df = create_readable_columns(df)

    # Diagnostic output
    print(f"\n{'='*80}")
    print("DATA CLEANING COMPLETE - DIAGNOSTIC CHECK")
    print(f"{'='*80}")
    print(f"Final shape: {df.shape}")
    print(f"Budget missing: {df['budget'].isna().sum()}")
    print(f"Revenue missing: {df['revenue'].isna().sum()}")
    print(f"Both budget AND revenue present: {df[['budget', 'revenue']].notna().all(axis=1).sum()}")
    print(f"Columns present: {df.columns.tolist()}")
    print(f"{'='*80}\n")

    return df