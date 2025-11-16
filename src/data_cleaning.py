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
    """
    return pd.read_csv(filepath, low_memory = False, on_bad_lines = 'skip')

def clean_column_names(df):
    """Strip whitespace from column names"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def drop_irrelevant_columns(df):
    """Remove columns not needed for analysis"""
    columns_to_drop = [
        'adult', 'belongs_to_collection', 'homepage', 'poster_path',
        'tagline', 'video', 'spoken_languages', 'backdrop_path', 
        'imdb_id', 'original_title', 'overview', 'status'
    ]
    return df.drop(columns = columns_to_drop, errors = 'ignore')

def clean_dates_and_years(df):
    """Parse release dates and filter unrealistic years"""
    df = df.copy()
    df['release_date'] = pd.to_datetime(df['release_date'], errors = 'coerce')
    df = df.dropna(subset = ['title', 'release_date'])
    df['release_year'] = df['release_date'].dt.year
    df = df[(df['release_year'] >= 1900) & (df['release_year'] <= 2025)]
    return df

def convert_numeric_columns(df):
    """Convert specified columns to numeric type"""
    df = df.copy()
    numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average',
                       'vote_count', 'popularity']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = 'coerce')
    return df

def handle_missing_values(df):
    """Handle missing values using appropriate strategies"""
    df = df.copy()

    # Replace runtime zeros with NaN
    df['runtime'] = df['runtime'].replace(0, np.nan)

    # Impute safe columns with median
    safe_impute = ['runtime', 'popularity', 'vote_average', 'vote_count']
    for col in safe_impute:
        df[col] = df[col].fillna(df[col].median())

    # Filter extreme runtime valuesà
    df = df[(df['runtime'] >= 40) & (df['runtime'] <= 300)]

    # Replace budget and revenue zeros with NaN
    df[['budget', 'revenue']] = df[['budget', 'revenue']].replace(0, np.nan)

    # Remove unrealistic values
    df = df[(df['budget'].isna()) | (df['budget'] >= 1000)]
    df = df[(df['revenue'].isna()) | (df['revenue'] >= 1000)]

    return df

def remove_duplicates(df):
    """Remove duplicate entries from the dataset"""
    df = df.drop_duplicates(keep = 'first')
    df = df.drop_duplicates(subset = ['title', 'release_date'], keep = 'first')
    return df

def filter_vote_count(df, min_votes = 100):
    """Filter movies by minimum vote count"""
    return df[df['vote_count'] >= min_votes]

def parse_json_columns(df): 
    """Parse JSON-like columns into Python lists"""
    df = df.copy()
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
    """Extract names from JSON list of dictionaries"""
    if not isinstance(value, list) or len(value) == 0:
        return pd.NA
    names = [str(item.get('name', '')) for item in value
             if isinstance(item, dict) and 'name' in item]
    
    return ', '.join(names) if names else pd.NA

def create_readable_columns(df):
    """Create human readable string columns from JSON data"""
    df = df.copy()
    df['genres_str'] = df['genres'].apply(extract_names_from_json)
    df['companies_str'] = df['production_companies'].apply(extract_names_from_json)
    df['countries_str'] = df['production_countries'].apply(extract_names_from_json)
    df['primary_genre'] = df['genres_str'].apply(
        lambda x: x.split(', ')[0] if pd.notna(x) else 'Unknown'
    )

    # Drop original JSON columns
    json_columns = ['genres', 'production_companies', 'production_countries']
    df = df.drop(columns = json_columns, errors = 'ignore')

    return df

def clean_data(filepath):
    """Execute complete data cleaning pipeline"""
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

    print(f"Data cleaning complete. Final shape: {df.shape}")
    return df