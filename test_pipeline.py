"""
Test script to verify all modules work correctly.
"""

from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.analysis import print_analysis_summary
from src.visualization import generate_all_plots

def main():
    print("\n" + "="*80)
    print("TESTING MOVIE ANALYSIS PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Data Cleaning
    print("STEP 1: Data Cleaning")
    print("-" * 80)
    df = clean_data('data/movies_metadata.csv')
    print(f"Data cleaning complete! Shape: {df.shape}\n")
    
    # Step 2: Feature Engineering
    print("STEP 2: Feature Engineering")
    print("-" * 80)
    df = engineer_features(df)
    print(f"Feature engineering complete!\n")
    
    # Step 3: Save processed data
    print("STEP 3: Saving Processed Data")
    print("-" * 80)
    df.to_csv('data/movies_with_features.csv', index=False)
    print("Saved to 'data/movies_with_features.csv'\n")
    
    # Step 4: Analysis
    print("STEP 4: Statistical Analysis")
    print("-" * 80)
    print_analysis_summary(df)
    print("Analysis complete!\n")
    
    # Step 5: Visualizations
    print("STEP 5: Generating Visualizations")
    print("-" * 80)
    generate_all_plots(df, output_dir='outputs')
    print("Visualizations complete!\n")
    
    # Final summary
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nResults:")
    print("  - Processed data: data/movies_with_features.csv")
    print("  - Plots saved in: outputs/")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()