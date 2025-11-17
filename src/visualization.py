"""
Visualization module for movie dataset

This module containd functions to create plots and charts
for the blockbuster analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_budget_vs_revenue(df, save_path = None):
    """Create scatter plot of budget vs revenue with trend line"""
    plt.close('all')
    fig = plt.figure(figsize = (12, 8))

    clean_data = df[['budget', 'revenue']].dropna()

    plt.scatter(clean_data['budget'], clean_data['revenue'],
                alpha = 0.5, color = 'steelblue', label = 'Movies')
    
    # Calculate the trend line
    log_budget = np.log10(clean_data['budget'])
    log_revenue = np.log10(clean_data['revenue'])
    z = np.polyfit(log_budget, log_revenue, 1)
    p = np.poly1d(z)

    budget_range = np.logspace(np.log10(clean_data['budget'].min()),
                               np.log10(clean_data['budget'].max()), 100)
    trend_revenue = 10 ** p(np.log10(budget_range))

    plt.plot(budget_range, trend_revenue, color = 'red', linewidth = 2,
             linestyle = '--', label = f"Trend (slope = {z[0]:.2f})")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Budget ($)', fontsize = 12)
    plt.ylabel('Revenue ($)', fontsize = 12)
    plt.title("Budget vs Revenue: Is there a formula for success?",
              fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize = 10)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

    return fig

def plot_profitability_pie(df, save_path = None):
    """Create a pie chart of profitable vs unprofitable films"""
    plt.close("all")
    fig = plt.figure(figsize = (10, 8))

    clean_profit_data = df[['budget', 'profit']].dropna()
    profitability_counts = clean_profit_data['profit'].apply(
        lambda x: 'Profitable' if x > 0 else 'Unprofitable'
    ).value_counts()

    colors = ['lightblue', 'lightcoral']
    explode = (0.05, 0)

    plt.pie(profitability_counts, labels = profitability_counts.index,
            autopct = '%1.1f%%', startangle = 90, colors = colors,
            explode = explode, textprops = {'fontsize': 12, 'fontweight': 'bold'})

    plt.title ("Film Profitability Distribution",
               fontsize = 14, fontweight = 'bold', pad = 20)
    
    total = profitability_counts.sum()
    profitable_count = profitability_counts.get('Profitable', 0)
    unprofitable_count = profitability_counts.get('Unprofitable', 0)

    plt.text(0, -1.3, f"Total films: {total}\nProfitable: {profitable_count} | Unprofitable: {unprofitable_count}",
             ha = 'center', fontsize = 12, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

    return fig

def plot_budget_vs_profit(df, save_path = None):
    """Create dual plot of budget vs profit analysis"""
    plt.close('all')

    clean_profit_data = df[['budget', 'profit']].dropna()
    profitable = clean_profit_data[clean_profit_data['profit'] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

    # LEFT PLOT: Scatter plot
    ax1.scatter(profitable['budget'], profitable['profit'],
                alpha = 0.5, color = 'seagreen', label = 'Profitable Movies', s = 20)
    
    # Add trend line
    log_budget_p = np.log10(profitable['budget'])
    log_profit = np.log10(profitable['profit'])
    z_profit = np.polyfit(log_budget_p, log_profit, 1)
    p_profit = np.poly1d(z_profit)

    budget_range_p = np.logspace(np.log10(profitable['budget'].min()),
                                 np.log10(profitable['budget'].max()), 100)
    
    trend_proft = 10 ** p_profit(np.log10(budget_range_p))

    ax1.plot(budget_range_p, trend_proft, color = 'red', linewidth = 2,
             linestyle = '--', label = f"Trend (slope = {z_profit[0]:.2f})")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Budget ($)', fontsize = 12)
    ax1.set_ylabel('Profit ($)', fontsize = 12)
    ax1.set_title('Budget vs Profit: Investment Returns in Filmmaking Industry?',
                  fontsize = 14, fontweight = 'bold')
    ax1.legend(fontsize = 10, loc = 'upper left')
    ax1.grid(True, alpha = 0.3)
    ax1.set_ylim(1e3, 1e9)

    # RIGHT PLOT: Bar plot
    budget_categories = []
    profit_values = []

    for _, row in clean_profit_data.iterrows():
        if row['budget'] < 1e6:
            budget_categories.append('Low\n(<$1M)')
            profit_values.append(row['profit'])
        elif row['budget'] < 5e7:
            budget_categories.append('Mid\n($1M - $50M)')
            profit_values.append(row['profit'])
        else:
            budget_categories.append('High\n(>$50M)')
            profit_values.append(row['profit'])

    profit_df = pd.DataFrame({'Budget Category': budget_categories, 'Profit ($)': profit_values})

    category_order = ['Low\n(<$1M)', 'Mid\n($1M - $50M)', 'High\n(>$50M)']
    avg_profit_by_category = profit_df.groupby('Budget Category')['Profit ($)'].mean() / 1e6
    avg_profit_by_category = avg_profit_by_category.reindex(category_order)

    colors = ['lightcoral', 'lightblue', 'lightgreen']
    bars = ax2.bar(category_order, avg_profit_by_category,
                   color = colors, alpha = 0.8, 
                   edgecolor = 'black', linewidth = 1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f"${height:.1f}M",
                 ha = 'center', va ='bottom', fontsize = 11, fontweight = 'bold')
        
    ax2.set_ylabel('Average Profit ($ Millions)', fontsize = 12)
    ax2.set_xlabel('Budget Category', fontsize = 12)
    ax2.set_title('Average Profit by Budget Category', fontsize = 14, fontweight = 'bold')
    ax2.grid(True, alpha = 0.3, axis = 'y')

    plt.tight_layout(pad = 2.0)

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

    return fig

def plot_rating_vs_revenue(df, save_path = None):
    """Create rating vs revenue analysis plots"""
    plt.colse('all')

    rating_revenue = df[['vote_average', 'revenue']].dropna()

    # Identify blockbusters
    Q1 = rating_revenue['revenue'].quantile(0.25)
    Q3 = rating_revenue['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    blockbuster_threshold = Q3 + 1.5 * IQR
    rating_revenue['is_blockbuster'] = rating_revenue['revenue'] > blockbuster_threshold

    def categorize_ratings(rating):
        if rating < 5.0:
            return 'Poor\n(<5.0)'
        elif rating < 6.5:
            return 'Average\n(5.0 - 6.5)'
        elif rating < 7.5:
            return 'Good\n(6.5 - 7.5)'
        else:
            return 'Excellent\n(>= 7.5)'
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

    # LEFT PLOT: Scatter plot
    ax1.scatter(rating_revenue['vote_average'],
                rating_revenue['revenue'],
                alpha = 0.5, color = 'steelblue', s = 50,
                edgecolors = 'white', linewidth = 1, label = 'All movies')
    
    blockbusters = rating_revenue[rating_revenue['is_blockbuster']]
    ax1.scatter(blockbusters['vote_average'],
                blockbusters['revenue'],
                alpha = 0.5, color = 'red', s = 50,
                edgecolors = 'darkred', linewidth = 1,
                marker = '*', label = f"Blockbuster (n= {len(blockbusters)})")
    
    # Add trend line
    z = np.polyfit(rating_revenue['vote_average'],
                   np.log10(rating_revenue['revenue']), 1)
    p_all = np.poly1d(z)

    vote_range = np.linspace(0, 10, 100)
    trend = 10 ** p_all(vote_range)

    ax1.plot(vote_range, trend, 
             color = 'darkgreen', linewidth = 2.5, linestyle = '--',
             label = f"Trend (slope = {z[0]:.2f})")
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Vote Average (rating)', fontsize = 12, fontweight = 'bold')
    ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
    ax1.set_title("Rating vs Revenue: Does Quality drive Box Office success?",
                  fontsize = 13, fontweight = 'bold')
    ax1.set_xlim(0, 10)
    ax1.legend(fontsize = 10, loc = 'lower right')
    ax1.grid(True, alpha = 0.3)

    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])
    ax1.text(0.05, 0.95, f"Correlation: {correlation:.3f}\nn = {len(rating_revenue)}",
             transform = ax1.transAxes, fontsize = 11,
             verticalalignment = 'top',
             bbox = dict(boxstyle = 'round', facecolor = 'lightblue', alpha = 0.8))
    

    # RIGHT PLOT: Bar plot
    rating_revenue['category'] = rating_revenue['vote_average'].apply(categorize_ratings)
    category_order = ['Poor\n(<5.0)', 'Average\n(5.0 - 6.5)', 'Good\n(6.5 - 7.5)', 'Excellent\n(>= 7.5)']
    
    avg_revenue_all = rating_revenue.groupby('category')['revenue'].mean() / 1e6
    avg_revenue_all = avg_revenue_all.reindex(category_order)
    
    colors = ['red', 'orange', 'yellow', 'green']
    bars = ax2.bar(category_order, avg_revenue_all, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'${height:.1f}M',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Rating Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Revenue by Rating Category', fontsize=13, fontweight='bold')
    
    fig.suptitle('Rating vs Revenue (Data with Outliers)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_genre_analysis(df, save_path=None):
    """Create horizontal bar chart of top genres by revenue."""
    plt.close('all')
    fig = plt.figure(figsize=(12, 8))
    
    genre_data = df[['primary_genre', 'revenue']].dropna()
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count']
    }).reset_index()
    
    genre_stats.columns = ['genre', 'avg_revenue', 'count']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    genre_stats = genre_stats[genre_stats['count'] >= 50]
    genre_stats = genre_stats.sort_values('avg_revenue', ascending=False).head(10)
    
    bars = plt.barh(range(len(genre_stats)), genre_stats['avg_revenue'],
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars[0].set_color('red')
    bars[0].set_edgecolor('darkred')
    bars[0].set_linewidth(2.5)
    
    plt.yticks(range(len(genre_stats)), genre_stats['genre'], fontsize=11)
    
    for i, (idx, row) in enumerate(genre_stats.iterrows()):
        plt.text(row['avg_revenue'] + 5, i, f"${row['avg_revenue']:.1f}M",
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    plt.ylabel('Genre', fontsize=12, fontweight='bold')
    plt.title('Top 10 Genres by Average Revenue', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_all_plots(df, output_dir='outputs'):
    """Generate all visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("Creating Budget vs Revenue plot...")
    plot_budget_vs_revenue(df, f"{output_dir}/budget_vs_revenue.png")
    
    print("Creating Profitability pie chart...")
    plot_profitability_pie(df, f"{output_dir}/profitability_pie.png")
    
    print("Creating Budget vs Profit plot...")
    plot_budget_vs_profit(df, f"{output_dir}/budget_vs_profit.png")
    
    print("Creating Rating vs Revenue plot...")
    plot_rating_vs_revenue(df, f"{output_dir}/rating_vs_revenue.png")
    
    print("Creating Genre analysis plot...")
    plot_genre_analysis(df, f"{output_dir}/genre_analysis.png")
    
    print("Visualization generation complete!")
    print(f"Plots saved in '{output_dir}/' directory")

