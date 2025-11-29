"""
Visualization module for movie dataset

This module contains functions to create plots and charts
for the blockbuster analysis

Author: Legolas
Date: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


def plot_distribution_histograms(df, save_path=None):
    """
    Create histograms for key variable distributions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with distribution variables
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    dist_variables = ['budget', 'revenue', 'profit', 'runtime',
                      'popularity', 'vote_average', 'vote_count',
                      'release_year']
    
    n = len(dist_variables)
    cols = 3
    rows = math.ceil(n/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*4))
    axes = axes.flatten()
    
    for i, col in enumerate(dist_variables):
        ax = axes[i]
        series = df[col].dropna()
        
        log_transform = series.min() > 0 and series.skew() > 1.2
        
        if log_transform:
            sns.histplot(series, kde=True, ax=ax, bins=50, log_scale=10)
            ax.set_title(f"{col} (log scale)", fontsize=12, fontweight='bold')
        else:
            sns.histplot(series, kde=True, ax=ax, bins=50)
            ax.set_title(col, fontsize=12, fontweight='bold')
        
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
    
    for j in range(len(dist_variables), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_distribution_density(df, save_path=None):
    """
    Create density plots for key variable distributions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with distribution variables
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    dist_variables = ['budget', 'revenue', 'profit', 'runtime',
                      'popularity', 'vote_average', 'vote_count',
                      'release_year']
    
    n = len(dist_variables)
    cols = 3
    rows = math.ceil(n/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*4))
    axes = axes.flatten()
    
    for i, col in enumerate(dist_variables):
        ax = axes[i]
        series = df[col].dropna()
        
        log_transform = series.min() > 0 and series.skew() > 1.2
        
        if log_transform:
            log_series = np.log10(series)
            sns.kdeplot(log_series, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
            ax.set_title(f"{col} Density (log scale)", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"log10({col})", fontsize=10)
            
            median_val = np.log10(series.median())
            ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
        else:
            sns.kdeplot(series, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
            ax.set_title(f"{col} Density", fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            
            median_val = series.median()
            ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
        
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_budget_vs_revenue(df, save_path=None):
    """
    Create scatter plot of budget vs revenue with trend line
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    fig = plt.figure(figsize=(12, 8))
    
    clean_data = df[['budget', 'revenue']].dropna()
    
    plt.scatter(clean_data['budget'], clean_data['revenue'], 
                alpha=0.5, color='steelblue', label='Movies')
    
    log_budget = np.log10(clean_data['budget'])
    log_revenue = np.log10(clean_data['revenue'])
    z = np.polyfit(log_budget, log_revenue, 1)
    p = np.poly1d(z)
    
    budget_range = np.logspace(np.log10(clean_data['budget'].min()),
                               np.log10(clean_data['budget'].max()), 100)
    trend_revenue = 10 ** p(np.log10(budget_range))
    
    plt.plot(budget_range, trend_revenue, color='red', linewidth=2, 
             linestyle='--', label=f'Trend (slope = {z[0]:.2f})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Budget ($)', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.title('Budget vs Revenue: Is There a Formula for Success?', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_profitability_pie(df, save_path=None):
    """
    Create pie chart of profitable vs unprofitable films
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and profit columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    
    clean_profit_data = df[['budget', 'profit']].dropna()
    profitability_counts = clean_profit_data['profit'].apply(
        lambda x: 'Profitable' if x > 0 else 'Unprofitable'
    ).value_counts()
    
    colors = ['lightblue', 'lightcoral']
    explode = (0.05, 0)
    
    plt.pie(profitability_counts, labels=profitability_counts.index,
            autopct='%1.1f%%', startangle=90, colors=colors,
            explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    plt.title('Film Profitability Distribution', 
              fontsize=14, fontweight='bold', pad=20)
    
    total = profitability_counts.sum()
    profitable_count = profitability_counts.get('Profitable', 0)
    unprofitable_count = profitability_counts.get('Unprofitable', 0)
    
    plt.text(0, -1.3, 
             f"Total films: {total}\nProfitable: {profitable_count} | Unprofitable: {unprofitable_count}",
             ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_budget_vs_profit(df, save_path=None):
    """
    Create dual plot of budget vs profit analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with budget and profit columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    clean_profit_data = df[['budget', 'profit']].dropna()
    profitable = clean_profit_data[clean_profit_data['profit'] > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.scatter(profitable['budget'], profitable['profit'],
                alpha=0.5, color='seagreen', label='Profitable Movies', s=20)
    
    log_budget_p = np.log10(profitable['budget'])
    log_profit = np.log10(profitable['profit'])
    z_profit = np.polyfit(log_budget_p, log_profit, 1)
    p_profit = np.poly1d(z_profit)
    
    budget_range_p = np.logspace(np.log10(profitable['budget'].min()),
                                 np.log10(profitable['budget'].max()), 100)
    trend_profit = 10 ** p_profit(np.log10(budget_range_p))
    
    ax1.plot(budget_range_p, trend_profit, color="red", linewidth=2,
             linestyle='--', label=f"Trend (slope = {z_profit[0]:.2f})")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Budget ($)', fontsize=12)
    ax1.set_ylabel('Profit ($)', fontsize=12)
    ax1.set_title('Budget vs Profit: Investment Returns',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e3, 1e9)
    
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
                   color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f"${height:.1f}M",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Average Profit ($ Millions)', fontsize=12)
    ax2.set_xlabel('Budget Category', fontsize=12)
    ax2.set_title('Average Profit by Budget Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rating_vs_revenue(df, save_path=None):
    """
    Create rating vs revenue analysis plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with vote_average and revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    rating_revenue = df[['vote_average', 'revenue']].dropna()
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.scatter(rating_revenue['vote_average'],
                rating_revenue['revenue'],
                alpha=0.5, color='steelblue', s=50,
                edgecolors='white', linewidth=1, label='All movies')
    
    blockbusters = rating_revenue[rating_revenue['is_blockbuster']]
    ax1.scatter(blockbusters['vote_average'],
                blockbusters['revenue'],
                alpha=0.5, color='red', s=50,
                edgecolors='darkred', linewidth=1,
                marker='*', label=f"Blockbuster (n = {len(blockbusters)})")
    
    z = np.polyfit(rating_revenue['vote_average'],
                   np.log10(rating_revenue['revenue']), 1)
    p_all = np.poly1d(z)
    
    vote_range = np.linspace(0, 10, 100)
    trend = 10 ** p_all(vote_range)
    
    ax1.plot(vote_range, trend,
             color='darkgreen', linewidth=2.5, linestyle='--',
             label=f"Trend (slope = {z[0]:.2f})")
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Vote Average (Rating)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Rating vs Revenue: Does Quality Drive Box Office Success?', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])
    ax1.text(0.05, 0.95, f"Correlation: {correlation:.3f}\nn = {len(rating_revenue)}",
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
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
    
    fig.suptitle('Rating vs Revenue',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_runtime_analysis(df, save_path=None):
    """
    Create runtime vs revenue analysis plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with runtime and revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    runtime_revenue = df[['runtime', 'revenue']].dropna()
    
    blockbuster_threshold = runtime_revenue['revenue'].quantile(0.90)
    runtime_revenue['is_blockbuster'] = runtime_revenue['revenue'] >= blockbuster_threshold
    
    def categorize_runtime(runtime):
        if runtime < 90:
            return 'Short\n(<90 min)'
        elif runtime < 120:
            return 'Standard\n(90-120 min)'
        elif runtime < 150:
            return 'Long\n(120-150 min)'
        else:
            return 'Epic\n(>=150 min)'
    
    runtime_revenue['runtime_category'] = runtime_revenue['runtime'].apply(categorize_runtime)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    regular_movies = runtime_revenue[~runtime_revenue['is_blockbuster']]
    blockbusters = runtime_revenue[runtime_revenue['is_blockbuster']]
    
    ax1.scatter(regular_movies['runtime'], regular_movies['revenue'],
                alpha=0.5, color='steelblue', s=50, edgecolor='white', linewidth=0.5, label='All movies')
    
    ax1.scatter(blockbusters['runtime'], blockbusters['revenue'],
                alpha=0.7, color='red', s=100, edgecolors='darkred', linewidth=1,
                marker='*', label=f"Blockbusters (n = {len(blockbusters)})")
    
    z_runtime = np.polyfit(runtime_revenue['runtime'], np.log10(runtime_revenue['revenue']), 1)
    p_runtime = np.poly1d(z_runtime)
    runtime_range = np.linspace(runtime_revenue['runtime'].min(), runtime_revenue['runtime'].max(), 100)
    trend_runtime = 10 ** p_runtime(runtime_range)
    
    ax1.plot(runtime_range, trend_runtime, color='darkgreen', linewidth=2.5, linestyle='--',
             label=f"Trend (slope = {z_runtime[0]:.3f})")
    
    ax1.axvline(x=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=120, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=150, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Scatter plot: Runtime vs Revenue', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    correlation_runtime = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])
    ax1.text(0.05, 0.05, f'Correlation: {correlation_runtime:.3f}\nn = {len(runtime_revenue)}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax1.legend(fontsize=10, loc='upper left')
    
    category_order = ['Short\n(<90 min)', 'Standard\n(90-120 min)', 'Long\n(120-150 min)', 'Epic\n(>=150 min)']
    avg_revenue_runtime = runtime_revenue.groupby('runtime_category')['revenue'].mean() / 1e6
    avg_revenue_runtime = avg_revenue_runtime.reindex(category_order)
    
    colors_runtime = ['blue', 'violet', 'purple', 'pink']
    bars = ax2.bar(category_order, avg_revenue_runtime, color=colors_runtime, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'${height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Runtime Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Bar Plot: Average Revenue by Runtime', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Runtime vs Revenue: Does Movie Length Affect Box Office Success?',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_popularity_analysis(df, save_path=None):
    """
    Create popularity vs revenue analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with popularity and revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    popularity_revenue = df[['popularity', 'revenue']].dropna()
    
    blockbuster_threshold = popularity_revenue['revenue'].quantile(0.90)
    popularity_revenue['is_blockbuster'] = popularity_revenue['revenue'] >= blockbuster_threshold
    
    def categorize_popularity(popularity):
        if popularity < 5:
            return 'Low\n(<5)'
        elif popularity < 15:
            return 'Medium\n(5-15)'
        elif popularity < 30:
            return 'High\n(15-30)'
        else:
            return 'Very High\n(>=30)'
    
    popularity_revenue['popularity_category'] = popularity_revenue['popularity'].apply(categorize_popularity)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    regular_movies = popularity_revenue[~popularity_revenue['is_blockbuster']]
    blockbusters = popularity_revenue[popularity_revenue['is_blockbuster']]
    
    ax1.scatter(regular_movies['popularity'], regular_movies['revenue'],
                alpha=0.5, color='steelblue', s=50, edgecolor='white', linewidth=0.5, label='All movies')
    
    ax1.scatter(blockbusters['popularity'], blockbusters['revenue'],
                alpha=0.7, color='orange', s=100, edgecolors='darkred', linewidth=1,
                marker='*', label=f"Blockbusters (n = {len(blockbusters)})")
    
    z_pop = np.polyfit(np.log10(popularity_revenue['popularity'] + 1),
                       np.log10(popularity_revenue['revenue']), 1)
    p_pop = np.poly1d(z_pop)
    
    pop_range = np.logspace(np.log10(popularity_revenue['popularity'].min() + 1),
                           np.log10(popularity_revenue['popularity'].max() + 1), 100)
    trend_pop = 10 ** p_pop(np.log10(pop_range))
    
    ax1.plot(pop_range, trend_pop, color='darkgreen', linewidth=2.5, linestyle='--',
             label=f"Trend (slope = {z_pop[0]:.2f})")
    
    ax1.axvline(x=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=30, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Popularity Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Scatter plot: Popularity vs Revenue', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    correlation_pop = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])
    ax1.text(0.05, 0.95, f"Correlation: {correlation_pop:.3f}\nn = {len(popularity_revenue)}",
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                       edgecolor='black', linewidth=1.5))
    
    ax1.legend(fontsize=10, loc='lower right')
    
    category_order = ['Low\n(<5)', 'Medium\n(5-15)', 'High\n(15-30)', 'Very High\n(>=30)']
    avg_revenue_pop = popularity_revenue.groupby('popularity_category')['revenue'].mean() / 1e6
    avg_revenue_pop = avg_revenue_pop.reindex(category_order)
    
    colors_pop = ['lightcoral', 'orange', 'yellow', 'green']
    bars = ax2.bar(category_order, avg_revenue_pop, color=colors_pop, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f"${height:.1f}M", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Popularity Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Bar Plot: Average Revenue by Popularity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Popularity vs Revenue: Does Buzz Equal Box Office Success?',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_temporal_trends(df, save_path=None):
    """
    Create temporal trends visualization by decade
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with decade, revenue, budget columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    decade_data = df[['decade', 'revenue', 'budget', 'vote_average', 'is_blockbuster']].dropna()
    
    decade_counts = decade_data['decade'].value_counts()
    valid_decades = decade_counts[decade_counts >= 50].index
    decade_data = decade_data[decade_data['decade'].isin(valid_decades)]
    
    decade_stats = decade_data.groupby('decade').agg({
        'revenue': ['mean', 'count'],
        'budget': 'mean',
        'vote_average': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    decade_stats.columns = ['decade', 'avg_revenue', 'count', 'avg_budget', 'avg_rating', 'blockbuster_rate']
    decade_stats['avg_revenue'] = decade_stats['avg_revenue'] / 1e6
    decade_stats['avg_budget'] = decade_stats['avg_budget'] / 1e6
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    decade_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(decade_stats)))
    
    bars1 = ax1.bar(decade_stats['decade'].astype(str), decade_stats['avg_revenue'],
                    color=decade_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f"${height:.0f}M", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Revenue ($ Millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Revenue Evolution by Decade', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    bars2 = ax2.bar(decade_stats['decade'].astype(str), decade_stats['avg_budget'],
                    color=decade_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f"${height:.0f}M", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Budget ($ Millions)', fontsize=11, fontweight='bold')
    ax2.set_title('Budget Growth by Decade', fontsize=11, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    bars3 = ax3.bar(decade_stats['decade'].astype(str), decade_stats['avg_rating'],
                    color=decade_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f"{height:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Rating', fontsize=11, fontweight='bold')
    ax3.set_title('Film Quality by Decade', fontsize=11, fontweight='bold', pad=15)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.set_ylim(5.5, 7.6)
    ax3.grid(True, alpha=0.3, axis='y')
    
    bars4 = ax4.bar(decade_stats['decade'].astype(str), decade_stats['blockbuster_rate'],
                    color=decade_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Blockbuster Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Blockbuster Concentration by Decade', fontsize=11, fontweight='bold', pad=15)
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Industry Evolution: How the Blockbuster Formula Changed Over Time',
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.45, wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_seasonal_patterns(df, save_path=None):
    """
    Create seasonal release patterns analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with release_season, revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    seasonal_data = df[['release_season', 'release_month', 'revenue', 'is_blockbuster']].dropna()
    
    season_stats = seasonal_data.groupby('release_season').agg({
        'revenue': ['mean', 'median', 'count'],
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    season_stats.columns = ['season', 'avg_revenue', 'median_revenue', 'count', 'blockbuster_rate']
    season_stats['avg_revenue'] = season_stats['avg_revenue'] / 1e6
    
    monthly_stats = seasonal_data.groupby('release_month').agg({
        'revenue': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    monthly_stats.columns = ['month', 'avg_revenue', 'blockbuster_rate']
    monthly_stats['avg_revenue'] = monthly_stats['avg_revenue'] / 1e6
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[int(x) - 1])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_colors = {'Winter': 'blue', 'Spring': 'lightgreen', 'Summer': 'orange', 'Fall': 'brown'}
    season_stats_ordered = season_stats.set_index('season').reindex(season_order).reset_index()
    
    bars1 = ax1.bar(season_stats_ordered['season'], season_stats_ordered['avg_revenue'],
                    color=[season_colors[s] for s in season_stats_ordered['season']],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f"${height:.1f}M", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Release Season', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Revenue by Release Season', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    bars2 = ax2.bar(season_stats_ordered['season'], season_stats_ordered['blockbuster_rate'],
                    color=[season_colors[s] for s in season_stats_ordered['season']],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Release Season', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Blockbuster Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Blockbuster Concentration by Season', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    bars3 = ax3.bar(monthly_stats['month_name'], monthly_stats['avg_revenue'],
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    max_month_idx = monthly_stats['avg_revenue'].idxmax()
    bars3[max_month_idx].set_color('yellow')
    bars3[max_month_idx].set_edgecolor('orange')
    bars3[max_month_idx].set_linewidth(2.5)
    
    for i, row in monthly_stats.iterrows():
        height = row['avg_revenue']
        ax3.text(i, height, f"{height:.0f}M", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_xlabel('Release Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Revenue by Month (Peak in Yellow)', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(monthly_stats['avg_revenue']) * 1.15)
    
    bars4 = ax4.bar(season_stats_ordered['season'], season_stats_ordered['count'],
                    color=[season_colors[s] for s in season_stats_ordered['season']],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f"{int(height)}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Release Season', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Films Released', fontsize=12, fontweight='bold')
    ax4.set_title('Release Volume by Season', fontsize=13, fontweight='bold')
    
    fig.suptitle('Release Timing Strategy: Does When You Release a Film Matter?',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985], h_pad=3.5, w_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_genre_analysis(df, save_path=None):
    """
    Create comprehensive 4-plot genre analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with primary_genre, revenue, popularity columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    genre_data = df[['primary_genre', 'revenue', 'popularity', 'is_blockbuster', 'vote_average']].dropna()
    
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count'],
        'popularity': 'mean',
        'vote_average': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    genre_stats.columns = ['genre', 'avg_revenue', 'count', 'avg_popularity', 'avg_rating', 'blockbuster_rate']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    
    # Filter genres with at least 50 films
    genre_stats = genre_stats[genre_stats['count'] >= 50].copy()
    genre_stats_sorted = genre_stats.sort_values('avg_revenue', ascending=False)
    
    # Create 2x2 subplot grid
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3, left=0.10, right=0.96, top=0.93, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # PLOT 1: Top 10 genres by average revenue
    top_revenue_genres = genre_stats_sorted.head(10)
    bars1 = ax1.barh(range(len(top_revenue_genres)), top_revenue_genres['avg_revenue'],
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars1[0].set_color('red')
    bars1[0].set_edgecolor('darkred')
    bars1[0].set_linewidth(2.5)
    
    ax1.set_yticks(range(len(top_revenue_genres)))
    ax1.set_yticklabels(top_revenue_genres['genre'], fontsize=11)
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                 f"${width:.1f}M", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Genres by Average Revenue', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(top_revenue_genres['avg_revenue']) * 1.15)
    
    # PLOT 2: Top 10 genres by blockbuster rate
    top_blockbuster_genres = genre_stats.nlargest(10, 'blockbuster_rate')
    bars2 = ax2.barh(range(len(top_blockbuster_genres)), top_blockbuster_genres['blockbuster_rate'],
                     color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars2[0].set_color('red')
    bars2[0].set_edgecolor('darkred')
    bars2[0].set_linewidth(2.5)
    
    ax2.set_yticks(range(len(top_blockbuster_genres)))
    ax2.set_yticklabels(top_blockbuster_genres['genre'], fontsize=11)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                 f"{width:.1f}%", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Blockbuster Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax2.set_title('Top 10 Genres by Blockbuster Rate', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(top_blockbuster_genres['blockbuster_rate']) * 1.12)
    
    # PLOT 3: Top 10 genres by average popularity
    top_popularity_genres = genre_stats.nlargest(10, 'avg_popularity')
    bars3 = ax3.barh(range(len(top_popularity_genres)), top_popularity_genres['avg_popularity'],
                     color='purple', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars3[0].set_color('darkviolet')
    bars3[0].set_edgecolor('black')
    bars3[0].set_linewidth(2.5)
    
    ax3.set_yticks(range(len(top_popularity_genres)))
    ax3.set_yticklabels(top_popularity_genres['genre'], fontsize=11)
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                 f"{width:.1f}", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Average Popularity Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax3.set_title('Top 10 Genres by Average Popularity', fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(top_popularity_genres['avg_popularity']) * 1.12)
    
    # PLOT 4: Genre success matrix (bubble chart)
    top_count_genres = genre_stats.nlargest(12, 'count')
    
    scatter = ax4.scatter(top_count_genres['avg_revenue'],
                          top_count_genres['blockbuster_rate'],
                          s=top_count_genres['count'] * 2,
                          c=top_count_genres['avg_popularity'],
                          cmap='viridis',
                          alpha=0.6,
                          edgecolors='black',
                          linewidth=1.5)
    
    for _, row in top_count_genres.iterrows():
        ax4.annotate(row['genre'],
                     (row['avg_revenue'], row['blockbuster_rate']),
                     fontsize=9,
                     ha='center',
                     va='center',
                     fontweight='bold')
    
    ax4.set_xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Blockbuster Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Genre Success Matrix (Size = Film Count, Color = Popularity)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    color_bar = plt.colorbar(scatter, ax=ax4)
    color_bar.set_label('Avg Popularity', fontsize=10, fontweight='bold')
    
    fig.suptitle('Genre Dominance: Which Types of Films Rule the Box Office?',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99], h_pad=3, w_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_production_origins(df, save_path=None):
    """
    Create production country and company analysis with 4 plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with countries_str, companies_str, revenue columns
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.close('all')
    
    country_data = df[['countries_str', 'revenue', 'is_blockbuster', 'popularity']].dropna()
    company_data = df[['companies_str', 'revenue', 'is_blockbuster', 'popularity']].dropna()
    
    # Extract primary country and company
    country_data['primary_country'] = country_data['countries_str'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
    )
    
    company_data['primary_company'] = company_data['companies_str'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
    )
    
    # Country statistics
    country_stats = country_data.groupby('primary_country').agg({
        'revenue': ['mean', 'count'],
        'popularity': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    country_stats.columns = ['country', 'avg_revenue', 'count', 'avg_popularity', 'blockbuster_rate']
    country_stats['avg_revenue'] = country_stats['avg_revenue'] / 1e6
    country_stats = country_stats[country_stats['count'] >= 100].copy()
    country_stats_sorted = country_stats.sort_values('avg_revenue', ascending=False)
    
    # Company statistics
    company_stats = company_data.groupby('primary_company').agg({
        'revenue': ['mean', 'count'],
        'popularity': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    company_stats.columns = ['company', 'avg_revenue', 'count', 'avg_popularity', 'blockbuster_rate']
    company_stats['avg_revenue'] = company_stats['avg_revenue'] / 1e6
    company_stats = company_stats[company_stats['count'] >= 20].copy()
    company_stats_sorted = company_stats.sort_values('avg_revenue', ascending=False)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 2, left=0.22, right=0.98, bottom=0.07, top=0.90, hspace=0.46, wspace=0.90)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # PLOT 1: Top countries by average revenue
    top_countries_revenue = country_stats_sorted.head(12)
    bars1 = ax1.barh(range(len(top_countries_revenue)), top_countries_revenue['avg_revenue'],
                     color='green', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars1[0].set_color('yellow')
    bars1[0].set_edgecolor('orange')
    bars1[0].set_linewidth(2.5)
    
    ax1.set_yticks(range(len(top_countries_revenue)))
    ax1.set_yticklabels(top_countries_revenue['country'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_countries_revenue.iterrows()):
        ax1.text(row['avg_revenue'] + 3, i,
                 f"${row['avg_revenue']:.1f}M", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax1.set_title('Top Countries by Average Revenue', fontsize=14, fontweight='bold', pad=20)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(top_countries_revenue['avg_revenue']) * 1.25)
    
    # PLOT 2: Top countries by blockbuster rate
    top_countries_blockbuster = country_stats.nlargest(12, 'blockbuster_rate')
    bars2 = ax2.barh(range(len(top_countries_blockbuster)), top_countries_blockbuster['blockbuster_rate'],
                     color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars2[0].set_color('red')
    bars2[0].set_edgecolor('darkred')
    bars2[0].set_linewidth(2.5)
    
    ax2.set_yticks(range(len(top_countries_blockbuster)))
    ax2.set_yticklabels(top_countries_blockbuster['country'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_countries_blockbuster.iterrows()):
        ax2.text(row['blockbuster_rate'] + 0.5, i,
                 f"{row['blockbuster_rate']:.1f}%", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Blockbuster Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax2.set_title('Top Countries by Blockbuster Rate', fontsize=14, fontweight='bold', pad=20)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(top_countries_blockbuster['blockbuster_rate']) * 1.20)
    
    # PLOT 3: Top 12 companies by average revenue
    top_companies_revenue = company_stats_sorted.head(12)
    bars3 = ax3.barh(range(len(top_companies_revenue)), top_companies_revenue['avg_revenue'],
                     color='purple', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars3[0].set_color('darkviolet')
    bars3[0].set_edgecolor('black')
    bars3[0].set_linewidth(2.5)
    
    ax3.set_yticks(range(len(top_companies_revenue)))
    ax3.set_yticklabels(top_companies_revenue['company'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_companies_revenue.iterrows()):
        ax3.text(row['avg_revenue'] + 3, i,
                 f"${row['avg_revenue']:.1f}M", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Production Company', fontsize=12, fontweight='bold')
    ax3.set_title('Top 12 Companies by Average Revenue', fontsize=14, fontweight='bold', pad=20)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(top_companies_revenue['avg_revenue']) * 1.25)
    
    # PLOT 4: Top 12 companies by blockbuster rate
    top_companies_blockbuster = company_stats.nlargest(12, 'blockbuster_rate')
    bars4 = ax4.barh(range(len(top_companies_blockbuster)), top_companies_blockbuster['blockbuster_rate'],
                     color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars4[0].set_color('darkgreen')
    bars4[0].set_edgecolor('black')
    bars4[0].set_linewidth(2.5)
    
    ax4.set_yticks(range(len(top_companies_blockbuster)))
    ax4.set_yticklabels(top_companies_blockbuster['company'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_companies_blockbuster.iterrows()):
        ax4.text(row['blockbuster_rate'] + 0.5, i,
                 f"{row['blockbuster_rate']:.1f}%", ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Blockbuster Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Production Company', fontsize=12, fontweight='bold')
    ax4.set_title('Top 12 Companies by Blockbuster Rate', fontsize=14, fontweight='bold', pad=20)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, max(top_companies_blockbuster['blockbuster_rate']) * 1.20)
    
    # Adjust spacing
    for ax in (ax1, ax2, ax3, ax4):
        ax.margins(x=0.08)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(ax.get_title(), pad=14)
        
        if ax in (ax1, ax3):
            ax.tick_params(axis='y', pad=6)
        else:
            ax.tick_params(axis='y', pad=0)
    
    fig.suptitle('Production Origins: Do Country & Company Determine Success?',
                 fontsize=18, fontweight='bold', y=0.985)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_all_plots(df, output_dir='outputs'):
    """
    Generate all visualization plots and save to directory
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fully processed dataframe with all features
    output_dir : str, default='outputs'
        Directory to save generated plots
    
    Returns:
    --------
    None
        Saves all plots to specified directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Creating distribution histograms...")
    plot_distribution_histograms(df, f"{output_dir}/distribution_histograms.png")
    
    print("2. Creating distribution density plots...")
    plot_distribution_density(df, f"{output_dir}/distribution_density.png")
    
    print("3. Creating Budget vs Revenue plot...")
    plot_budget_vs_revenue(df, f"{output_dir}/budget_vs_revenue.png")
    
    print("4. Creating Profitability pie chart...")
    plot_profitability_pie(df, f"{output_dir}/profitability_pie.png")
    
    print("5. Creating Budget vs Profit plot...")
    plot_budget_vs_profit(df, f"{output_dir}/budget_vs_profit.png")
    
    print("6. Creating Rating vs Revenue plot...")
    plot_rating_vs_revenue(df, f"{output_dir}/rating_vs_revenue.png")
    
    print("7. Creating Runtime analysis plot...")
    plot_runtime_analysis(df, f"{output_dir}/runtime_analysis.png")
    
    print("8. Creating Popularity analysis plot...")
    plot_popularity_analysis(df, f"{output_dir}/popularity_analysis.png")
    
    print("9. Creating Temporal trends plot...")
    plot_temporal_trends(df, f"{output_dir}/temporal_trends.png")
    
    print("10. Creating Seasonal patterns plot...")
    plot_seasonal_patterns(df, f"{output_dir}/seasonal_patterns.png")
    
    print("11. Creating Genre analysis plot...")
    plot_genre_analysis(df, f"{output_dir}/genre_analysis.png")
    
    print("12. Creating Production origins plot...")
    plot_production_origins(df, f"{output_dir}/production_origins.png")
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print(f"All plots saved in '{output_dir}/' directory")
    print("="*80 + "\n")
