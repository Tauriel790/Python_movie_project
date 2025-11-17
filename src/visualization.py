"""
Visualization module for movie dataset

This module contains functions to create plots and charts
for the blockbuster analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


def plot_distribution_histograms(df, save_path=None):
    """Create histograms for key variable distributions."""
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
    """Create density plots for key variable distributions."""
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
    """Create scatter plot of budget vs revenue with trend line."""
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
    """Create pie chart of profitable vs unprofitable films."""
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
    """Create dual plot of budget vs profit analysis."""
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
    """Create rating vs revenue analysis plots."""
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
    """Create runtime vs revenue analysis plots."""
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
                marker='*', label=f"Blockbusters (n={len(blockbusters)})")
    
    z_runtime = np.polyfit(runtime_revenue['runtime'], np.log10(runtime_revenue['revenue']), 1)
    p_runtime = np.poly1d(z_runtime)
    runtime_range = np.linspace(runtime_revenue['runtime'].min(), runtime_revenue['runtime'].max(), 100)
    trend_runtime = 10 ** p_runtime(runtime_range)
    
    ax1.plot(runtime_range, trend_runtime, color='darkgreen', linewidth=2.5, linestyle='--',
             label=f"Trend (slope={z_runtime[0]:.3f})")
    
    ax1.axvline(x=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=120, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=150, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Scatter plot: Runtime vs Revenue', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
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
    ax2.set_title('Average Revenue by Runtime Category', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Runtime vs Revenue: Does Movie Length Affect Box Office Success?',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_popularity_analysis(df, save_path=None):
    """Create popularity vs revenue analysis."""
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
                marker='*', label=f"Blockbusters (n={len(blockbusters)})")
    
    z_pop = np.polyfit(np.log10(popularity_revenue['popularity'] + 1),
                       np.log10(popularity_revenue['revenue']), 1)
    p_pop = np.poly1d(z_pop)
    
    pop_range = np.logspace(np.log10(popularity_revenue['popularity'].min() + 1),
                           np.log10(popularity_revenue['popularity'].max() + 1), 100)
    trend_pop = 10 ** p_pop(np.log10(pop_range))
    
    ax1.plot(pop_range, trend_pop, color='darkgreen', linewidth=2.5, linestyle='--',
             label=f"Trend (slope={z_pop[0]:.2f})")
    
    ax1.axvline(x=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=30, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Popularity Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Scatter plot: Popularity vs Revenue', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
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
    ax2.set_title('Average Revenue by Popularity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Popularity vs Revenue: Does Buzz Equal Box Office Success?',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_temporal_trends(df, save_path=None):
    """Create temporal trends visualization by decade."""
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
    ax2.set_title('Budget Growth by Decade', fontsize=11, fontweight='bold')
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
    ax3.set_title('Film Quality by Decade', fontsize=11, fontweight='bold')
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
    ax4.set_title('Blockbuster Concentration by Decade', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Industry Evolution: How the Blockbuster Formula Changed Over Time',
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.45, wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_seasonal_patterns(df, save_path=None):
    """Create seasonal release patterns analysis."""
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
    ax1.set_title('Average Revenue by Season', fontsize=13, fontweight='bold')
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
    
    fig.suptitle('Release Timing Strategy: Does When You Release Matter?',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985], h_pad=3.5, w_pad=2.5)
    
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
    
    print("Creating distribution histograms...")
    plot_distribution_histograms(df, f"{output_dir}/distribution_histograms.png")
    
    print("Creating distribution density plots...")
    plot_distribution_density(df, f"{output_dir}/distribution_density.png")
    
    print("Creating Budget vs Revenue plot...")
    plot_budget_vs_revenue(df, f"{output_dir}/budget_vs_revenue.png")
    
    print("Creating Profitability pie chart...")
    plot_profitability_pie(df, f"{output_dir}/profitability_pie.png")
    
    print("Creating Budget vs Profit plot...")
    plot_budget_vs_profit(df, f"{output_dir}/budget_vs_profit.png")
    
    print("Creating Rating vs Revenue plot...")
    plot_rating_vs_revenue(df, f"{output_dir}/rating_vs_revenue.png")
    
    print("Creating Runtime analysis plot...")
    plot_runtime_analysis(df, f"{output_dir}/runtime_analysis.png")
    
    print("Creating Popularity analysis plot...")
    plot_popularity_analysis(df, f"{output_dir}/popularity_analysis.png")
    
    print("Creating Temporal trends plot...")
    plot_temporal_trends(df, f"{output_dir}/temporal_trends.png")
    
    print("Creating Seasonal patterns plot...")
    plot_seasonal_patterns(df, f"{output_dir}/seasonal_patterns.png")
    
    print("Creating Genre analysis plot...")
    plot_genre_analysis(df, f"{output_dir}/genre_analysis.png")
    
    print("Visualization generation complete!")
    print(f"Plots saved in '{output_dir}/' directory")