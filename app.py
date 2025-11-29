"""
Streamlit web application for Blockbuster Movie Analysis

This interactive dashboard explores factors contributing to movie box office success
through comprehensive data analysis and visualization
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Anatomy of a Blockbuster",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv('data/movies_with_features.csv')
        df['release_date'] = pd.to_datetime(df['release_date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found! Please run the analysis pipeline first")
        st.info("Run: `python test_pipeline.py`")
        st.stop()

# Load data
df = load_data()

# Sidebar
st.sidebar.markdown("# 🎬 Navigation")
page = st.sidebar.radio(
    "Select Analysis:",
    [
        "Overview",
        "Financial Analysis",
        "Profitability",
        "Rating Analysis",
        "Runtime Analysis",
        "Popularity Analysis",
        "Temporal Trends",
        "Seasonal Patterns",
        "Genre Analysis",
        "Production Origins",
        "Conclusions"
    ]
)

# Main content
if page == "Overview":
    st.markdown("# 🎬 Anatomy of a Blockbuster")
    st.markdown("### Understanding the Factors Behind Box Office Success")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", f"{len(df):,}")
    with col2:
        st.metric("Time Span", f"{int(df['release_year'].min())} - {int(df['release_year'].max())}")
    with col3:
        profitable = df['is_profitable'].sum()
        st.metric("Profitable Films", f"{profitable:,}")
    with col4:
        blockbusters = df['is_blockbuster'].sum()
        st.metric("Blockbusters", f"{blockbusters:,}")
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Financial Data")
        films_with_budget = df['budget'].notna().sum()
        films_with_revenue = df['revenue'].notna().sum()
        st.write(f"- Films with budget data: **{films_with_budget:,}**")
        st.write(f"- Films with revenue data: **{films_with_revenue:,}**")
        st.write(f"- Average budget: **${df['budget'].mean()/1e6:.1f}M**")
        st.write(f"- Average revenue: **${df['revenue'].mean()/1e6:.1f}M**")
    
    with col2:
        st.markdown("#### Quality Metrics")
        st.write(f"- Average rating: **{df['vote_average'].mean():.2f}/10**")
        st.write(f"- Average runtime: **{df['runtime'].mean():.0f} minutes**")
        st.write(f"- Unique genres: **{df['primary_genre'].nunique()}**")
        st.write(f"- Movies with 100+ votes: **{len(df):,}**")
    
    st.markdown("---")
    
    st.markdown("### Project Goals")
    st.write("""
    This comprehensive analysis explores nine major components:
    
    1. **Budget vs Revenue** - Does spending more guarantee higher revenue?
    2. **Profitability Patterns** - What drives profit margins?
    3. **Rating Correlations** - Does quality equal box office success?
    4. **Runtime Analysis** - Is there an optimal movie length?
    5. **Popularity Trends** - Does buzz translate to revenue?
    6. **Temporal Analysis** - How has the industry evolved?
    7. **Seasonal Timing** - When should films be released?
    8. **Genre Performance** - Which types dominate?
    9. **Production Origins** - Do country and company matter?
    """)

    st.markdown("### 📈 Data Distributions")
    st.write("Understanding the distribution of key variables in our dataset:")
    
    # Distribution Histograms
    st.markdown("#### Distribution Histograms")
    
    dist_variables = ['budget', 'revenue', 'profit', 'runtime',
                      'popularity', 'vote_average', 'vote_count',
                      'release_year']
    
    n = len(dist_variables)
    cols = 3
    rows = 3  # 3x3 grid for 8 variables
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col_name in enumerate(dist_variables):
        ax = axes[i]
        series = df[col_name].dropna()
        
        # Check if log scale is needed (highly skewed data)
        log_transform = series.min() > 0 and series.skew() > 1.2
        
        if log_transform:
            sns.histplot(series, kde=True, ax=ax, bins=50, log_scale=10, color='steelblue', alpha=0.7)
            ax.set_title(f"{col_name} (log scale)", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"{col_name} (log scale)", fontsize=10)
        else:
            sns.histplot(series, kde=True, ax=ax, bins=50, color='steelblue', alpha=0.7)
            ax.set_title(col_name, fontsize=12, fontweight='bold')
            ax.set_xlabel(col_name, fontsize=10)
        
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[8].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Distribution Density Plots
    st.markdown("#### Distribution Density Plots")
    
    fig2, axes2 = plt.subplots(rows, cols, figsize=(18, 12))
    axes2 = axes2.flatten()
    
    for i, col_name in enumerate(dist_variables):
        ax = axes2[i]
        series = df[col_name].dropna()
        
        # Check if log scale is needed
        log_transform = series.min() > 0 and series.skew() > 1.2
        
        if log_transform:
            log_series = np.log10(series)
            sns.kdeplot(log_series, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
            ax.set_title(f"{col_name} Density (log scale)", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"log10({col_name})", fontsize=10)
            
            median_val = np.log10(series.median())
            ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
        else:
            sns.kdeplot(series, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
            ax.set_title(f"{col_name} Density", fontsize=12, fontweight='bold')
            ax.set_xlabel(col_name, fontsize=10)
            
            median_val = series.median()
            ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
        
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes2[8].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    st.pyplot(fig2)
    plt.close()
    
    # Key observations
    st.info(r"""
    **Key Observations:**
    
    - **Budget and Revenue** show right-skewed distributions on log scale - most films have modest budgets/revenues, but a few mega-productions create a long right tail
    - **Profit** shows left-skewed distribution - most films are profitable with modest gains, but some massive hits create extreme positive outliers, while losses cluster near zero
    - **Runtime** shows roughly normal distribution clustering around 90-120 minutes, following industry standards for theatrical releases
    - **Popularity** is extremely right-skewed on log scale - the vast majority of films have low to moderate buzz, while a small number of releases dominate public attention
    - **Rating (vote_average)** shows a relatively normal distribution centered around 6.4, with few extreme failures (below 4) or masterpieces (above 8)
    - **Vote Count** shows exponential decay on log scale - most films receive relatively few ratings, while blockbusters accumulate thousands or tens of thousands of votes
    - **Release Year** distribution shows exponential growth in film production from 1915 to 2017, with dramatic acceleration from the 1980s onward
    - The gap between mean and median in financial variables (budget, revenue, profit) indicates strong influence of blockbuster outliers on averages
    """)


elif page == "Financial Analysis":
    st.markdown("## Budget vs Revenue Analysis")
    st.markdown("### Does Spending More Guarantee Higher Revenue?")
    
    clean_data = df[['budget', 'revenue']].dropna()
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(clean_data['budget'], clean_data['revenue'],
               alpha=0.5, color='steelblue', label='Movies')
    
    # Trend line
    log_budget = np.log10(clean_data['budget'])
    log_revenue = np.log10(clean_data['revenue'])
    z = np.polyfit(log_budget, log_revenue, 1)
    p = np.poly1d(z)
    
    budget_range = np.logspace(np.log10(clean_data['budget'].min()),
                               np.log10(clean_data['budget'].max()), 100)
    trend_revenue = 10 ** p(np.log10(budget_range))
    
    ax.plot(budget_range, trend_revenue, color='red', linewidth=2,
            linestyle='--', label=f'Trend (slope = {z[0]:.2f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Budget ($)', fontsize=12)
    ax.set_ylabel('Revenue ($)', fontsize=12)
    ax.set_title('Budget vs Revenue: Is There a Formula for Success?',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.markdown("### 📊 Key Statistics by Budget Category")
    
    # Calculate statistics
    correlation = clean_data['budget'].corr(clean_data['revenue'])
    low_budget = clean_data[clean_data['budget'] < 1e6]
    mid_budget = clean_data[(clean_data['budget'] >= 1e6) & (clean_data['budget'] < 5e7)]
    high_budget = clean_data[clean_data['budget'] >= 5e7]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Correlation", f"{correlation:.3f}")
    
    with col2:
        st.metric("Low Budget (<$1M)",
                  f"{len(low_budget)} films",
                  f"${low_budget['revenue'].mean()/1e6:.1f}M avg")
    
    with col3:
        st.metric("Mid Budget ($1M-$50M)",
                  f"{len(mid_budget)} films",
                  f"${mid_budget['revenue'].mean()/1e6:.1f}M avg")
    
    with col4:
        st.metric("High Budget (>$50M)",
                  f"{len(high_budget)} films",
                  f"${high_budget['revenue'].mean()/1e6:.1f}M avg")
    
    st.markdown("---")
    
    st.success(r"""
    **Key Insights:**
    
    - **Strong positive correlation** (0.71) between budget and revenue
    - **Diminishing returns**: Doubling budget doesn't double revenue (slope = 0.78)
    - **High-budget films** average \$273M revenue vs \$20M for low-budget
    - **Wide scatter** indicates budget alone doesn't guarantee success
    - Other factors like quality, timing, and audience appeal matter equally
    """)

elif page == "Profitability":
    st.markdown("## Profitability Analysis")
    st.markdown("### Does Budget Really Influence Success?")
    
    clean_profit_data = df[['budget', 'profit']].dropna()
    profitable = clean_profit_data[clean_profit_data['profit'] > 0]
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Profitability Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        profitability_counts = clean_profit_data['profit'].apply(
            lambda x: 'Profitable' if x > 0 else 'Unprofitable'
        ).value_counts()
        
        colors = ['lightblue', 'lightcoral']
        explode = (0.05, 0)
        
        ax.pie(profitability_counts, labels=profitability_counts.index,
               autopct='%1.1f%%', startangle=90, colors=colors,
               explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title('Film Profitability Distribution',
                     fontsize=14, fontweight='bold', pad=20)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Budget vs Profit")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(profitable['budget'], profitable['profit'],
                   alpha=0.5, color='seagreen', label='Profitable Movies', s=20)
        
        # Trend line
        log_budget_p = np.log10(profitable['budget'])
        log_profit = np.log10(profitable['profit'])
        z_profit = np.polyfit(log_budget_p, log_profit, 1)
        p_profit = np.poly1d(z_profit)
        
        budget_range_p = np.logspace(np.log10(profitable['budget'].min()),
                                     np.log10(profitable['budget'].max()), 100)
        trend_profit = 10 ** p_profit(np.log10(budget_range_p))
        
        ax.plot(budget_range_p, trend_profit, color="red", linewidth=2,
                linestyle='--', label=f"Trend (slope = {z_profit[0]:.2f})")
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Budget ($)', fontsize=12)
        ax.set_ylabel('Profit ($)', fontsize=12)
        ax.set_title('Budget vs Profit: Investment Returns',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e3, 1e9)
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Average profit by category
    st.markdown("### Average Profit by Budget Category")
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    bars = ax.bar(category_order, avg_profit_by_category,
                  color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"${height:.1f}M",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Profit ($ Millions)', fontsize=12)
    ax.set_xlabel('Budget Category', fontsize=12)
    ax.set_title('Average Profit by Budget Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Statistics
    total = len(clean_profit_data)
    profitable_count = (clean_profit_data['profit'] > 0).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Films", f"{total:,}")
    with col2:
        st.metric("Profitable Films",
                  f"{profitable_count:,}",
                  f"{100*profitable_count/total:.1f}%")
    with col3:
        st.metric("Average Profit", f"${clean_profit_data['profit'].mean()/1e6:.1f}M")
    
    st.info(r"""
    **Key Findings:**
    
    - 78.3% of films with complete financial data are profitable
    - Higher budgets increase profit with diminishing returns (slope = 0.60)
    - High-budget films average \$177.9M profit compared to \$19.7M for low-budget films
    - Wide scatter shows budget alone doesn't guarantee success
    """)

elif page == "Rating Analysis":
    st.markdown("## Rating vs Revenue Analysis")
    st.markdown("### Do Better-Rated Movies Make More Money?")
    
    rating_revenue = df[['vote_average', 'revenue']].dropna()
    
    # Calculate blockbuster threshold using IQR method
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Scatter Plot: Rating vs Revenue")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(rating_revenue['vote_average'], rating_revenue['revenue'],
                   alpha=0.5, color='steelblue', s=50,
                   edgecolors='white', linewidth=1, label='All movies')
        
        blockbusters = rating_revenue[rating_revenue['is_blockbuster']]
        ax.scatter(blockbusters['vote_average'], blockbusters['revenue'],
                   alpha=0.5, color='red', s=50,
                   edgecolors='darkred', linewidth=1,
                   marker='*', label=f"Blockbusters (n = {len(blockbusters)})")
        
        z = np.polyfit(rating_revenue['vote_average'],
                       np.log10(rating_revenue['revenue']), 1)
        p_all = np.poly1d(z)
        
        vote_range = np.linspace(0, 10, 100)
        trend = 10 ** p_all(vote_range)
        
        ax.plot(vote_range, trend,
                color='darkgreen', linewidth=2.5, linestyle='--',
                label=f"Trend (slope = {z[0]:.2f})")
        
        ax.set_yscale('log')
        ax.set_xlabel('Vote Average (Rating)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax.set_title('Rating vs Revenue', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Average Revenue by Rating Category")
        
        rating_revenue['category'] = rating_revenue['vote_average'].apply(categorize_ratings)
        category_order = ['Poor\n(<5.0)', 'Average\n(5.0 - 6.5)', 'Good\n(6.5 - 7.5)', 'Excellent\n(>= 7.5)']
        
        avg_revenue_all = rating_revenue.groupby('category')['revenue'].mean() / 1e6
        avg_revenue_all = avg_revenue_all.reindex(category_order)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['red', 'orange', 'yellow', 'green']
        bars = ax.bar(category_order, avg_revenue_all, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'${height:.1f}M',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Rating Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
        ax.set_title('Average Revenue by Rating', fontsize=13, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correlation", f"{correlation:.3f}")
    with col2:
        st.metric("Blockbusters Identified", f"{len(blockbusters)}")
    with col3:
        st.metric("Blockbuster Threshold", f"${blockbuster_threshold/1e6:.1f}M")
    
    st.warning("""
    **Key Insight:**
    
    - **Weak positive correlation (0.116)** - Critical acclaim doesn't guarantee box office success!
    - Better-rated films earn more **on average**, but quality alone isn't enough
    - Blockbusters require factors beyond critical acclaim: marketing, timing, and audience appeal
    - The bar chart shows progressive revenue increases, but the relationship is weak
    """)

elif page == "Runtime Analysis":
    st.markdown("## Runtime Analysis")
    st.markdown("### Is There an Optimal Movie Length for Box Office Success?")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Scatter Plot: Runtime vs Revenue")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        regular_movies = runtime_revenue[~runtime_revenue['is_blockbuster']]
        blockbusters = runtime_revenue[runtime_revenue['is_blockbuster']]
        
        ax.scatter(regular_movies['runtime'], regular_movies['revenue'],
                   alpha=0.5, color='steelblue', s=50,
                   edgecolor='white', linewidth=0.5, label='All movies')
        
        ax.scatter(blockbusters['runtime'], blockbusters['revenue'],
                   alpha=0.7, color='red', s=100,
                   edgecolors='darkred', linewidth=1,
                   marker='*', label=f"Blockbusters (n = {len(blockbusters)})")
        
        z_runtime = np.polyfit(runtime_revenue['runtime'],
                               np.log10(runtime_revenue['revenue']), 1)
        p_runtime = np.poly1d(z_runtime)
        runtime_range = np.linspace(runtime_revenue['runtime'].min(),
                                    runtime_revenue['runtime'].max(), 100)
        trend_runtime = 10 ** p_runtime(runtime_range)
        
        ax.plot(runtime_range, trend_runtime, color='darkgreen', linewidth=2.5,
                linestyle='--', label=f"Trend (slope = {z_runtime[0]:.3f})")
        
        ax.axvline(x=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axvline(x=120, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axvline(x=150, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.set_yscale('log')
        ax.set_xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax.set_title('Runtime vs Revenue', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Average Revenue by Runtime Category")
        
        category_order = ['Short\n(<90 min)', 'Standard\n(90-120 min)',
                         'Long\n(120-150 min)', 'Epic\n(>=150 min)']
        avg_revenue_runtime = runtime_revenue.groupby('runtime_category')['revenue'].mean() / 1e6
        avg_revenue_runtime = avg_revenue_runtime.reindex(category_order)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_runtime = ['blue', 'violet', 'purple', 'pink']
        bars = ax.bar(category_order, avg_revenue_runtime,
                      color=colors_runtime, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}M', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Runtime Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
        ax.set_title('Average Revenue by Runtime', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    correlation_runtime = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Correlation", f"{correlation_runtime:.3f}")
    with col2:
        st.metric("Short Films (<90 min)",
              f"${runtime_revenue[runtime_revenue['runtime_category'] == 'Short\n(<90 min)']['revenue'].mean()/1e6:.0f}M")
    with col3:
        st.metric("Standard Films (90-120 min)",
              f"${runtime_revenue[runtime_revenue['runtime_category'] == 'Standard\n(90-120 min)']['revenue'].mean()/1e6:.0f}M")
    with col4:
        st.metric("Long Films (120-150 min)",
              f"${runtime_revenue[runtime_revenue['runtime_category'] == 'Long\n(120-150 min)']['revenue'].mean()/1e6:.0f}M")
    with col5:
        st.metric("Epic Films (>=150 min)",
              f"${runtime_revenue[runtime_revenue['runtime_category'] == 'Epic\n(>=150 min)']['revenue'].mean()/1e6:.0f}M")
    
    st.info("""
    **Key Insights:**
    
    - **Positive correlation (0.247)** between runtime and revenue
    - Longer films generate higher revenues on average:
      - Epic films (>=150 min): **$245M average**
      - Long films (120-150 min): **$172M average**
      - Standard films (90-120 min): **$88M average**
      - Short films (<90 min): **$73M average**
    - Runtime correlates with revenue **not because length drives success**, but because studios invest more in longer films
    - Blockbusters (143 min mean) are longer than regular films (106 min mean)
    """)

elif page == "Popularity Analysis":
    st.markdown("## Popularity Analysis")
    st.markdown("### Does Higher Popularity Translate to Higher Revenue?")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Scatter Plot: Popularity vs Revenue")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        regular_movies = popularity_revenue[~popularity_revenue['is_blockbuster']]
        blockbusters = popularity_revenue[popularity_revenue['is_blockbuster']]
        
        ax.scatter(regular_movies['popularity'], regular_movies['revenue'],
                   alpha=0.5, color='steelblue', s=50,
                   edgecolor='white', linewidth=0.5, label='All movies')
        
        ax.scatter(blockbusters['popularity'], blockbusters['revenue'],
                   alpha=0.7, color='orange', s=100,
                   edgecolors='darkred', linewidth=1,
                   marker='*', label=f"Blockbusters (n = {len(blockbusters)})")
        
        z_pop = np.polyfit(np.log10(popularity_revenue['popularity'] + 1),
                           np.log10(popularity_revenue['revenue']), 1)
        p_pop = np.poly1d(z_pop)
        
        pop_range = np.logspace(np.log10(popularity_revenue['popularity'].min() + 1),
                               np.log10(popularity_revenue['popularity'].max() + 1), 100)
        trend_pop = 10 ** p_pop(np.log10(pop_range))
        
        ax.plot(pop_range, trend_pop, color='darkgreen', linewidth=2.5,
                linestyle='--', label=f"Trend (slope = {z_pop[0]:.2f})")
        
        ax.axvline(x=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axvline(x=30, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Popularity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax.set_title('Popularity vs Revenue', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Average Revenue by Popularity")
        
        category_order = ['Low\n(<5)', 'Medium\n(5-15)', 'High\n(15-30)', 'Very High\n(>=30)']
        avg_revenue_pop = popularity_revenue.groupby('popularity_category')['revenue'].mean() / 1e6
        avg_revenue_pop = avg_revenue_pop.reindex(category_order)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_pop = ['lightcoral', 'orange', 'yellow', 'green']
        bars = ax.bar(category_order, avg_revenue_pop,
                      color=colors_pop, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"${height:.1f}M", ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Popularity Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
        ax.set_title('Average Revenue by Popularity', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    correlation_pop = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])
    
    st.metric("Popularity-Revenue Correlation", f"{correlation_pop:.3f}")
    
    st.success("""
    **Key Insights:**
    
    - **Moderate positive correlation (0.401)** - Popularity is the **strongest single predictor** among variables analyzed!
    - Unlike budget/rating/runtime, popularity represents marketing reach, social media buzz, and audience anticipation
    - Blockbusters are heavily concentrated in high popularity regions
    - High popularity is practically a **prerequisite for blockbuster success**
    - Studios must invest in generating pre-release buzz and maintaining visibility throughout theatrical runs
    """)

elif page == "Temporal Trends":
    st.markdown("## Temporal Trends")
    st.markdown("### How Has the Film Industry Evolved Over Time?")
    
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
    
    # Four plots in 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    decade_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(decade_stats)))
    
    # Revenue evolution
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
    
    # Budget growth
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
    
    # Film quality
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
    
    # Blockbuster concentration
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
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.info(r"""
    **Key Insights:**
    
    - **Revenue evolution**: Grew from \$46M (1960s) to \$149M (2010s) - a **224% increase**
    - **Budget growth**: Exploded from \$6M (1960s) to \$48M (2010s), steepest rise in 1980s-1990s
    - **Film quality**: Remarkably stable (6.26-7.04 range), older films score slightly higher
    - **Blockbuster concentration**: Surged from 1.4% (1960s) to 15.5% (2010s)
    - Industry shifted from diverse mid-budget films to fewer, bigger "tentpole" releases
    """)

elif page == "Seasonal Patterns":
    st.markdown("## Seasonal Patterns")
    st.markdown("### Does Release Timing Matter for Box Office Success?")
    
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
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[int(x) - 1])
    
    # Four plots in 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_colors = {'Winter': 'blue', 'Spring': 'lightgreen', 'Summer': 'orange', 'Fall': 'brown'}
    season_stats_ordered = season_stats.set_index('season').reindex(season_order).reset_index()
    
    # Average revenue by season
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
    
    # Blockbuster rate by season
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
    
    # Average revenue by month
    bars3 = ax3.bar(monthly_stats['month_name'], monthly_stats['avg_revenue'],
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    max_month_idx = monthly_stats['avg_revenue'].idxmax()
    bars3[max_month_idx].set_color('yellow')
    bars3[max_month_idx].set_edgecolor('orange')
    bars3[max_month_idx].set_linewidth(2.5)
    
    for i, row in monthly_stats.iterrows():
        height = row['avg_revenue']
        ax3.text(i, height, f"{height:.0f}M", ha='center', va='bottom',
                 fontsize=8, fontweight='bold')
    ax3.set_xlabel('Release Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Revenue by Month (Peak in Yellow)', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(monthly_stats['avg_revenue']) * 1.15)
    
    # Release volume by season
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
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Best performing season/month
    best_season = season_stats_ordered.loc[season_stats_ordered['avg_revenue'].idxmax()]
    best_month = monthly_stats.loc[monthly_stats['avg_revenue'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Season", best_season['season'],
                  f"${best_season['avg_revenue']:.1f}M avg")
    with col2:
        st.metric("Best Month", best_month['month_name'],
                  f"${best_month['avg_revenue']:.1f}M avg")
    with col3:
        st.metric("Peak Blockbuster Rate",
                  best_season['season'],
                  f"{best_season['blockbuster_rate']:.1f}%")
    
    st.success("""
    **Key Insights:**
    
    - **Summer dominates** as the blockbuster season - the "summer blockbuster" phenomenon is real
    - Studios strategically release biggest films during school vacations when families have maximum availability
    - **Winter (holidays)** ranks second, capitalizing on Christmas and New Year audiences
    - **Spring and Fall** serve as "dump months" for lower-budget films
    - Release timing significantly impacts box office performance
    """)

elif page == "Genre Analysis":
    st.markdown("## Genre Analysis")
    st.markdown("### Which Genres Dominate the Box Office?")
    
    genre_data = df[['primary_genre', 'revenue', 'popularity', 'is_blockbuster', 'vote_average']].dropna()
    
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count'],
        'popularity': 'mean',
        'vote_average': 'mean',
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    genre_stats.columns = ['genre', 'avg_revenue', 'count', 'avg_popularity', 'avg_rating', 'blockbuster_rate']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    genre_stats = genre_stats[genre_stats['count'] >= 50].copy()
    genre_stats_sorted = genre_stats.sort_values('avg_revenue', ascending=False)
    
    # Create 2x2 grid of plots matching main code
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, left=0.10, right=0.96, top=0.93, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Top 10 genres by average revenue
    top_revenue_genres = genre_stats_sorted.head(10)
    
    bars1 = ax1.barh(range(len(top_revenue_genres)), top_revenue_genres['avg_revenue'],
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars1[0].set_color('red')
    bars1[0].set_edgecolor('darkred')
    bars1[0].set_linewidth(2.5)
    
    ax1.set_yticks(range(len(top_revenue_genres)))
    ax1.set_yticklabels(top_revenue_genres['genre'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_revenue_genres.iterrows()):
        width = row['avg_revenue']
        ax1.text(width, i, f"${width:.1f}M",
                 ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Average Revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 genres by average revenue', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(top_revenue_genres['avg_revenue']) * 1.15)
    
    # Plot 2: Top 10 genres by blockbuster rate
    top_blockbuster_genres = genre_stats.nlargest(10, 'blockbuster_rate')
    
    bars2 = ax2.barh(range(len(top_blockbuster_genres)), top_blockbuster_genres['blockbuster_rate'],
                     color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars2[0].set_color('red')
    bars2[0].set_edgecolor('darkred')
    bars2[0].set_linewidth(2.5)
    
    ax2.set_yticks(range(len(top_blockbuster_genres)))
    ax2.set_yticklabels(top_blockbuster_genres['genre'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_blockbuster_genres.iterrows()):
        width = row['blockbuster_rate']
        ax2.text(width, i, f"{width:.1f}%",
                 ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Blockbuster rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax2.set_title('Top 10 genres by blockbuster rate', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.8, axis='x')
    ax2.set_xlim(0, max(top_blockbuster_genres['blockbuster_rate']) * 1.12)
    
    # Plot 3: Top 10 genres by average popularity
    top_popularity_genres = genre_stats.nlargest(10, 'avg_popularity')
    
    bars3 = ax3.barh(range(len(top_popularity_genres)), top_popularity_genres['avg_popularity'],
                     color='purple', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    bars3[0].set_color('darkviolet')
    bars3[0].set_edgecolor('black')
    bars3[0].set_linewidth(2.5)
    
    ax3.set_yticks(range(len(top_popularity_genres)))
    ax3.set_yticklabels(top_popularity_genres['genre'], fontsize=11)
    
    for i, (idx, row) in enumerate(top_popularity_genres.iterrows()):
        width = row['avg_popularity']
        ax3.text(width, i, f"{width:.1f}",
                 ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Average popularity score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Genre', fontsize=12, fontweight='bold')
    ax3.set_title('Top 10 genres by average popularity', fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(top_popularity_genres['avg_popularity']) * 1.12)
    
    # Plot 4: Genre success matrix (scatter plot)
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
    
    ax4.set_xlabel('Average revenue ($ Millions)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Blockbuster rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Genre success matrix (Size = film count, Color = popularity)',
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    color_bar = plt.colorbar(scatter, ax=ax4)
    color_bar.set_label('Avg popularity', fontsize=10, fontweight='bold')
    
    fig.suptitle('Genre dominance: which types of films rule the box office?',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99], h_pad=3, w_pad=2.5)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.markdown("### 📊 Genre Performance Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    top_genre = genre_stats.sort_values('avg_revenue', ascending=False).iloc[0]
    top_blockbuster = genre_stats.sort_values('blockbuster_rate', ascending=False).iloc[0]
    top_popularity = genre_stats.sort_values('avg_popularity', ascending=False).iloc[0]
    
    with col1:
        st.metric("Highest Revenue Genre",
                  top_genre['genre'],
                  f"${top_genre['avg_revenue']:.1f}M avg")
    
    with col2:
        st.metric("Highest Blockbuster Rate",
                  top_blockbuster['genre'],
                  f"{top_blockbuster['blockbuster_rate']:.1f}%")
    
    with col3:
        st.metric("Most Popular Genre",
                  top_popularity['genre'],
                  f"{top_popularity['avg_popularity']:.1f} avg")
    
    st.markdown("---")
    
    # Genre comparison table
    st.markdown("### Top 10 Genres Comparison")
    
    display_stats = genre_stats.sort_values('avg_revenue', ascending=False).head(10)[
        ['genre', 'avg_revenue', 'count', 'blockbuster_rate', 'avg_popularity']
    ].copy()
    
    display_stats.columns = ['Genre', 'Avg Revenue ($M)', 'Films', 'Blockbuster Rate (%)', 'Avg Popularity']
    display_stats['Avg Revenue ($M)'] = display_stats['Avg Revenue ($M)'].round(1)
    display_stats['Blockbuster Rate (%)'] = display_stats['Blockbuster Rate (%)'].round(1)
    display_stats['Avg Popularity'] = display_stats['Avg Popularity'].round(1)
    
    st.dataframe(display_stats, use_container_width=True, hide_index=True)
    
    st.info(r"""
    **Key Insights:**
    
    - **Family films lead** (\$257M avg), followed by Animation (\$242M) and Adventure (\$223M)
    - **Animation** shows highest blockbuster rate (32.9%) - nearly 1 in 3 animation films is a blockbuster!
    - **Family-friendly, spectacle-driven content** dominates box offices
    - The scatter plot reveals Animation and Adventure occupy the "sweet spot" of high revenue + high blockbuster rate
    - Drama is the most produced genre but generates relatively low revenues
    - Genre choice is a **critical success factor** - studios prioritize spectacle and family appeal for maximum returns
    """)

elif page == "Production Origins":
    st.markdown("## Production Origins")
    st.markdown("### Do Country & Company Determine Success?")
    
    # Country analysis
    country_data = df[['countries_str', 'revenue', 'is_blockbuster']].dropna()
    country_data['primary_country'] = country_data['countries_str'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
    )
    
    country_stats = country_data.groupby('primary_country').agg({
        'revenue': ['mean', 'count'],
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    country_stats.columns = ['country', 'avg_revenue', 'count', 'blockbuster_rate']
    country_stats['avg_revenue'] = country_stats['avg_revenue'] / 1e6
    country_stats = country_stats[country_stats['count'] >= 100]
    
    # Company analysis
    company_data = df[['companies_str', 'revenue', 'is_blockbuster']].dropna()
    company_data['primary_company'] = company_data['companies_str'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
    )
    
    company_stats = company_data.groupby('primary_company').agg({
        'revenue': ['mean', 'count'],
        'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    company_stats.columns = ['company', 'avg_revenue', 'count', 'blockbuster_rate']
    company_stats['avg_revenue'] = company_stats['avg_revenue'] / 1e6
    company_stats = company_stats[company_stats['count'] >= 20]
    
    # Create 2x2 grid of plots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.60, left=0.22, right=0.96, top=0.98, bottom=0.07)
    
    # Top countries by revenue
    top_countries_revenue = country_stats.nlargest(12, 'avg_revenue')
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(range(len(top_countries_revenue)), top_countries_revenue['avg_revenue'],
                     color='green', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars1[0].set_color('yellow')
    bars1[0].set_edgecolor('orange')
    bars1[0].set_linewidth(2.5)
    ax1.set_yticks(range(len(top_countries_revenue)))
    ax1.set_yticklabels(top_countries_revenue['country'], fontsize=10)
    for i, (idx, row) in enumerate(top_countries_revenue.iterrows()):
        ax1.text(row['avg_revenue'] + 3, i, f"${row['avg_revenue']:.1f}M",
                va='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Average Revenue ($ Millions)', fontsize=11)
    ax1.set_ylabel('Country', fontsize=11)
    ax1.set_title('Top Countries by Average Revenue', fontsize=12, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top countries by blockbuster rate
    top_countries_blockbuster = country_stats.nlargest(12, 'blockbuster_rate')
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(range(len(top_countries_blockbuster)), top_countries_blockbuster['blockbuster_rate'],
                     color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2[0].set_color('red')
    bars2[0].set_edgecolor('darkred')
    bars2[0].set_linewidth(2.5)
    ax2.set_yticks(range(len(top_countries_blockbuster)))
    ax2.set_yticklabels(top_countries_blockbuster['country'], fontsize=10)
    for i, (idx, row) in enumerate(top_countries_blockbuster.iterrows()):
        ax2.text(row['blockbuster_rate'] + 0.3, i, f"{row['blockbuster_rate']:.1f}%",
                va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Blockbuster Rate (%)', fontsize=11)
    ax2.set_ylabel('Country', fontsize=11)
    ax2.set_title('Top Countries by Blockbuster Rate', fontsize=12, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Top 12 companies by revenue
    top_companies_revenue = company_stats.nlargest(12, 'avg_revenue')
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.barh(range(len(top_companies_revenue)), top_companies_revenue['avg_revenue'],
                     color='purple', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3[0].set_color('darkviolet')
    bars3[0].set_edgecolor('black')
    bars3[0].set_linewidth(2.5)
    ax3.set_yticks(range(len(top_companies_revenue)))
    ax3.set_yticklabels(top_companies_revenue['company'], fontsize=10)
    for i, (idx, row) in enumerate(top_companies_revenue.iterrows()):
        ax3.text(row['avg_revenue'] + 10, i, f"${row['avg_revenue']:.1f}M",
                va='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Average Revenue ($ Millions)', fontsize=11)
    ax3.set_ylabel('Production Company', fontsize=11)
    ax3.set_title('Top 12 Companies by Average Revenue', fontsize=12, fontweight='bold', pad=10)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Top 12 companies by blockbuster rate
    top_companies_blockbuster = company_stats.nlargest(12, 'blockbuster_rate')
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.barh(range(len(top_companies_blockbuster)), top_companies_blockbuster['blockbuster_rate'],
                     color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4[0].set_color('darkgreen')
    bars4[0].set_edgecolor('darkgreen')
    bars4[0].set_linewidth(2.5)
    ax4.set_yticks(range(len(top_companies_blockbuster)))
    ax4.set_yticklabels(top_companies_blockbuster['company'], fontsize=10)
    for i, (idx, row) in enumerate(top_companies_blockbuster.iterrows()):
        ax4.text(row['blockbuster_rate'] + 1.5, i, f"{row['blockbuster_rate']:.1f}%",
                va='center', fontsize=9, fontweight='bold')
    ax4.set_xlabel('Blockbuster Rate (%)', fontsize=11)
    ax4.set_ylabel('Production Company', fontsize=11)
    ax4.set_title('Top 12 Companies by Blockbuster Rate', fontsize=12, fontweight='bold', pad=10)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Statistics
    top_country = country_stats.sort_values('avg_revenue', ascending=False).iloc[0]
    top_company = company_stats.sort_values('avg_revenue', ascending=False).iloc[0]
    top_blockbuster_country = country_stats.sort_values('blockbuster_rate', ascending=False).iloc[0]
    top_blockbuster_company = company_stats.sort_values('blockbuster_rate', ascending=False).iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Country Leaders")
        st.metric("Highest Revenue Country",
                  top_country['country'],
                  f"${top_country['avg_revenue']:.1f}M avg")
        st.metric("Highest Blockbuster Rate",
                  top_blockbuster_country['country'],
                  f"{top_blockbuster_country['blockbuster_rate']:.1f}%")
    
    with col2:
        st.markdown("#### Company Leaders")
        st.metric("Highest Revenue Company",
                  top_company['company'],
                  f"${top_company['avg_revenue']:.1f}M avg")
        st.metric("Highest Blockbuster Rate",
                  top_blockbuster_company['company'],
                  f"{top_blockbuster_company['blockbuster_rate']:.1f}%")
    
    st.info("""
    **Key Insights:**
    
    - **Production origins significantly influence success**
    - English-speaking countries (US, UK) dominate blockbuster production
    - Major studios have established formulas for blockbuster success
    - Lucasfilm leads companies with 60% blockbuster rate
    - UK tops blockbuster rate (likely due to US-UK co-productions like Harry Potter, James Bond)
    """)

elif page == "Conclusions":
    st.markdown("## Conclusions")
    st.markdown("### Is There a Blockbuster Formula?")
    
    st.markdown("---")
    
    st.markdown("### The Answer: No Guaranteed Formula, But Clear Patterns")
    
    st.warning("""
    The data reveals there is **NO guaranteed formula** for blockbuster success,
    but there are patterns that can increase the likelihood of success.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Success Factors")
        st.markdown("""
        **Successful Blockbusters Typically Combine:**
        
        - **High budgets** ($50M+) enabling spectacle and star power
        - **Family-friendly genres** (Animation, Adventure, Family)
        - **Strategic timing** (summer or holiday releases)
        - **Longer runtimes** (120-150 minutes)
        - **High pre-release buzz** (popularity/marketing)
        - **Major studio backing**
        """)
    
    with col2:
        st.markdown("#### Important Caveats")
        st.markdown("""
        **However:**
        
        - High variance at all budget levels
        - **Exceptional creative execution** matters
        - **Cultural timing** is crucial
        - **Luck** remains a factor
        - Quality ≠ Commercial success
        - No factor guarantees success alone
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Key Correlations Summary")
    
    correlations = {
        'Budget → Revenue': 0.71,
        'Popularity → Revenue': 0.40,
        'Runtime → Revenue': 0.25,
        'Rating → Revenue': 0.12
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    factors = list(correlations.keys())
    values = list(correlations.values())
    colors_corr = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
    
    bars = ax.barh(factors, values, color=colors_corr, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (factor, value) in enumerate(correlations.items()):
        ax.text(value + 0.02, i, f'{value:.2f}', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Strength: Factors vs Revenue', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.markdown("### 🎬 The Modern Film Industry")
    
    st.info("""
    **The "Tentpole" Strategy:**
    
    The industry operates on fewer, bigger bets on franchise-ready spectacles:
    
    - Strategic release during peak seasons (Summer, Winter holidays)
    - **78.3% of films are profitable**
    - But only **~10% achieve true blockbuster status**
    - Blockbuster concentration has increased from 1.4% (1960s) to 15.5% (2010s)
    - Average revenue grew 224% from 1960s to 2010s
    
    **Bottom line:** Data can improve odds of success, but cannot guarantee it.
    The magic and risk of cinema lie in that irreducible uncertainty.
    """)
    
    st.markdown("---")
    
    st.markdown("### Final Insights")
    
    st.success("""
    **What We Learned:**
    
    1. **Budget matters most** (0.71 correlation), but shows diminishing returns
    2. **Popularity is the strongest operational predictor** - marketing and buzz are crucial
    3. **Critical acclaim has minimal impact** on commercial success (0.12 correlation)
    4. **Timing is strategic** - summer blockbusters are real, not myth
    5. **Genre choice is critical** - family-friendly spectacles dominate
    6. **The industry has evolved** toward fewer, bigger tentpole releases
    7. **Production origins matter** - major studios and English-speaking markets dominate
    8. **No single factor guarantees success** - it's a complex, multi-dimensional equation
    
    While data analysis reveals patterns, **exceptional creative execution, cultural timing,
    and some element of luck remain essential** to blockbuster success.
    """)
    
    st.markdown("---")
    
    st.markdown("### About This Project")
    
    st.write(f"""
    This comprehensive analysis examined **{len(df):,} films** from The Movies Dataset (Kaggle)
    spanning {int(df['release_year'].min())} to {int(df['release_year'].max())} to understand
    what factors contribute to box office success.
    
    Through nine major analytical components - Budget vs Revenue, Profitability, Rating Analysis,
    Runtime Analysis, Popularity Analysis, Temporal Trends, Seasonal Patterns, Genre Analysis,
    and Production Origins - we explored the complex, multi-dimensional nature of film industry
    economics.
    
    **Key Dataset Statistics:**
    - {len(df):,} total movies analyzed
    - {df['budget'].notna().sum():,} films with budget data
    - {df['revenue'].notna().sum():,} films with revenue data
    - {df['is_profitable'].sum():,} profitable films (78.3%)
    - {df['is_blockbuster'].sum():,} blockbusters (top 10% by revenue)
    """)

