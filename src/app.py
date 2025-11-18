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

# Page configuation
st.set_page_config(
    page_title = "Anatomy of a Blockbuster",
    page_icon = "🎬",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS
st.markdown("""
    <style
    .main-header{
            font-sizw: 3rem;
            font-weight: bold;
            color: blue;
            text-align: center;
            margin-bottom: rem;
    }
            
    .section-header {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
    }
    <style>
""", unsafe_allow_html = True)

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

# Loading the data
df = load_data

# Sidebar
st.sidebar.markdown("# 🎬 Navigation")
page = st.sidebar.radio(
    "Select Analysis:",
    [
        "Overview",
        "Budget vs Revenue",
        "Profitability",
        "Rating Analysis",
        "Runtime Analysis",
        "Popularity Analysis",
        "Temporal Trends",
        "Seasonal Patterns",
        "Genre Analysis",
        "Conclusions"
    ]
)

# Main content
if page == "Overview":
    st.markdown('<p class = "main-header">🎬 Anatomy of a Blockbuster</p>', unsafe_allow_html = True)
    st.markdown("### Understanding the Factors Behind Box Office Success")

    st.markdown("---")

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
        st.markdown("### Quality Metrics")
        st.write(f"- Average rating: **{df['vote_average'].mean():.2f}/10**")
        st.write(f"- Average runtime: **{df['runtime'].mean():.0f} minutes")
        st.write(f"- Unique genres: **{df['primary_genre'].nunique()}**")

    st.markdown("---")

    st.markdown("### Project Goals")
    st.write("""
    This analysis explores:
    - **Financial patterns**: How budget influences revenue and profit
    - **Quality metrics**: The relationship between ratings and box office success
    - **Timing strategies**: Optimal release windows and seasonal patterns
    - **Genre performance**: Which types of films dominate the box office
    """)

elif page == "Budget vs Revenue":
    st.markdown("## Budget vs Revenue Analysis")
    st.markdown("### Does Spending More Guarantee Higher Revenue?")

    clean_data = df[['budget', 'revenue']].dropna()

    fig, ax = plt.subplots(figsize = (12, 8))

    ax.scatter(clean_data['budget'], clean_data['revenue'],
               alpha = 0.5, color = 'steelblue', label = 'Movies')
    
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
    ax.set_title('Budget vs Revenue', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📊 Key Statistics")

    correlation = clean_data['budget'].corr(clean_data['revenue'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Correlation", f"{correlation:.3f}")

    with col2: 
        low_budget = clean_data[clean_data['budget'] < 1e6]
        st.metric("Low Budget (<$1M)", f"${low_budget['revenue'].mean()/1e6:.1f}M avg")

    with col3:
        high_budget = clean_data[clean_data['budget'] >= 5e7]
        st.metric("High Budget (>$50M)", f"${high_budget['revenue'].mean()/1e6:.1f}M avg")

    st.markdown("---")

    st.info("""
    **Key Insights: **Strong positive correlation (0.71) between budget and revenue,
            but diminishing returns - doubling budget doesn't double revenue.
    """)

elif page == "Profitability":
    st.markdown("##Profitability Analysis")
    st.markdown("### Does Budget Influence Profitability?")

    clean_profit_data = df[['budget', 'profit']].dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize = (8, 6))

        profitability_counts = clean_profit_data['profit'].apply(
            lambda x: "Profitable" if x > 0 else "Unprofitable"
        ).value_counts()

        colors = ['lightblue', 'lightcoral']
        explode = (0.05, 0)

        ax.pie(profitability_counts, labels = profitability_counts.index,
               autopct = '%1.1f%%', startangle = 90, colors = colors,
               explode = explode, textprops = {'fontsize':12, 'fontweight': 'bold'})
        
        ax.set_title("Film Profitability Distribution", fontsize = 14, fontweight = "bold")

        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📊 Profitability Stats")
        total = len(clean_profit_data)
        profitable_count = (clean_profit_data['profit'] > 0).sum()

        st.metric("Total Films", f"{total:,}")

        st.metric("Total Films", f"{total:,}")
        st.metric("Profitable", f"{profitable_count:,} ({100*profitable_count/total:.1f}%)")
        st.metric("Average Profit", f"${clean_profit_data['profit'].mean()/1e6:.1f}M")
    
    st.success("""
    **Key Finding:** 78.3% of films with complete financial data are profitable!
    """)

elif page == "Rating Analysis":
    st.markdown("## Rating vs Revenue Analysis")
    st.markdown("### Do Better-Rated Movies Make More Money?")

    rating_revenue = df[['vote_average', 'revenue']].dropna()

    fig, ax = plt.subplots(figsize = (12, 8))

    ax.scatter(rating_revenue['vote_average'], rating_revenue['revenue'],
               alpha = 0.5, color = 'steelblue', s = 50)
    
    z = np.polyfit(rating_revenue['vote_average'],
                   np.log10(rating_revenue['revenue']), 1)
    p_rating = np.poly1d(z)
    vote_range = np.linspace(0, 10, 100)
    trend = 10 ** p_rating(vote_range)

    ax.plot(vote_range, trend, color='darkgreen', linewidth=2.5, linestyle='--',
            label=f"Trend (slope = {z[0]:.2f})")
    
    ax.set_yscale('log')
    ax.set_xlabel('Vote Average (Rating)', fontsize=12)
    ax.set_ylabel('Revenue ($)', fontsize=12)
    ax.set_title('Rating vs Revenue', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

    correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])
    st.metric("Correlation", f"{correlation:.3f}")

    st.warning("""
    **Key Insight:** Weak positive correlation (0.116) - Critical acclaim doesn't guarantee box office success!
    Quality alone is not enough; marketing, timing, and audience appeal matter equally.
    """)

elif page == "Runtime Analysis":
    st.markdown("## Runtime Analysis")
    st.markdown("### Is there an Optimal Movie Length?")

    runtime_revenue = df[['runtime', 'revenue']].dropna()

    def categorize_runtime(runtime):
        if runtime < 90:
            return 'Short (<90 min)'
        elif runtime < 120:
            return 'Standard (90-120 min)'
        elif runtime < 150:
            return 'Long (120-150 min)'
        else:
            return 'Epic (>=150 min)'
    
    runtime_revenue['runtime_category'] = runtime_revenue['runtime'].apply(categorize_runtime)
    category_order = ['Short (<90 min)', 'Standard (90-120 min)', 'Long (120-150 min)', 'Epic (>=150 min)']
    avg_revenue_runtime = runtime_revenue.groupby('runtime_category')['revenue'].mean() / 1e6
    avg_revenue_runtime = avg_revenue_runtime.reindex(category_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_runtime = ['blue', 'violet', 'purple', 'pink']
    bars = ax.bar(category_order, avg_revenue_runtime, color=colors_runtime, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'${height:.1f}M',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Runtime Category', fontsize=12)
    ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12)
    ax.set_title('Average Revenue by Runtime Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()
    
    correlation = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])
    st.metric("Runtime-Revenue Correlation", f"{correlation:.3f}")
    
    st.info("""
    Longer films generate higher revenues on average:
    - Epic films (>=150 min): **$245M average**
    - Standard films (90-120 min): **$88M average**
    """)

elif page == "Popularity Analysis":
    st.markdown("## Popularity Analysis")
    st.markdown("### Does Buzz Equal Box Office Success?")

    popularity_revenue = df[['popularity', 'revenue']].dropna()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(popularity_revenue['popularity'], popularity_revenue['revenue'],
               alpha=0.5, color='steelblue', s=50)
    
    z_pop = np.polyfit(np.log10(popularity_revenue['popularity'] + 1),
                       np.log10(popularity_revenue['revenue']), 1)
    p_pop = np.poly1d(z_pop)
    
    pop_range = np.logspace(np.log10(popularity_revenue['popularity'].min() + 1),
                           np.log10(popularity_revenue['popularity'].max() + 1), 100)
    trend_pop = 10 ** p_pop(np.log10(pop_range))
    
    ax.plot(pop_range, trend_pop, color='darkgreen', linewidth=2.5, linestyle='--',
            label=f"Trend (slope = {z_pop[0]:.2f})")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Popularity Score', fontsize=12)
    ax.set_ylabel('Revenue ($)', fontsize=12)
    ax.set_title('Popularity vs Revenue', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    correlation = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])
    st.metric("Popularity-Revenue Correlation", f"{correlation:.3f}")
    
    st.success("""
    **Key Insight:** Moderate positive correlation (0.401) - Popularity is the strongest 
    single predictor among variables analyzed! Pre-release buzz and marketing reach are crucial.
    """)

elif page == "Temporal Trends":
    st.markdown("## Tempooral Trends")
    st.markdown("### How has the Industry Evolved Over Time?")

    decade_data = df[['decade', 'revenue']].dropna()
    decade_stats = decade_data.groupby('decade')['revenue'].mean() / 1e6
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(decade_stats.index.astype(str), decade_stats.values,
           color='steelblue', alpha=0.8, edgecolor='black')
    
    for i, (decade, revenue) in enumerate(decade_stats.items()):
        ax.text(i, revenue, f'${revenue:.0f}M',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12)
    ax.set_title('Revenue Evolution by Decade', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()
    
    st.info("""
    **Key Insights:**
    - Average revenue grew from **$46M (1960s)** to **$149M (2010s)** - a 224% increase
    - Dramatic growth from 1990s onwards reflects the "blockbuster era"
    - Industry shifted from diverse mid-budget films to fewer, bigger tentpole releases
    """)

elif page == "Seasonal Patterns":
    st.markdown("## Seasonal Patterns")
    st.markdown("### Does Release Timing Matter?")

    season_data = df[['release_season', 'revenue']].dropna()
    season_stats = season_data.groupby('release_season')['revenue'].mean() / 1e6
    
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_stats = season_stats.reindex(season_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    season_colors = {'Winter': 'blue', 'Spring': 'lightgreen', 'Summer': 'orange', 'Fall': 'brown'}
    colors = [season_colors[s] for s in season_order]
    
    bars = ax.bar(season_order, season_stats.values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'${height:.1f}M',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Release Season', fontsize=12)
    ax.set_ylabel('Average Revenue ($ Millions)', fontsize=12)
    ax.set_title('Average Revenue by Season', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()
    
    st.success("""
    **Summer dominates!** The "summer blockbuster" phenomenon is real.
    
    Studios strategically release biggest films during school vacation when families have maximum availability.
    Winter (holidays) ranks second, while Spring/Fall serve as "dump months."
    """)

elif page == "Genre Analysis":
    st.markdown("## Genre Analysis")
    st.markdown("### Which Genres Dominate the Box Office?")

    genre_data = df[['primary_genre', 'revenue']].dropna()
    genre_stats = genre_data.groupby('primary_genre').agg({
        'revenue': ['mean', 'count']
    }).reset_index()
    
    genre_stats.columns = ['genre', 'avg_revenue', 'count']
    genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6
    genre_stats = genre_stats[genre_stats['count'] >= 50]
    genre_stats = genre_stats.sort_values('avg_revenue', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(genre_stats)), genre_stats['avg_revenue'],
                   color='steelblue', alpha=0.8, edgecolor='black')
    
    bars[0].set_color('red')
    bars[0].set_edgecolor('darkred')
    bars[0].set_linewidth(2.5)
    
    ax.set_yticks(range(len(genre_stats)))
    ax.set_yticklabels(genre_stats['genre'])
    
    for i, (idx, row) in enumerate(genre_stats.iterrows()):
        ax.text(row['avg_revenue'] + 5, i, f"${row['avg_revenue']:.1f}M",
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Average Revenue ($ Millions)', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)
    ax.set_title('Top 10 Genres by Average Revenue', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    st.pyplot(fig)
    plt.close()
    
    st.info("""
    **Key Insights:**
    - **Family films lead** ($257M avg), followed by Animation ($242M) and Adventure ($223M)
    - Family-friendly, spectacle-driven content dominates box offices
    - Drama is most produced but generates low revenues - Hollywood prioritizes spectacle over artistic merit
    """)

elif page == "Conclusions":
    st.markdown("## Conclusions")
    st.markdown("### Is There a Blockbuster Formula?")

    st.markdown("---")
    st.markdown("### The answer is: No Guaranted Formula for Blockbuster Success")

    st.warning("""
    The data reveals there is **NO guaranteed formula** for blockbuster success, 
    but there are patterns that can increase the likelihood of success.
    """)
    
    st.markdown("---")
    
    st.markdown("### ✅ Success Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎬 What Successful Blockbusters Typically Have:")
        st.markdown("""
        - **High budgets** ($50M+) enabling spectacle and star power
        - **Family-friendly genres** (Animation, Adventure, Family)
        - **Strategic timing** (summer or holiday releases)
        - **Longer runtimes** (120-150 minutes)
        - **High pre-release buzz** (popularity/marketing)
        - **Major studio backing**
        """)
    
    with col2:
        st.markdown("#### Important Caveats:")
        st.markdown("""
        - High variance at all budget levels
        - Exceptional **creative execution** matters
        - **Cultural timing** is crucial
        - **Luck** remains a factor
        - Quality ≠ Commercial success
        """)

    st.markdown("---")

    st.markdown("### 📊 Key Correlations Summary")

    correlations = {
        'Budget → Revenue': 0.71,
        'Popularity → Revenue': 0.40,
        'Runtime → Revenue': 0.25,
        'Rating → Revenue': 0.12
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = list(correlations.keys())
    values = list(correlations.values())
    colors_corr = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
    
    bars = ax.barh(factors, values, color=colors_corr, alpha=0.8, edgecolor='black')
    
    for i, (factor, value) in enumerate(correlations.items()):
        ax.text(value + 0.02, i, f'{value:.2f}', va='center', fontweight='bold')
    
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation Strength: Factors vs Revenue', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.markdown("### 🎬 The Modern Film Industry")
    
    st.info("""
    The industry operates on a **"tentpole" strategy**:
    - Fewer, bigger bets on franchise-ready spectacles
    - Strategic release during peak seasons
    - **78.3% of films are profitable**
    - But only **10% achieve true blockbuster status**
    
    **Bottom line:** Data can improve odds of success, but cannot guarantee it. 
    The magic and risk of cinema lie in that irreducible uncertainty.
    """)
    
    st.markdown("---")

    st.markdown("### About This Project")
    st.write(f"""
    This analysis examined **{len(df):,} films** from The Movies Dataset to understand 
    what factors contribute to box office success. Through comprehensive statistical 
    analysis and visualization, we explored budget, ratings, runtime, popularity, 
    timing, genres, and production origins.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write("The Movies Dataset (Kaggle)")
st.sidebar.write(f"**{len(df):,}** films analyzed")
st.sidebar.markdown("### Project")
st.sidebar.write("Anatomy of a Blockbuster")
st.sidebar.write("Data Science Analysis")