# first, libraries needed for the project are loaded into the environment
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from ast import literal_eval
import json
import math

# -------------------------------------------------- DATA CLEANING AND PREPARATION ------------------------------------------------------------
# DATA LOADING AND INITIAL EXPLORATION:
# ---------------------------------------------------------------------------------------------------------------------------------------------
# loading the dataset
data = pd.read_csv("movies_metadata.csv", low_memory = False, on_bad_lines = "skip")

# visualization of the dataset composizion
data.head(8)
pd.set_option('display.max_columns', 100)

# Exploration of the dataset structure
data.columns.to_list()
data.shape
data.info()
data.describe()

# Clean column names to avoid errors
print(data.columns.tolist())
data.columns = data.columns.str.strip()

# Check for missing values in the dataset
data.isna().sum().sort_values(ascending = False).head(20)

# ---------------------------------------------------------------------------------------------------------------------------------------------
# COLUMN SELECTION:
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Drop columns irrelevant for the blockbuster analysis
columns_to_drop = ['adult','belongs_to_collection', 'homepage','poster_path', 'tagline', 'video', 'spoken_languages', 'backdrop_path',
                   'imdb_id', 'original_title', 'overview', 'status']
data = data.drop(columns = columns_to_drop, errors = 'ignore')

# Verify columns were dropped successfully
data.columns.to_list()

# ---------------------------------------------------------------------------------------------------------------------------------------------
# DATE HANDLING:
# ---------------------------------------------------------------------------------------------------------------------------------------------
# parsing release dates
data ['release_date'] = pd.to_datetime(data['release_date'], errors = 'coerce')

# Drop rows with missing title or release date
data = data.dropna(subset = ['title', 'release_date'])

# extract year from the release date
data['release_year'] = data['release_date'].dt.year

# better to filter unrealistic release years
data = data[(data['release_year'] >= 1900) & (data['release_year'] <= 2025)]

# now, verify the changes made
data.columns.to_list()

# ---------------------------------------------------------------------------------------------------------------------------------------------
# NUMERIC DATA TYPE CONVERSION:
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Convert specified columns to numeric type
numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')

# identify numeric and categorical columns
numeric_columns = data.select_dtypes(include = [np.number]).columns
categorical_columns = data.select_dtypes(exclude = [np.number]).columns

# ---------------------------------------------------------------------------------------------------------------------------------------------
# MISSING VALUE HANDLING:
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Replace zero runtime with NaN (invalid movie length)
data['runtime'] = data['runtime'].replace(0, np.nan)

# Impute missing values with median for reliability metrics
safe_impute = ['runtime', 'popularity', 'vote_average', 'vote_count']
for col in safe_impute:
    data[col] = data[col].fillna(data[col].median())

# Filter extreme runtime values (so only reasonable movie lengths are retained (40-300 minutes))
data = data[(data['runtime'] >= 40) & (data['runtime'] <= 300)]

# Replace zero budget and revenue with NaN
data[['budget', 'revenue']] = data[['budget', 'revenue']].replace(0, np.nan)

# now we remove unrealistic values of budget and revenue (< $1000)
data = data[(data['budget'].isna()) | (data['budget'] >= 1000)]
data = data[(data['revenue'].isna()) | (data['revenue'] >= 1000)]

# now, we verify that there are no more missing values in the dataset. \\
# Budget and revenues Nan are intentionally kept for flexible analysis
missing_check = data.isna().sum()
print("Missing values per column:")
print(missing_check[missing_check > 0])

# -------------------------------------------------------------------------------------------------------------------------------------------
# DUPLICATES REMOVAL:
# -------------------------------------------------------------------------------------------------------------------------------------------
# Checking for duplicate rows
data.duplicated().sum()

# Drop duplicate rows
data = data.drop_duplicates(keep = 'first')

# then we check for duplicates also by title and release_date since some movies can have the same title but be different in release 
# date, which is a common thing that can happen to remakes of old movies and so on.
data.duplicated(subset = ['title', 'release_date']).sum()

# Drop duplicates with the same title and release_date
data = data.drop_duplicates(subset = ['title','release_date'], keep = 'first')

# Verify that no duplicates remain after the cleaning
data.duplicated().sum()
data.duplicated(subset = ['title', 'release_date']).sum()

# -------------------------------------------------------------------------------------------------------------------------------------------
# VOTE COUNT FILTERING:
# -------------------------------------------------------------------------------------------------------------------------------------------
# Retain only movies with sufficient votes (>= 100) for reliable audience rating
data = data [data['vote_count'] >= 100]

# -------------------------------------------------------------------------------------------------------------------------------------------
# JSON COLUMN PARSING:
# -------------------------------------------------------------------------------------------------------------------------------------------
# Parse JSON formatted columns
json_columns = ['genres', 'production_companies', 'production_countries']

for col in json_columns:
    if col in data.columns:
        s = data[col].fillna('[]').astype(str)
        looks_like_list = s.str.strip().str.startswith('[') & s.str.strip().str.endswith(']')
        s = s.where(looks_like_list, '[]')

        parsed = s.apply(lambda x: literal_eval(x) if x.strip().startswith('[') else [])
        data[col] = parsed.apply(lambda v:v if isinstance(v, list) else [])

# Verifying that each column contains now a list after the parse
data[json_columns].apply(lambda x: x.apply(type)).head()

# View content of the parsed JSON columns
data[['genres', 'production_companies', 'production_countries']].head()

# -------------------------------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FROM JSON
# -------------------------------------------------------------------------------------------------------------------------------------------
# Extract names from JSON columns as comma - separated strings
def extract_names(value):
    if not isinstance(value, list) or len(value) == 0:
        return pd.NA
    names = [str(item.get('name', '')) for item in value if isinstance(item, dict) and 'name' in item]
    return ', '.join(names) if names else pd.NA

# create new readable columns with extracted names
data['genres_str'] = data['genres'].apply(extract_names)
data['companies_str'] = data['production_companies'].apply(extract_names)
data['countries_str'] = data['production_countries'].apply(extract_names)

# the primary genre of the movies will be extract for further analysis purposes
data['primary_genre'] = data['genres_str'].apply(lambda x: x.split(', ')[0] if pd.notna(x) else 'Unknown')

# --------------------------------------------------------------------------------------------------------------------------------------------
# ZERO VALUE VERIFICATION:
# --------------------------------------------------------------------------------------------------------------------------------------------
# Check remaining zero values in numeric columns
for col in numeric_columns:
    zero_count = (data[col] == 0).sum()
    zero_percent = 100 * zero_count / len (data)
    print(f"{col}: {zero_count} zeros ({zero_percent:.2f}%)")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# FINAL CLEANED DATASET SUMMARY:
# ---------------------------------------------------------------------------------------------------------------------------------------------
print ("\n" + "="*60)
print("Final cleaned dataset summary")
print("="*60)

# Dataset shape
print(f"\nFinal dataset shape: {data.shape}")

# Missing values summary that were retained for further analysis
print(f"\nBudget missing: {data['budget'].isna().sum()} ({100*data['budget'].isna().sum()/len(data):.1f}%)")
print(f"Revenue missing: {data['revenue'].isna().sum()} ({100*data['revenue'].isna().sum()/len(data):.1f}%)")
print(f"Movies with BOTH budget and revenue: {data[['budget', 'revenue']].notna().all(axis=1).sum()}")

# checking on the quality of the data
print (f"\nData Quality checks:")
print(f"  - Release year range: {data['release_year'].min():.0f} to {data['release_year'].max():.0f}")
print(f"  - Runtime range: {data['runtime'].min():.0f} to {data['runtime'].max():.0f} minutes")
print(f"  - Vote average range: {data['vote_average'].min():.1f} to {data['vote_average'].max():.1f}")
print(f"  - Unique genres: {data['primary_genre'].nunique()}")

# top genres distribution
print (f"\nThe top 5 Primary Genres are:")
print (data['primary_genre'].value_counts().head())

# final columns retained for the analysis
print(f"\nFinal columns retained: {data.columns.to_list()}")

# we are not eliminating outliers (blockbuster films) during the data cleaning process because they can represent legitimate
# box office phenomena (films like Avatar, Titanic ...) and are essential to understanding the film industy's 'blockbuster model
# Removing them would mispresent how the industry actually operates. However, we will analyze the data both with and without outliers
# to understand their impact on our findings


# --------------------------------------------------- FEATURE ENGINEERING ----------------------------------------------------------------------

data['release_date'] = pd.to_datetime(data['release_date'])

# before going on with the analysis of insights ans EDA (exploratory data analysis), feature enginireen are conducted in order to have a more
# complete analysis of the dataset. 6 feature enginireeng are used

# 1) adding financial metrics in order to answer the most important question of this project: "What makes a blockbuster profitable?"
df_financial = data[data[['budget', 'revenue']].notna().all(axis = 1)].copy() 
df_financial['profit'] = df_financial['revenue'] - df_financial['budget']
df_financial['is_profitable'] = df_financial ['profit'] > 0

# now we merge them together
data = data.merge(df_financial[['title', 'release_date', 'profit','is_profitable']],
                  on = ['title', 'release_date'], how = 'left')

# 2) adding temporal features to analyze trends within the data

# for example, we can add "decade" to see the changes over a period of time of the industry
data['release_date'] = pd.to_datetime(data['release_date'])
data['decade'] = (data['release_year'] // 10) * 10

# we can add also season, to see if the release time of the movie matters
seasons_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring', 4: 'Spring', 5: 'Spring',
               6: 'Summer', 7: 'Summer', 8: 'Summer',
               9: 'Fall', 10: 'Fall', 11: 'Fall'}
data['release_month'] = data['release_date'].dt.month
data['release_season'] = data['release_month'].map(seasons_map)

# 3) then, we can add a feature to identify the blockbuster movies
revenue_with_data = data[data['revenue'].notna()].copy()
blockbuster_threshold = revenue_with_data['revenue'].quantile(0.90)
revenue_with_data['is_blockbuster'] = revenue_with_data['revenue'] >= blockbuster_threshold

data = data.merge(revenue_with_data[['title', 'release_date', 'is_blockbuster']],
                  on = ['title', 'release_date'], how = 'left')

print(f"Blockbuster indicator added (threshold: $ {blockbuster_threshold.round(2)})")

# first we drop the original json columns to keep only the readable string versions
data = data.drop(columns = json_columns, errors = 'ignore')

# now we verify the final dataset structure
print(data.columns.to_list())
data[['genres_str', 'companies_str', 'countries_str', 'primary_genre']].dtypes

# then, before starting the analysis, the cleaned data are saved as CSV
data.to_csv('movies_with_features.csv', index = False)

# --------------------------------------------- EDA (Exploratory data analysis) ----------------------------------------------------------------
# first we have to analyze the distribution of the variables within the data through bar plots and density plots to have a general view of the
# data itself

data = pd.read_csv("movies_with_features.csv")
data.head()
data.info()

# variables to use for the histograms 
dist_variables = ['budget', 'revenue', 'profit', 'runtime',
                  'popularity', 'vote_average', 'vote_count',
                  'release_year']

n = len(dist_variables)
cols = 3
rows = math.ceil(n/cols)

fig, axes = plt.subplots(rows, cols, figsize = (18, rows*4))
axes = axes.flatten()

for i, col in enumerate(dist_variables):
    ax = axes[i]
    series = data[col].dropna()

    # use log transformation for heavy right skewed variables
    log_transform = series.min() > 0 and series.skew() > 1.2
    
    if log_transform:
        sns.histplot(series, kde = True, ax = ax, bins = 50, log_scale = 10)
        ax.set_title(f"{col} (log scale)", fontsize = 12, fontweight = 'bold')
    else:
        sns.histplot(series, kde = True, ax = ax, bins = 50)
        ax.set_title(col, fontsize = 12, fontweight = 'bold')
    
    ax.set_xlabel(col, fontsize = 10)
    ax.set_ylabel("Count", fontsize = 10)
    ax.tick_params(axis = 'both', labelsize = 9)

# it is better to hyde any unused subplots 
for j in range (len(dist_variables), len(axes)):
    axes[j].set_visible(False)

# since at first some lables were overlapping in the image, the issue was fixed by adding a pad of 2
plt.tight_layout(pad = 2.0)
plt.show()

# after the histograms, also density plots can be used to show the distribution of the variables
fig, axes = plt.subplots(rows, cols, figsize = (18, rows *4))
axes = axes.flatten()

for i, col in enumerate(dist_variables):
    ax = axes[i]
    series = data[col].dropna()

    #log transform for heavy right skewed variables
    log_transform = series.min() > 0 and series.skew() > 1.2

    if log_transform:
        log_series = np.log10(series)
        sns.kdeplot(log_series, ax = ax, fill = True, color = 'steelblue', alpha = 0.6, linewidth = 2)
        ax.set_title(f"{col} Density (log scale)", fontsize = 12, fontweight = 'bold')
        ax.set_xlabel(f"log10({col})", fontsize = 10)

        # also the median line will be added to the plot
        median_val = np.log10(series.median())
        ax.axvline(median_val, color = 'red', linestyle = '--', linewidth = 1.5, alpha = 0.7, label = 'Median')
    else:
        sns.kdeplot(series, ax = ax, fill = True, color = 'steelblue', alpha = 0.6, linewidth = 2)
        ax.set_title(f"{col} Density", fontsize = 12, fontweight = 'bold')
        ax.set_xlabel(col, fontsize = 10)

        median_val = series.median()
        ax.axvline(median_val, color = 'red', linestyle = '--', linewidth = 1.5, alpha = 0.7, label = 'Median')

    ax.set_ylabel ("Density", fontsize = 10)
    ax.tick_params(axis = 'both', labelsize = 9)
    ax.legend(fontsize = 8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(pad = 2.0)
plt.savefig('density_plots.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#  --------------------------------------------- CORE BLOCKBUSTER ANALYSIS ----------------------------------------------------------------------
# A first question that can be analyzed is the following and its the most important one
# in this project (the most critical relationship)

# 1) DOES SPENDING MORE ON A MOVIE GUARANTEE HIGHER REVENUE? (BUDGET VS REVENUE) ----------------------------------------------------------------

# PIE CHART OF THE DISTRIBUTION OF THE PROFITABLE AND UNPROFITABLE FILMS
# to show this relationship, a scatter plot with a trend line will be used
plt.close('all')

plt.figure(figsize = (12, 8))

clean_data = data[['budget', 'revenue']].dropna()

plt.scatter(clean_data['budget'], clean_data['revenue'], alpha = 0.5, color = 'steelblue', label = 'Movies')

# calculate the trend line
log_budget = np.log10(clean_data['budget'])
log_revenue = np.log10(clean_data['revenue'])

# fit linear regression on log scale
z = np.polyfit(log_budget, log_revenue, 1)
p = np.poly1d(z)

# create the trend line points
budget_range = np.logspace(np.log10(clean_data['budget'].min()),
                           np.log10(clean_data['budget'].max()), 100)
trend_revenue = 10 ** p(np.log10(budget_range))

# now we plot the trend line
plt.plot(budget_range, trend_revenue, color = 'red', linewidth = 2, linestyle = '--', label = f'Trend (slope = {z[0]:.2f})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Budget ($)')
plt.ylabel('Revenue ($)')
plt.title('Budget vs Revenue: is there a formula for success?', fontsize = 14, fontweight = 'bold')
plt.legend(fontsize = 10)
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()

# the scatter plot reveals a strong positive correlation (slope = 0.78) between the budget and the revenue of the movies, indicating that
# higher budgets generally lead to higher revenues. However, the relationship shows diminishing returns, meaning that doubling the budget 
# doesn't automatically double the revenue. The wide scatter underlines the fact that the budget alone doesn't guarantee the succes of the 
# movie. Many high budget films underperform and some low budget ones overperform and becomes suprise hits. This suggest that there is no
# simple formula for blockbuster success. Other factors like quality, timing, and audience appeal metter when considering the success of
# a film. 

# after the plotting is is better to show also these key insights to better understand the relationship already analyzed

# correlation
correlation = clean_data['budget'].corr(clean_data['revenue'])
print (f"The correlation coefficient is: {correlation.round(2)}")      

# now we analyze the budget categories included in the plot
low_budget = clean_data[clean_data['budget'] < 1e6]
mid_budget = clean_data[(clean_data['budget'] >= 1e6) & (clean_data['budget'] < 5e7)]
high_budget = clean_data[clean_data['budget'] >= 5e7]

print(f"\nLow budget films (<$1M): {len(low_budget)}, have an average revenue of: ${low_budget['revenue'].mean()/1e6} M")
print(f"Mid budget films ($1M - $50M): {len(mid_budget)}, average revenue: ${mid_budget['revenue'].mean()/1e6} M")
print(f"High budget films (> $50M): {len(high_budget)}, average revenue: ${high_budget['revenue'].mean()/1e6} M")

# before going on with the next analysis is it better to understand the distribution of profitable and unprofitable films in the dataset
# just to have a clearer view of the dataset structure for this part

# first we visualize thorugh a pie chart the proportions of profitable and non profitable films present in the dataset
# just to add comprehend better this part of the dataset
plt.close('all')

plt.figure(figsize = (10, 8))

# data that we will use for plotting also in the next analysis
clean_profit_data = data[['budget', 'profit']].dropna()

# prepare the data
profitability_counts = clean_profit_data['profit'].apply(lambda x: 'Profitable' if x > 0 else 'Unprofitable').value_counts()

# now we create the pie chart
colors = ['lightblue', 'lightcoral']
explode = (0.05, 0)     # this separates slightly the profitable slice of the pie chart

plt.pie(profitability_counts, labels = profitability_counts.index,
        autopct = '%1.1f%%',
        startangle = 90,
        colors = colors,
        explode = explode,
        textprops = {'fontsize': 12, 'fontweight': 'bold'})

plt.title('Film profitability distribution', fontsize = 14, fontweight = 'bold', pad = 20)

# add count information as text
total = profitability_counts.sum()
profitable_count = profitability_counts.get('Profitable', 0)
unprofitable_count = profitability_counts.get('Unprofitable', 0)

plt.text(0, -1.3, f"Total films: {total} \nProfitable: {profitable_count} | Unprofitable: {unprofitable_count}",
         ha = 'center', fontsize = 12, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))

plt.tight_layout()
plt.show()

# since we are searching for the blockbuster formula, in the next analysis we will focus only on profitable films

# 2) DOES THE BUDGET REALLY INFLUENCE THE SUCCESS OF A FILM? (BUDGET VS PROFIT) -----------------------------------------------------------------

plt.close('all')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# LEFT PLOT: budget vs profit scatter plot

# we will analyze only profitable films for better visualization
profitable = clean_profit_data[clean_profit_data['profit'] > 0]

ax1.scatter(profitable['budget'], profitable['profit'],
            alpha = 0.5, color = 'seagreen', label = 'Profitable Movies', s = 20)

# add trend line to the profit
log_budget_p = np.log10(profitable['budget'])
log_profit = np.log10(profitable['profit'])

z_profit = np.polyfit(log_budget_p, log_profit, 1)
p_profit = np.poly1d(z_profit)

budget_range_p = np.logspace(np.log10(profitable['budget'].min()),
                             np.log10(profitable['budget'].max()), 100)

trend_profit = 10 ** p_profit(np.log10(budget_range_p))

ax1.plot(budget_range_p, trend_profit, color = "red", linewidth = 2,
         linestyle = '--', label = f"Trend (slope = {z_profit[0]:.2f})")


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Budget ($)', fontsize = 12)
ax1.set_ylabel('Profit ($)', fontsize = 12)
ax1.set_title('Budget vs Profit: Investment returns in filmmaking industry?',
              fontsize = 14, fontweight = 'bold')

ax1.legend(fontsize = 10, loc = 'upper left')
ax1.grid(True, alpha = 0.3)

# exlude extreme negatives for better visualization
ax1.set_ylim(1e3, 1e9)

# RIGHT PLOT: Average profit by budget category (bar plot)
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

# Calculate the average profit for each category
category_order = ['Low\n(<$1M)', 'Mid\n($1M - $50M)', 'High\n(>$50M)']
avg_profit_by_category = profit_df.groupby('Budget Category')['Profit ($)'].mean() /1e6
avg_profit_by_category = avg_profit_by_category.reindex(category_order)

# now we create the barplot
colors = ['lightcoral', 'lightblue', 'lightgreen']
bars = ax2.bar(category_order, avg_profit_by_category,
               color = colors, alpha = 0.8,
               edgecolor = 'black', linewidth = 1.5)

# then, we add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f"${height:.1f}M",
             ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
ax2.set_ylabel ('Average profit ($ Millions)', fontsize = 12)
ax2.set_xlabel ('Budget category', fontsize = 12)
ax2.set_title ('Average Profit by budget category', fontsize = 14, fontweight = 'bold')
ax2.legend (fontsize = 10)
ax2.grid(True, alpha = 0.3, axis = 'y')

plt.tight_layout(pad = 2.0)
plt.show()

# The analysis reveals that higher budgets increase profit with diminishing returns (slope = 0.60). High - budget
# films average $177.9 M profit compared to $47.1 M for mid - budget and $19.7 M for low - budget films. However, the
# wide scatter demonstrates that budget alone doesn't guarantee success - creative and timing matter equally.

# 3) DO BETTER RATED MOVIES MAKE MORE MONEY (VOTE AVERAGE (RATING) VS REVENUE) ------------------------------------------------------------------

plt.close('all')

 # so, first we prepare the data for the visualization
rating_revenue = data[['vote_average', 'revenue']].dropna()

# identify blockbusters (high revenue films) for visualization
Q1 = rating_revenue['revenue'].quantile(0.25)
Q3 = rating_revenue['revenue'].quantile(0.75)
IQR = Q3 - Q1
blockbuster_threshold = Q3 + 1.5 * IQR

rating_revenue['is_blockbuster'] = rating_revenue['revenue'] > blockbuster_threshold

# this is be the function that categorize the ratings
def categorize_ratings(rating):
    if rating < 5.0:
        return 'Poor\n(<5.0)'
    elif rating < 6.5:
        return 'Average\n(5.0 - 6.5)'
    elif rating < 7.5:
        return 'Good\n(6.5 - 7.5)'
    else:
        return 'Excellent\n(>= 7.5)'
    
# visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# LEFT PLOT: Scatterplot 
ax1.scatter(rating_revenue['vote_average'],
            rating_revenue['revenue'],
            alpha = 0.5,
            color = 'steelblue',
            s = 50,
            edgecolors = 'white',
            linewidth = 1,
            label = 'All movies')

# now we highlight blockbusters
blockbusters = rating_revenue[rating_revenue['is_blockbuster']]
ax1.scatter(blockbusters['vote_average'],
            blockbusters['revenue'],
            alpha = 0.5,
            color = 'red',
            s = 50,
            edgecolors = 'darkred',
            linewidth = 1,
            marker = '*',
            label = f"Blockbuster (n = {len(blockbusters)})")

# then we add a trend line inside the scatter plot
z = np.polyfit(rating_revenue['vote_average'],
                   np.log10(rating_revenue['revenue']), 1)
p_all = np.poly1d(z)

vote_range = np.linspace(0, 10, 100)
trend = 10 ** p(vote_range)

ax1.plot(vote_range, trend,
         color = 'darkgreen', linewidth = 2.5, linestyle = '--',
         label = f"Trend (slope = {z[0]:.2f})")

ax1.set_yscale('log')
ax1.set_xlabel('Vote Average (Rating)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Rating vs revenue: Does quality drive box office success?', fontsize = 13, fontweight = 'bold')
ax1.set_xlim(0, 10)
ax1.legend(fontsize = 10, loc = 'lower right')
ax1.grid(True, alpha = 0.3)

# calculate and display correlation
correlation = rating_revenue['vote_average'].corr(rating_revenue['revenue'])
ax1.text(0.05, 0.95, f"Correlation: {correlation:.3f}\n = {len(rating_revenue)}",
         transform = ax1.transAxes, fontsize = 11,
         verticalalignment = 'top',
         bbox = dict(boxstyle = 'round', facecolor = 'lightblue', alpha = 0.8))

# RIGHT PLOT:  BAR PLOT about average revenue by category --------
rating_revenue['category'] = rating_revenue['vote_average'].apply(categorize_ratings)
category_order = ['Poor\n(<5.0)', 'Average\n(5.0 - 6.5)', 'Good\n(6.5 - 7.5)', 'Excellent\n(>= 7.5)']

avg_revenue_all = rating_revenue.groupby('category')['revenue'].mean() / 1e6
avg_revenue_all = avg_revenue_all.reindex(category_order)

colors = ['red', 'orange', 'yellow', 'green']
bars = ax2.bar(category_order, avg_revenue_all, color = colors, alpha = 0.8,
               edgecolor = 'black', linewidth = 1.5)

# add the labels of the values on the bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.1f}M',
             ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')

ax2.set_xlabel('Rating Category', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax2.set_title('Average revenue by Rating category', fontsize = 13, fontweight = 'bold')

fig.suptitle('Rating vs Revenue (data with outliers)',
             fontsize = 16, fontweight = 'bold', y = 0.98)

plt.tight_layout(rect = [0.02, 0, 1, 0.96])
plt.show()

# RESULTS:
# The weak positive correlation reveals that while better rated films earn more on average, quality alone doesn't
# guarantee box office success. The bar chart shows progressive revenue increases across rating
# categories, but blockbusters require factors beyond critical acclaim - marketing, timing, and audience appeal matter
# equally.

# 4) IS THERE AN OPTIMAL MOVIE LENGTH FOR BOX OFFICE SUCCESS? -----------------------------------------------------------------------------------

plt.close ('all')

# prepare the clean data for plotting
runtime_revenue = data [['runtime', 'revenue']].dropna()

# we identify the blockbusters
blockbuster_threshold = runtime_revenue['revenue'].quantile(0.90)
runtime_revenue['is_blockbuster'] = runtime_revenue['revenue'] >= blockbuster_threshold

print (f"\nBlockbusters (top 10% revenue): {runtime_revenue['is_blockbuster'].sum()}")
print (f"Blockbuster threshold: ${blockbuster_threshold/1e6:.1f}M")

# create runtime categories
def categorize_runtime(runtime):
    if runtime < 90:
        return 'Short\n(<90 min)'
    elif runtime < 120:
        return 'Standard\n(90 - 120 min)'
    elif runtime < 150:
        return 'Long\n(120 - 150 min)'
    else:
        return 'Epic\n(>=150 min)'
    
runtime_revenue['runtime_category'] = runtime_revenue['runtime'].apply(categorize_runtime)

# FIRST THE SCATTERPLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

regular_movies = runtime_revenue[~runtime_revenue['is_blockbuster']]
blockbusters = runtime_revenue[runtime_revenue['is_blockbuster']]

ax1.scatter(regular_movies['runtime'],
            regular_movies['revenue'],
            alpha = 0.5,
            color = 'steelblue',
            s = 50,
            edgecolor = 'white',
            linewidth = 0.5,
            label = 'All movies')

# now we highlight the blockbusters in the plot
ax1.scatter(blockbusters['runtime'],
            blockbusters['revenue'],
            alpha = 0.7,
            color = 'red',
            s = 100,
            edgecolors = 'darkred',
            linewidth = 1,
            marker = '*',
            label = f"Blockbusters (n = {len(blockbusters)})")

# add again a trend line as we did before with the other scatter plots
z_runtime = np.polyfit(runtime_revenue['runtime'],
                       np.log10(runtime_revenue['revenue']), 1)
p_runtime = np.poly1d(z_runtime)
runtime_range = np.linspace(runtime_revenue['runtime'].min(),
                            runtime_revenue['runtime'].max(), 100)

trend_runtime = 10 ** p_runtime(runtime_range)

ax1.plot(runtime_range, trend_runtime,
         color = 'darkgreen', linewidth = 2.5, linestyle = '--',
         label = f"Trend (slope = {z_runtime[0]:.3f})")

# vertical lines for categories boundaries
ax1.axvline(x = 90, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)
ax1.axvline(x = 120, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)
ax1.axvline(x = 150, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)

ax1.set_yscale ('log')
ax1.set_xlabel('Runtime (minutes)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Scatter plot: Runtime vs Revenue', fontsize = 13, fontweight = 'bold')
ax1.grid(True, alpha = 0.3)

# correlation
correlation_runtime = runtime_revenue['runtime'].corr(runtime_revenue['revenue'])
ax1.text(0.05, 0.05, f'Correlation: {correlation_runtime:.3f}\n = {len(runtime_revenue)}',
         transform = ax1.transAxes, fontsize = 11,
         verticalalignment = 'bottom',
         horizontalalignment = 'center',
         bbox = dict(boxstyle = 'round', facecolor = 'lightyellow', alpha = 0.9, edgecolor = 'black', linewidth = 1.5))

print (correlation_runtime)

ax1.legend(fontsize = 10, loc = 'upper left')

# now the bar plot
category_order = ['Short\n(<90 min)', 'Standard\n(90 - 120 min)', 'Long\n(120 - 150 min)', 'Epic\n(>=150 min)']
avg_revenue_runtime = runtime_revenue.groupby('runtime_category')['revenue'].mean() / 1e6
avg_revenue_runtime = avg_revenue_runtime.reindex(category_order)

colors_runtime = ['blue', 'violet', 'purple', 'pink']
bars = ax2.bar(category_order, avg_revenue_runtime, color = colors_runtime, alpha = 0.8,
               edgecolor = 'black', linewidth = 1.5)

# bar's labels 
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.1f}M',
             ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
ax2.set_xlabel('Runtime Category', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax2.set_title('Bar Plot: Average Revenue by Runtime', fontsize = 13, fontweight = 'bold')
ax2.grid(True, alpha = 0.3, axis = 'y')

# title 
fig.suptitle('Runtime vs Revenue: Does Movie length affects box office success?',
             fontsize = 16, fontweight = 'bold', y = 0.98)

plt.tight_layout(rect = [0.02, 0, 1, 0.96])
plt.show()

# now, we go on with the statistical analysis
# correlation
if correlation_runtime < 0.3:
    strength = "weak"
elif correlation_runtime < 0.7:
    strength = "moderate"
else:
    strength = "strong"

print (f"This indicates a {strength} positive relationship")

# now, we analyze the optimal runtime based on the data we have by analyzing the revenue by 15 minute bins
runtime_bins = pd.cut(runtime_revenue['runtime'],
                      bins = range (40, 305, 15),
                      labels = [f"{i} - {i+15}" for i in range (40, 290, 15)])
runtime_revenue['runtime_bin'] = runtime_bins

bin_analysis = runtime_revenue.groupby('runtime_bin', observed = True)['revenue'].agg(['mean', 'median', 'count'])
bin_analysis = bin_analysis[bin_analysis['count'] >= 20]
bin_analysis_sorted = bin_analysis.sort_values('mean', ascending = False)

# top 5 most profitable runtime ranges
for i, (bin_name, row) in enumerate (bin_analysis_sorted.head().iterrows(), 1):
    print (f"{i}. {bin_name} min: Mean = ${row['mean']/1e6:>6,.1f}M, Median = ${row['median']/1e6:>6,.1f}M, n = {int(row['count']):>3}")

# bottom 5 lowest revenue runtime ranges
for i, (bin_name, row) in enumerate (bin_analysis_sorted.tail().iterrows(), 1):
    print (f"{i}. {bin_name} min: Mean = ${row['mean']/1e6:>6,.1f}M, Median = ${row['median']/1e6:>6,.1f}M, n = {int(row['count']):>3}")

# now we analyze the blockbuster patterns about runtime
blockbuster_runtimes = runtime_revenue[runtime_revenue['is_blockbuster']]['runtime']
regular_runtimes = runtime_revenue[~runtime_revenue['is_blockbuster']]['runtime']

# so now we do a runtime comparison
print(f"\nBlockbuster films (top 10%):")
print(f"Mean runtime: {blockbuster_runtimes.mean().round()} minutes")
print(f"Median runtime: {blockbuster_runtimes.median().round()} minutes")
print(f"Range: {blockbuster_runtimes.min()} - {blockbuster_runtimes.max()} minutes")

# now we do the same comparison also for regular films
print(f"\nRegular films:")
print(f"Mean runtime: {regular_runtimes.mean().round()} minutes")
print(f"Median runtime: {regular_runtimes.median().round()} minutes")
print(f"Range: {regular_runtimes.min()} - {regular_runtimes.max()} minutes")

# now we calculate also the revenues of the films by category of runtime
# - SHORT RUNTIME FILMS: They have an average revenue of 73 Million dollars
short_avg= runtime_revenue[runtime_revenue['runtime_category'] == 'Short\n(<90 min)']['revenue'].mean()/1e6
print(f"The average revenue for films of short runtime is: $ {short_avg.round()} M")

# - STANDARD RUNTIME FILMS: They have an average revenue of 88 Million dollars
standard_avg= runtime_revenue[runtime_revenue['runtime_category'] == 'Standard\n(90 - 120 min)']['revenue'].mean()/1e6
print(f"The average revenue for films of standard runtime is: $ {standard_avg.round()} M")

# - LONG RUNTIME FILMS: They have an average revenue of 172 Million dollars
long_avg= runtime_revenue[runtime_revenue['runtime_category'] == 'Long\n(120 - 150 min)']['revenue'].mean()/1e6
print(f"The average revenue for films of long runtime is: $ {long_avg.round()} M")

# - EPIC RUNTIME FILMS: They have an average revenue of 245 Million dollars
epic_avg= runtime_revenue[runtime_revenue['runtime_category'] == 'Epic\n(>=150 min)']['revenue'].mean()/1e6
print(f"The average revenue for films of epic runtime is: $ {epic_avg.round()} M")

# now in addition we print out also the names of the 5 longest movies in the dataset (highest runtimes) and of the 
# bottom 5 shortest films by runtime

runtime_with_titles = data[['title', 'runtime', 'revenue']].dropna()

# - TOP 5 LONGEST FILMS
longest_movies = runtime_with_titles.nlargest(5, 'runtime')
for i, (idx, row) in enumerate(longest_movies.iterrows(), 1):
    print (f"{i}. {row['title']}")
    print (f"Runtime: {row['runtime']:.0f} minutes | Revenue: ${row['revenue']/1e6:.1f}M")

# - BOTTOM 5 SHORTEST MOVIES
shortest_movies = runtime_with_titles.nsmallest(5, 'runtime')
for i, (idx, row) in enumerate(shortest_movies.iterrows(), 1):
    print(f"{i}. {row['title']}")
    print(f"Runtime: {row['runtime']:.0f} minutes | Revenue: ${row['revenue']/1e6:.1f}M")

# INSIGHTS 
# So, as it can be seen from the analysis, in general films with higher runtimes tend to have higher revenues and so films that become blockbusters
# are usually the ones that have a longer duration. From the correlation analysis, we see that there is a positive correlation between the runtime of films
# and the revenue but this correlation is not so strong, meaning that is only one of the factors that contributes to make a blockbuster but not the only one as 
# we have seen in previous analysis.

# 5) DOOES HIGHER POPULARITY TRANSLATE TO HIGHER REVENUE? (POPULARITY VS REVENUE) ----------------------------------------------------------------
plt.close('all')

# prepare as always the data for the plotting
popularity_revenue = data[['popularity', 'revenue']].dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# identify blockbusters using the same approach as before
blockbuster_threshold = popularity_revenue['revenue'].quantile(0.90)
popularity_revenue['is_blockbuster'] = popularity_revenue['revenue'] >= blockbuster_threshold

print(f"\nBlockbusters (top 10% revenue): {popularity_revenue['is_blockbuster'].sum()}")
print(f"Blockbusters threshold: ${blockbuster_threshold/1e6:.1f}M")

# popularity categories
def categorize_popularity (popularity):
    if popularity < 5:
        return 'Low\n(<5)'
    elif popularity < 15:
        return 'Medium\n(5 - 15)'
    elif popularity < 30:
        return 'High\n(15 - 30)'
    else:
        return 'Very High\n(>=30)'
    
popularity_revenue['popularity_category'] = popularity_revenue['popularity'].apply(categorize_popularity)

# SCATTER PLOT
regular_movies = popularity_revenue[~popularity_revenue['is_blockbuster']]
blockbusters = popularity_revenue[popularity_revenue['is_blockbuster']]

ax1.scatter(regular_movies['popularity'],
            regular_movies['revenue'],
            alpha = 0.5,
            color = 'steelblue',
            s = 50,
            edgecolor = 'white',
            linewidth = 0.5,
            label = 'All movies')

# highlight blockbusters
ax1.scatter(blockbusters['popularity'],
            blockbusters['revenue'],
            alpha = 0.7,
            color = 'orange',
            s = 100,
            edgecolors = 'darkred',
            linewidth = 1,
            marker = '*',
            label = f"Blockbusters (n = {len(blockbusters)})")

# trend line
z_pop = np.polyfit(np.log10(popularity_revenue['popularity'] + 1),
                   np.log10(popularity_revenue['revenue']), 1)
p_pop = np.poly1d(z_pop)

pop_range = np.logspace(np.log10(popularity_revenue['popularity'].min() + 1),
                        np.log10(popularity_revenue['popularity'].max() + 1), 100)

trend_pop = 10 ** p_pop(np.log10(pop_range))

ax1.plot(pop_range, trend_pop, color = 'darkgreen',
         linewidth = 2.5, linestyle = '--',
         label = f"Trend (slope = {z_pop[0]:.2f})")

# vertical lines for category boundaries
ax1.axvline(x = 5, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)
ax1.axvline(x = 15, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)
ax1.axvline(x = 30, color = 'orange', linestyle = ':', linewidth = 1.5, alpha = 0.6)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Popularity score', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Scatter plot: Popularity vs Revenue', fontsize = 13, fontweight = 'bold')
ax1.grid(True, alpha = 0.3)

# correlation
correlation_pop = popularity_revenue['popularity'].corr(popularity_revenue['revenue'])
ax1.text(0.05, 0.95, f"Correlation: {correlation_pop:.3f}\n = {len(popularity_revenue)}",
         transform = ax1.transAxes, fontsize = 11,
         verticalalignment = 'top',
         bbox = dict(boxstyle = 'round', facecolor = 'lightblue', alpha = 0.9,
                     edgecolor = 'black', linewidth = 1.5))

ax1.legend(fontsize = 10, loc = 'lower right')

# BAR PLOT
category_order = ['Low\n(<5)', 'Medium\n(5 - 15)', 'High\n(15 - 30)', 'Very High\n(>=30)']
avg_revenue_pop = popularity_revenue.groupby('popularity_category')['revenue'].mean() / 1e6
avg_revenue_pop = avg_revenue_pop.reindex(category_order)

colors_pop = ['lightcoral', 'orange', 'yellow', 'green']
bars = ax2.bar(category_order, avg_revenue_pop, color = colors_pop, alpha = 0.8,
               edgecolor = 'black', linewidth = 1.5)

# bar plot labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"${height:.1f}M", ha = 'center', va = 'bottom', fontsize = 12, fontweight = 'bold')

ax2.set_xlabel('Popularity Category', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Average Revenue ($ Millions)', fontsize = 12,fontweight = 'bold')
ax2.set_title('Bar Plot: Average Revenue by popularity', fontsize = 13, fontweight = 'bold')
ax2.grid(True, alpha = 0.3, axis = 'y')

# figure title
fig.suptitle('Popularity vs Revenue: Does Buzz Equal box office success?',
             fontsize = 16, fontweight = 'bold', y = 0.98)

plt.tight_layout(rect = [0.02, 0, 1, 0.96])
plt.show()

# INSIGHTS
# Unlike previous analysis of budget, rating, and runtime, popularity represents a fundamentally different metric that captures marketing reach,
# social media buzz, star power and audience anticipation before and during a film's release.
# FINDINGS:
# - the correlation coefficient of 0.401 represents a moderate positive relationship between popularity and revenue, in respect of the previous ones. So, popularity for now,
#   is one of the strongest single predictor of commercial success among the variables aalyzed. 

# THE BLOCKBUSTER CONCENTRATION EFFECT
# - the scatter plot visualzation reveals that blockbusters (orange stars) are heavily concentrated in the high popularity regions of the plot.
#   this concentration pattern indicates that popularity is not just correlated with revenue, its practically mostly a prerequisite for blockbuster success.
# IMPLICATION:
# Studios seeking blockbuster returns must invest in generating pre - release buzz and mantaining high visibility throughout the theatrical run. 
# But the scatter also shows great variations meaning that while there is a mild/strong correlation between popularity and revenue, it is not always the case,
# so, the blockbuster effect is not always guaranteed.

# ----------------------------------------------------------- TEMPORAL PATTERNS -----------------------------------------------------------------
# 6) DOES THE RELEASE YEAR AFFECTS THE SUCCESS OF A FILM AND HOW HAS THE FILM INDUSTRY EVOLVED OVER TIME? ---------------------------------------
plt.close('all')

# prepare the data before plotting
decade_data = data[['decade', 'revenue', 'budget', 'vote_average', 'is_blockbuster']].dropna()

# now, we filter for decades with sufficient data (50 in our case)
decade_counts = decade_data['decade'].value_counts()
valid_decades = decade_counts[decade_counts >= 50].index
decade_data = decade_data[decade_data['decade'].isin(valid_decades)]

# decade statistics
decade_stats = decade_data.groupby('decade').agg({
    'revenue': ['mean', 'count'],
    'budget': 'mean',
    'vote_average': 'mean',
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0}).reset_index()

decade_stats.columns = ['decade', 'avg_revenue', 'count', 'avg_budget', 'avg_rating', 'blockbuster_rate']
decade_stats['avg_revenue'] = decade_stats['avg_revenue'] / 1e6
decade_stats['avg_budget'] = decade_stats['avg_budget'] / 1e6

# VISUALIZATION:
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots (2, 2, figsize = (14, 9))
decade_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(decade_stats)))

# PLOT 1: Average revenue by Decade
bars1 = ax1.bar(decade_stats['decade'].astype(str), decade_stats['avg_revenue'],
                color = decade_colors, alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f"${height:.0f}M", ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')
    

ax1.set_xlabel ('Decade', fontsize = 11, fontweight = 'bold')
ax1.set_ylabel ('Average revenue ($ Millions)', fontsize = 11, fontweight = 'bold')
ax1.set_title ('Revenue evolution by decade', fontsize = 11, fontweight = 'bold')
ax1.tick_params(axis = 'x', rotation = 45, labelsize = 9)
ax1.tick_params(axis = 'y', labelsize = 9)
ax1.grid(True, alpha = 0.3, axis = 'y')

# PLOT 2: Average budget by decade
bars2 = ax2.bar(decade_stats['decade'].astype(str), decade_stats['avg_budget'],
                color = decade_colors, alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"${height:.0f}M", ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')

ax2.set_xlabel('Decade', fontsize = 11, fontweight = 'bold')
ax2.set_ylabel('Average Budget ($ Millions)', fontsize = 11, fontweight = 'bold')
ax2.set_title('Budget growth by decade', fontsize = 11, fontweight = 'bold', pad = 15)
ax2.tick_params(axis = 'x', rotation = 45, labelsize = 9)
ax2.tick_params(axis = 'y', labelsize = 9)
ax2.grid(True, alpha = 0.3, axis = 'y')

# PLOT 3: Film quality by decade
bars3 = ax3.bar(decade_stats['decade'].astype(str), decade_stats['avg_rating'],
                color = decade_colors, alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f"{height:.2f}", ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')
    
ax3.set_xlabel('Decade', fontsize = 11, fontweight = 'bold')
ax3.set_ylabel('Average Rating', fontsize = 11, fontweight = 'bold')
ax3.set_title('Film quality by decade', fontsize = 11, fontweight = 'bold', pad = 15)
ax3.tick_params(axis = 'x', rotation = 45, labelsize = 9)
ax3.tick_params(axis = 'y', labelsize = 9)
ax3.set_ylim(5.5, 7.6)
ax3.grid(True, alpha = 0.3, axis = 'y')

# PLOT 4: Blockbuster rate by decade
bars4 = ax4.bar(decade_stats['decade'].astype(str), decade_stats['blockbuster_rate'],
                color = decade_colors, alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars4:
    height = bar.get_height ()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f"{height:.1f}%", ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')

ax4.set_xlabel('Decade', fontsize = 11, fontweight = 'bold')
ax4.set_ylabel('Blockbuster rate (%)', fontsize = 11, fontweight = 'bold')
ax4.set_title('Blockbuster concentration by decade', fontsize = 11, fontweight = 'bold', pad = 15)
ax4.tick_params(axis = 'x', rotation = 45, labelsize = 9)
ax4.tick_params(axis = 'y', labelsize = 9)
ax4.grid(True, alpha = 0.3, axis = 'y')

fig.suptitle('Industry evolution: How the blockbuster formula changed over time',
             fontsize = 13, fontweight = 'bold', y = 0.98)

# udjust the settings of the final image for better control spacing 
plt.subplots_adjust(left = 0.08, right = 0.96, top = 0.92, bottom = 0.08, hspace = 0.45, wspace = 0.25)
plt.show()

# INSIGHTS:
# - Revenue evolution by decade: average film revenue has grown consistently from $46M in the 1960s to $149M in the 2010s,
#   representing almost a 224% increase. The most dramatic growth occurred from 1990 onwards, with 2010s showig the highest average
#   revenue, reflecting the dominance of blockbuster tentpole films

# - Budget growth by decade: production budgets have exploaded from $6M in the 1960s to $48M in the 2010s. The steepest rise occurred 
#   between the 1980s and 1990s, driven by increased special effects costs and star salaries

# - Film quality by decade: average ratings have remained remarkably stable, hovering between 6.26 - 7.04, with older films scoring slightly
#   higher than modern releases. This stability suggests that increased commercialization hasn't compromised film quality, though it indicates critics
#   and audiences rate modern blockbusters slightly lower than classic cinema

# - Blockbuster concentration by decade: The blockbuster rate has surged from just 1.4% in the 1960s to 15.5% in the 2010s. This concentration demonstrates 
#   the industry's shift from diverse mid budget filmmaking to a hits driven model where fewer films capture most of the revenue

# 7) DOES RELEASE TIME MATTER FOR BOX OFFICE SUCCESS? -------------------------------------------------------------------------------------------
plt.close('all')

# data for plot
seasonal_data = data[['release_season', 'release_month', 'revenue', 'is_blockbuster']].dropna()

# create the figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots (2, 2, figsize = (18, 12))

# seasonal statistics
season_stats = seasonal_data.groupby('release_season').agg({
    'revenue': ['mean', 'median', 'count'],
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()

season_stats.columns = ['season', 'avg_revenue', 'median_revenue', 'count', 'blockbuster_rate']
season_stats['avg_revenue'] = season_stats['avg_revenue'] / 1e6
season_stats['median_revenue'] = season_stats['median_revenue'] / 1e6

# monthly statistics
monthly_stats = seasonal_data.groupby('release_month').agg({
    'revenue': 'mean',
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()

monthly_stats.columns = ['month', 'avg_revenue', 'blockbuster_rate']
monthly_stats['avg_revenue'] = monthly_stats['avg_revenue'] / 1e6

# month names for better visualization
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[int(x) - 1])

# Release time analysis

# PLOT 1: Average revenue by season
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
season_colors = {'Winter': 'blue', 'Spring': 'lightgreen', 'Summer': 'orange', 'Fall': 'brown'}
season_stats_ordered = season_stats.set_index('season').reindex(season_order).reset_index()

bars1 = ax1.bar(season_stats_ordered['season'], season_stats_ordered['avg_revenue'],
                color = [season_colors[s] for s in season_stats_ordered['season']],
                alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f"${height:.1f}M", ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
ax1.set_xlabel('Release Season', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Average Revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Average Revenue by release Season', fontsize = 13, fontweight = 'bold')
ax1.grid(True, alpha = 0.3, axis = 'y')

# PLOT 2: Blockbuster rate by season
bars2 = ax2.bar(season_stats_ordered['season'], season_stats_ordered['blockbuster_rate'],
                color = [season_colors[s] for s in season_stats_ordered['season']],
                alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"{height:.1f}%", ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
ax2.set_xlabel('Release Season', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Blockbuster Rate (%)', fontsize = 12, fontweight = 'bold')
ax2.set_title('Blockbuster concentration by season', fontsize = 13, fontweight = 'bold')
ax2.grid(True, alpha = 0.3, axis = 'y')

# PLOT 3: Average revenue by month
monthly_revenue_clean = monthly_stats[monthly_stats['avg_revenue'] > 0].copy()

bars3 = ax3.bar(monthly_stats['month_name'], monthly_stats['avg_revenue'],
                color = 'steelblue', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

max_month_idx = monthly_stats['avg_revenue'].idxmax()
max_month_pos = monthly_revenue_clean.index.get_loc(max_month_idx)
bars3[max_month_idx].set_color('yellow')
bars3[max_month_idx].set_edgecolor('orange')
bars3[max_month_idx].set_linewidth(2.5)

for i, (idx, row) in enumerate(monthly_revenue_clean.iterrows()):
    height = row['avg_revenue']
    ax3.text(i, height,
             f"{height:.0f}M", ha = 'center', va = 'bottom', fontsize = 8, fontweight = 'bold')
    
ax3.set_xlabel('Release Month', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax3.set_title('Average revenue by month (Peak in yellow)', fontsize = 13, fontweight = 'bold')
ax3.tick_params(axis = 'x', rotation = 45, labelsize = 10)
ax3.grid(True, alpha = 0.3, axis = 'y')
ax3.set_ylim(0, max(monthly_stats['avg_revenue']) * 1.15)

# PLOT 4: Film count by season (to show the release strategy of the film industry)
bars4 = ax4.bar(season_stats_ordered['season'], season_stats_ordered['count'],
                color = [season_colors[s] for s in season_stats_ordered['season']],
                alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f"{int(height)}", ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
ax4.set_xlabel('Release season', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Number of films released', fontsize = 12, fontweight = 'bold')
ax4.set_title('Release volume by season', fontsize = 13, fontweight = 'bold')

fig.suptitle('Release timing strategy: Does when you release a film matter?',
             fontsize = 16, fontweight = 'bold', y = 0.995)

plt.tight_layout(rect = [0, 0.02, 1, 0.985], h_pad = 3.5, w_pad = 2.5)
plt.show()

# Key statistics of this analysis
# BEST SEASON REVENUE
best_season = season_stats_ordered.loc[season_stats_ordered['avg_revenue'].idxmax()]
worst_season = season_stats_ordered.loc[season_stats_ordered['avg_revenue'].idxmin()]

print (f"\nBest season: {best_season['season']} (${best_season['avg_revenue']:.1f}M) avg, {best_season['blockbuster_rate']:.1f}% blockbuster rate")
print (f"\nWorst season: {worst_season['season']} (${worst_season['avg_revenue']:.1f}M) avg, {worst_season['blockbuster_rate']:.1f}% blockbuster rate")

# BEST MONTH FOR REVENUE
best_month = monthly_stats.loc[monthly_stats['avg_revenue'].idxmax()]
print (f"\nPeak month: {best_month['month_name']} (${best_month['avg_revenue']:.1f}M avg, {best_month['blockbuster_rate']:.1f}% blockbuster rate)")

# RELEASE VOLUME PATTERN
most_releases = season_stats_ordered.loc[season_stats_ordered['count'].idxmax()]
print (f"\nMost releases: {most_releases['season']} ({int(most_releases['count'])} films)")

# INSIGHTS
# Release timing significantly impact box office performance. Summer consistently dominates as a blockbuster season, with studios strategically releasing their biggest films during school vacation periods when families and teenagers have maximum availability. 
# The 'summer blockbuster' phenomenon is real, these months generate substantially higher revenues and blockbuster rates than other season. Winter (holiday season) typically ranks second, capitalizing on Christmas
# and New Year audiences. Spring and Fall serve as "dump months" for lower budget films and counter programming, showing notably lower revenues. This seasonal pattern drives studios' release calendars and explains why tentpole films are rarely released 
# in February or September

# 8) DOES THE GENRE OF A FILM INFLUENCE IT'S SUCCESS AND WHICH GENRES DOMINATE THE BOX OFFICE? --------------------------------------------------
plt.close('all')

# data for plotting
genre_data = data[['primary_genre', 'revenue', 'popularity', 'is_blockbuster', 'vote_average']].dropna()

# genre statistics
genre_stats = genre_data.groupby('primary_genre').agg({
    'revenue': ['mean', 'count'],
    'popularity': 'mean',
    'vote_average': 'mean',
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()

genre_stats.columns = ['genre', 'avg_revenue', 'count', 'avg_popularity', 'avg_rating', 'blockbuster_rate']
genre_stats['avg_revenue'] = genre_stats['avg_revenue'] / 1e6

# filtering genres with at least 50 films for statistical significance
genre_stats = genre_stats[genre_stats['count'] >= 50].copy()

# sorting by average revenue for better visualization
genre_stats_sorted = genre_stats.sort_values('avg_revenue', ascending = False)

# genre analysis visualization
fig = plt.figure(figsize = (22, 17))
gs = fig.add_gridspec(2, 2, hspace = 0.5, wspace = 0.3, left = 0.10, right = 0.96, top = 0.93, bottom = 0.08)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# PLOT 1: Average revenue by genre (top 10)
top_revenue_genres = genre_stats_sorted.head(10)
print(top_revenue_genres)
bars1 = ax1.barh(range(len(top_revenue_genres)), top_revenue_genres['avg_revenue'],
                 color = 'steelblue', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# now, we highlight the top genre
bars1[0].set_color('red')
bars1[0].set_edgecolor('darkred')
bars1[0].set_linewidth(2.5)

ax1.set_yticks(range(len(top_revenue_genres)))
ax1.set_yticklabels(top_revenue_genres['genre'], fontsize = 11)

for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
             f"${width:.1f}M", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')
    
ax1.set_xlabel('Average Revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Genre', fontsize = 12, fontweight = 'bold')
ax1.set_title('Top 10 genres by average revenue', fontsize = 13, fontweight = 'bold')
ax1.invert_yaxis()
ax1.grid(True, alpha = 0.3, axis = 'x')
ax1.set_xlim(0, max(top_revenue_genres['avg_revenue']) * 1.15)

# PLOT 2: Blockbusters rate by genre (top 10)
top_blockbuster_genres = genre_stats.nlargest(10, 'blockbuster_rate')
bars2 = ax2.barh(range(len(top_blockbuster_genres)), top_blockbuster_genres['blockbuster_rate'],
                 color = 'orange', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlighting top genre
bars2[0].set_color('red')
bars2[0].set_edgecolor('darkred')
bars2[0].set_linewidth(2.5)

ax2.set_yticks(range(len(top_blockbuster_genres)))
ax2.set_yticklabels(top_blockbuster_genres['genre'], fontsize = 11)

for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
             f"{width:.1f}%", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')
    
ax2.set_xlabel('Blockbuster rate (%)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Genre', fontsize = 12, fontweight = 'bold')
ax2.set_title('Top 10 genres by blockbuster rate', fontsize = 13, fontweight = 'bold')
ax2.invert_yaxis()
ax2.grid(True, alpha = 0.8, axis = 'x')
ax2.set_xlim(0, max(top_blockbuster_genres['blockbuster_rate']) * 1.12)

# PLOT 3: Average popularity by genre (top 10)
top_popularity_genres = genre_stats.nlargest(10, 'avg_popularity')
bars3 = ax3.barh(range(len(top_popularity_genres)), top_popularity_genres['avg_popularity'],
                 color = 'purple', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlighting top genre
bars3[0].set_color('darkviolet')
bars3[0].set_edgecolor('black')
bars3[0].set_linewidth(2.5)

ax3.set_yticks(range(len(top_popularity_genres)))
ax3.set_yticklabels(top_popularity_genres['genre'], fontsize = 11)

for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f"{width:.1f}", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')
    
ax3.set_xlabel('Average popularity score', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Genre', fontsize = 12, fontweight = 'bold')
ax3.set_title('Top 10 genres by average popularity', fontsize = 13, fontweight = 'bold')
ax3.invert_yaxis()
ax3.grid(True, alpha = 0.3, axis = 'x')
ax3.set_xlim(0, max(top_popularity_genres['avg_popularity']) * 1.12)

# PLOT 4: Genre popularity - revenue vs count (bubble chart)
# we first select the top 12 genres by count for clarity
top_count_genres = genre_stats.nlargest(12, 'count')

scatter = ax4.scatter(top_count_genres['avg_revenue'],
                      top_count_genres['blockbuster_rate'],
                      s = top_count_genres['count'] * 2,
                      c = top_count_genres['avg_popularity'],
                      cmap = 'viridis',
                      alpha = 0.6,
                      edgecolors = 'black',
                      linewidth = 1.5)

# genre labels to be added
for _, row in top_count_genres.iterrows():
    ax4.annotate(row['genre'],
                 (row['avg_revenue'], row['blockbuster_rate']),
                 fontsize = 9,
                 ha = 'center',
                 va = 'center',
                 fontweight = 'bold')
    
ax4.set_xlabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Blockbuster rate (%)', fontsize = 12, fontweight = 'bold')
ax4.set_title('Genre success matrix (Size = film count, Color = popularity)', fontsize = 13, fontweight = 'bold')
ax4.grid(True, alpha = 0.3)

# colorbar
color_bar = plt.colorbar(scatter, ax = ax4)
color_bar.set_label('Avg popularity', fontsize = 10, fontweight = 'bold')

fig.suptitle('Genre dominance: which types of films rule the box office?',
             fontsize = 16, fontweight = 'bold', y = 0.995)

plt.tight_layout(rect = [0, 0.01, 1, 0.99], h_pad = 3, w_pad = 2.5)
plt.show()

# INSIGHTS:
# As we can see from the resulting plots, the genre of a film largely contributes to the commercial success of a film. Family films lead in revenue ($257.3 Million on average), followed
# by Animation ($241.9 Million avg) and adventure ($223.1 Million avg), demonstrating that family friendly, spectacle driven content dominates box offices. Animations shows the highest 
# blockbuster rate (32.9 %), with adventure (29.4%) and family (28.8 %) close behind, confirming these genres are the safest bets for blockbuster success. Family films also top popularity
# scores (27.3), validating their broad audience appeal. The bubble chart (plot 4) reveals that while drama is the most produced genre, it generates low revenues and blockbuster rates, highlighting Hollywood's 
# strategic shift toward high - budget franchise genres (Animation, Adventure, Sci-fi) over dramatic storytelling. Genre choice is then a critical success factor, studios prioritize spectacle and family appeal
# over artistic merit to maximize their commercial returns.

# 9) DOES THE TYPES OF PRODUCTION COMPANIES AND PRODUCTION COUNTRIES INFLUENCE THE BLOCKBUSTER SUCCESS OF FILMS? ---------------------------------
plt.close('all')

# data for plotting
country_data = data[['countries_str', 'revenue', 'is_blockbuster', 'popularity']].dropna()
company_data = data[['companies_str', 'revenue', 'is_blockbuster', 'popularity']].dropna()

# extract primary company and company (the first one listed)
country_data['primary_country'] = country_data['countries_str'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
)

company_data['primary_company'] = company_data['companies_str'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) and x != '' else 'Unknown'
)

# Country analysis
country_stats = country_data.groupby('primary_country').agg({
    'revenue': ['mean', 'count'],
    'popularity': 'mean',
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()

country_stats.columns = ['country', 'avg_revenue', 'count', 'avg_popularity', 'blockbuster_rate']
country_stats['avg_revenue'] = country_stats['avg_revenue'] / 1e6

# filtering countries with at least 30 films for statistical significance
country_stats = country_stats[country_stats['count'] >= 100].copy()
country_stats_sorted = country_stats.sort_values('avg_revenue', ascending = False)

# Company analysis
company_stats = company_data.groupby('primary_company').agg({
    'revenue': ['mean', 'count'],
    'popularity': 'mean',
    'is_blockbuster': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()

company_stats.columns = ['company', 'avg_revenue', 'count', 'avg_popularity', 'blockbuster_rate']
company_stats['avg_revenue'] = company_stats['avg_revenue'] / 1e6

# filtering companies with at least 20 films for statistical significance
company_stats = company_stats[company_stats['count'] >= 20].copy()
company_stats_sorted = company_stats.sort_values('avg_revenue', ascending = False)

# Visualization
fig = plt.figure(figsize = (24, 18))
gs = fig.add_gridspec(2, 2, left = 0.22, right = 0.98, bottom = 0.07, top = 0.90, hspace = 0.46, wspace = 0.90)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# PLOT 1: top 12 countries by average revenue
top_countries_revenue = country_stats_sorted.head(12)
bars1 = ax1.barh(range(len(top_countries_revenue)), top_countries_revenue['avg_revenue'],
                 color = 'green', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlighting top country
bars1[0].set_color('yellow')
bars1[0].set_edgecolor('orange')
bars1[0].set_linewidth(2.5)

ax1.set_yticks(range(len(top_countries_revenue)))
ax1.set_yticklabels(top_countries_revenue['country'], fontsize = 11)

for i, (idx, row) in enumerate(top_countries_revenue.iterrows()):
    ax1.text(row['avg_revenue'] + 3, i,
             f"${row['avg_revenue']:.1f}M", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')
    
ax1.set_xlabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Country', fontsize = 12, fontweight = 'bold')
ax1.set_title('Top 12 Countries by average revenue', fontsize = 14, fontweight = 'bold', pad = 20)
ax1.invert_yaxis()
ax1.grid(True, alpha = 0.3, axis = 'x')
ax1.set_xlim(0, max(top_countries_revenue['avg_revenue']) * 1.25)

# PLOT 2: Top 12 Countries by blockbuster rate
top_countries_blockbuster = country_stats.nlargest(12, 'blockbuster_rate')
bars2 = ax2.barh(range(len(top_countries_blockbuster)), top_countries_blockbuster['blockbuster_rate'],
                 color = 'orange', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlight top countries 
bars2[0].set_color('red')
bars2[0].set_edgecolor('darkred')
bars2[0].set_linewidth(2.5)

ax2.set_yticks(range(len(top_countries_blockbuster)))
ax2.set_yticklabels(top_countries_blockbuster['country'], fontsize = 11)

for i, (idx, row) in enumerate(top_countries_blockbuster.iterrows()):
    ax2.text(row['blockbuster_rate'] + 0.5, i,
    f"{row['blockbuster_rate']:.1f}%", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')

ax2.set_xlabel('Blockbuster rate (%)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Country', fontsize = 12, fontweight = 'bold')
ax2.set_title('Top 12 Countries by blockbuster rate', fontsize = 14, fontweight = 'bold', pad = 20)
ax2.invert_yaxis()
ax2.grid(True, alpha = 0.3, axis = 'x')
ax2.set_xlim(0, max(top_countries_blockbuster['blockbuster_rate']) * 1.20)

# PLOT 3: Top 12 Companies by average revenue
top_companies_revenue = company_stats_sorted.head(12)
bars3 = ax3.barh(range(len(top_companies_revenue)), top_companies_revenue['avg_revenue'],
                 color = 'purple', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlighting top company
bars3[0].set_color('darkviolet')
bars3[0].set_edgecolor('black')
bars3[0].set_linewidth(2.5)

ax3.set_yticks(range(len(top_companies_revenue)))
ax3.set_yticklabels(top_companies_revenue['company'], fontsize = 11)

for i, (idx, row) in enumerate (top_companies_revenue.iterrows()):
    ax3.text(row['avg_revenue'] + 3, i,
             f"${row['avg_revenue']:.1f}M", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')

ax3.set_xlabel('Average revenue ($ Millions)', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Production company', fontsize = 12, fontweight = 'bold')
ax3.set_title('Top 12 Companies by average revenue', fontsize = 14, fontweight = 'bold', pad = 20)
ax3.invert_yaxis()
ax3.grid(True, alpha = 0.3, axis = 'x')
ax3.set_xlim(0, max(top_companies_revenue['avg_revenue']) * 1.25)

# PLOT 4: Top 12 Companies by blockbuster rate
top_companies_blockbuster = company_stats.nlargest(12, 'blockbuster_rate')
bars4 = ax4.barh(range(len(top_companies_blockbuster)), top_companies_blockbuster['blockbuster_rate'],
                 color = 'teal', alpha = 0.8, edgecolor = 'black', linewidth = 1.5)

# highlighting top company
bars4[0].set_color('darkgreen')
bars4[0].set_edgecolor('black')
bars4[0].set_linewidth(2.5)

ax4.set_yticks(range(len(top_companies_blockbuster)))
ax4.set_yticklabels(top_companies_blockbuster['company'], fontsize = 11)

for i, (idx, row) in enumerate(top_companies_blockbuster.iterrows()):
    ax4.text(row['blockbuster_rate'] + 0.5, i,
    f"{row['blockbuster_rate']:.1f}%", ha = 'left', va = 'center', fontsize = 10, fontweight = 'bold')

ax4.set_xlabel('Blockbuster rate (%)', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Production Company', fontsize = 12, fontweight = 'bold')
ax4.set_title('Top 12 Companies by blockbuster rate', fontsize = 14, fontweight = 'bold', pad = 20)
ax4.invert_yaxis()
ax4.grid(True, alpha = 0.3, axis = 'x')
ax4.set_xlim(0, max(top_companies_blockbuster['blockbuster_rate']) * 1.20)

# we will give now each axes a little room so the value labels don't clip
for ax in (ax1, ax2, ax3, ax4):
    ax.margins(x = 0.08)
    ax.tick_params(axis = 'y', labelsize = 11)
    ax.set_title(ax.get_title (), pad = 14)

    if ax in (ax1, ax3):
        ax.tick_params(axis = 'y', pad = 6)
    else:
        ax.tick_params(axis = 'y', pad = 0)

fig.suptitle('Production origins: Do Country & Company determine success?', 
             fontsize = 18, fontweight = 'bold', y =  0.985)

plt.show()

# key statistics of this analysis:
# - Top country
top_country = country_stats_sorted.iloc[0]
print(f"\nTop Country by revenue: {top_country['country']}")
print(f"Average revenue: ${top_country['avg_revenue'].round()}M")
print(f"Blockbuster rate: {top_country['blockbuster_rate'].round()}%")
print(f"Films produced: {int(top_country['count'])}")

# - Top country by blockbuster rate
top_blockbuster_country = country_stats.loc[country_stats['blockbuster_rate'].idxmax()]
print(f"\nMost blockbuster prone Country: {top_blockbuster_country['country']}")
print(f"Blockbuster Rate: {top_blockbuster_country['blockbuster_rate'].round()}%")
print(f"Average revenue: ${top_blockbuster_country['avg_revenue'].round()}M")

# it results to be that the best country for blockbuster rate is United kingdom followed by the United states, probably United kingdom
# comes before the USA because it counts some films that are co - produced by both US and UK (like Harry Potter, James Bond and so on), 
# so the top blockbuster countries are in general english speaking countries.

# Top company
top_company = company_stats_sorted.iloc[0]
print(f"\nTop Company by revenue: {top_company['company']}")
print(f"Blockbuster Rate: {top_company['blockbuster_rate'].round()}%")
print(f"Average revenue: ${top_company['avg_revenue'].round()}M")
print(f"Films produced: {int(top_company['count'])}")

# the leading company in blockbuster film production is Lucas film, with a blockbuster rate of 60% and an average revenue of 494 Millions
# of dollars even though it didn't produced so many films in comparison to other companies (20 films were produced by Lucasfilms).

# CONCLUSIONS: IS THERE A BLOCKBUSTER FORMULA FOR SUCCESS?
# After the analysis conducted, the data reveals that there is NO guaranteed formula for blockbuster success, but there are some patterns 
# within the data that can increase the likelihood of a film to become a blockbuster success. Successful blockbusters typically combine:

# - high budgets ($ 50M +) that enable spectacle and star power
# - family friendly genres (Animation, adventure, family)
# - strategic summer or holiday releases
# - longer runtimes (120-150 minutes)
# - high pre-release marketing buzz (popularity)
# - major studio backing

# However, high variance at all budget levels proves that while these factors increase the probability of blockbuster success, exceptional 
# creative execution, cultural timing and luck remain essential. The modern film industry operates on a "tentpole" strategy-fewer, bigger bets
# on franchise ready spectacles during peak seasos with 78.3% of films profitable but only 10% achieving true blockbuster status. Ultimately, data 
# can improve the opportunity for success, but cannot guarantee it, the magic and risk of cinema lie in that
# irreducible uncertainty










