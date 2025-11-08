# first, libraries needed for the project are loaded into the environment
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from ast import literal_eval
import scipy
import json
import math

# -------------------------------------------------- Data cleaning and Preparation ---------------------------------------------------
# load the dataset movie_metadata.csv into the environment
data = pd.read_csv("movies_metadata.csv", low_memory = False, on_bad_lines = "skip")

# to visualize the composition of the dataset, the head of it has been printed (specifically the first 8 rows)
data.head(8)
pd.set_option('display.max_columns', 100)

# the name of the columns that we have in the dataset are the following
data.columns.to_list()
data.shape
data.info()
data.describe()

# it is better to clear the column names, to avoid errors when dropping the columns that are not needed, using the command strip
print(data.columns.tolist())
data.columns = data.columns.str.strip()

# Null values must be checked to handle them by substituting them or drop them. As it can be seen, there are several columns 
# that contains missing values.
data.isna().sum().sort_values(ascending = False).head(20)

# columns that are irrelevant for the analysis must be dropped
columns_to_drop = ['adult','belongs_to_collection', 'homepage','poster_path', 'tagline', 'video', 'spoken_languages', 'backdrop_path',
                   'imdb_id', 'original_title', 'overview', 'status']
data = data.drop(columns = columns_to_drop, errors = 'ignore')

# then, the columns name are printed again to ensure that everything worked out correctly
data.columns.to_list()

# parse the dates before drop
data ['release_date'] = pd.to_datetime(data['release_date'], errors = 'coerce')

# drop the null rows for the 'title' and 'release_date' because they contain really few null values
data = data.dropna(subset = ['title', 'release_date'])

# we extract the year, in case it is needed in the next analysis
data['release_year'] = data['release_date'].dt.year

# it is better to filter unrealistic release years
data = data[(data['release_year'] >= 1900) & (data['release_year'] <= 2025)]

# Now we show the changes
data.columns.to_list()

# Now, in the remaining data, missing values will be handled
# numeric types data
numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')

# separate the numerical and text columns 
numeric_columns = data.select_dtypes(include = [np.number]).columns
categorical_columns = data.select_dtypes(exclude = [np.number]).columns

# first, we replace only runtime zeros with NaN values
data['runtime'] = data['runtime'].replace(0, np.nan)

safe_impute = ['runtime', 'popularity', 'vote_average', 'vote_count']
for col in safe_impute:
    data[col] = data[col].fillna(data[col].median())

# Filter extreme runtime values (so only reasonable movie lengths are retained)
data = data[(data['runtime'] >= 40) & (data['runtime'] <= 300)]

# budget and revenues zeros has also to be fixed
data[['budget', 'revenue']] = data[['budget', 'revenue']].replace(0, np.nan)

# now we remove unrealistic values
data = data[(data['budget'].isna()) | (data['budget'] >= 1000)]
data = data[(data['revenue'].isna()) | (data['revenue'] >= 1000)]

# now, we verify that there are no more missing values in the dataset. \\
# Budget and revenues Nan are intentionally kept for flexible analysis
missing_check = data.isna().sum()
print("Missing values per column:")
print(missing_check[missing_check > 0])

# it is better to check if there are also duplicate data before going on with the analysis
# first we check generally the duplicates in the dataset
data.duplicated().sum()

# The duplicated values are 17, so it is better to drop them
data = data.drop_duplicates(keep = 'first')

# then we check for duplicates also by title and release_date since some movies can have the same title but be different in release 
# date, which is a common thing that can happen to remakes of old movies and so on.
data.duplicated(subset = ['title', 'release_date']).sum()

# In this case, the duplicate movies with the same name and exact date are 15, so also this ones shall be removed from the dataset
data = data.drop_duplicates(subset = ['title','release_date'], keep = 'first')

# now if we check again for duplicates there should be no one remaining
data.duplicated().sum()
data.duplicated(subset = ['title', 'release_date']).sum()

# since in the dataset the value_count some counts are really few is it better to drop the values that are under or equal to 100 to have 
# a more reliable mean of the opinions of the audiance about the movies
data = data [data['vote_count'] >= 100]

# JSON fields parse
json_columns = ['genres', 'production_companies', 'production_countries']

for col in json_columns:
    if col in data.columns:
        s = data[col].fillna('[]').astype(str)
        looks_like_list = s.str.strip().str.startswith('[') & s.str.strip().str.endswith(']')
        s = s.where(looks_like_list, '[]')

        parsed = s.apply(lambda x: literal_eval(x) if x.strip().startswith('[') else [])
        data[col] = parsed.apply(lambda v:v if isinstance(v, list) else [])

# now we verify at each of this columns now contains a list
data[json_columns].apply(lambda x: x.apply(type)).head()

# and now the content of the parsed json columns
data[['genres', 'production_companies', 'production_countries']].head()

# then, names from json columns will be extract as comma separated for better readability
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

# now, i drop
# check the 0 values in the columns of the dataset, to see if they need some fixing
for col in numeric_columns:
    zero_count = (data[col] == 0).sum()
    zero_percent = 100 * zero_count / len (data)
    print(f"{col}: {zero_count} zeros ({zero_percent:.2f}%)")


# --------------------------------------------- Final summary of the data cleaned --------------------------------------------------------------
print ("\n" + "="*60)
print("Final cleaned dataset summary")
print("="*60)

# dataset shape
print(f"\nFinal dataset shape: {data.shape}")

# missing values summary that were retained for further analysis
print(f"\nBudget missing: {data['budget'].isna().sum()} ({100*data['budget'].isna().sum()/len(data):.1f}%)")
print(f"Revenue missing: {data['revenue'].isna().sum()} ({100*data['revenue'].isna().sum()/len(data):.1f}%)")
print(f"Movies with BOTH budget and revenue: {data[['budget', 'revenue']].notna().all(axis=1).sum()}")

# check of the quality of the data
print (f"\nData Quality checks:")
print(f"  - Release year range: {data['release_year'].min():.0f} to {data['release_year'].max():.0f}")
print(f"  - Runtime range: {data['runtime'].min():.0f} to {data['runtime'].max():.0f} minutes")
print(f"  - Vote average range: {data['vote_average'].min():.1f} to {data['vote_average'].max():.1f}")
print(f"  - Unique genres: {data['primary_genre'].nunique()}")

# top genres found in the dataset
print (f"\nThe top 5 Primary Genres are:")
print (data['primary_genre'].value_counts().head())

# now we show the final columns list of the dataset
print(f"\nFinal columns retained: {data.columns.to_list()}")

# we are not eliminating outliers (blockbuster films) during the data cleaning process because they can represent legitimate
# box office phenomena (films like Avatar, Titanic ...) and are essential to understanding the film industy's 'blockbuster model
# Removing them would mispresent how the industry actually operates. However, we will analyze the data both with and without outliers
# to understand their impact on our findings


# --------------------------------------------------- feature engineering ----------------------------------------------------------------------

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

# core blockbuster analysis -------------------------------------------------------------------------------------------------------------------
# A first question that can be analyzed is the following and its the most important one
# in this project (the most critical relationship)

# 1) Does spending more on a movie guarantee higher revenue? (BUDGET VS REVENUE) --------------------------------------------------------------

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

# ROI (return of investment) analysis: the most remarkable finding in the dramatic divergence between the mean ROI (963.5%)
# and the median ROI (145.4%). This 818 percentage points gap is not a statistical anomaly, it's a defining characteristic of the film
# industry's economics. 
# - The average ROI of 963.5% suggests that movies generate approximately 10.6 times their production budget in revenue.
#   On the surface, this appears extraordinarily profitable, implying that for every dollar invested, the industry returns over 
#   ten dollars. However, this figure is misleading when examined in isolation. 
# - The median ROI of 145.4% paints a more realistic picture. This indicates that the typical film generates 2.5 times its budget,
#   still profitable, but far less spectacula than the mean suggests.

# This disparity between mean and median reveals a fundamental truth about the film industry: it operates on a highly skewed distribution. 
# A small number of extraordinary successes—think Avatar, Titanic, or Marvel blockbusters—generate returns so astronomical that they dramatically 
# inflate the average. These outliers can achieve ROIs of 1,000%, 2,000%, or even higher, particularly when low-budget films become unexpected cultural phenomena.
clean_data['roi'] = (clean_data['revenue'] - clean_data['budget'])/ clean_data['budget'] * 100
print(f"Average ROI: {clean_data['roi'].mean().round(2)}")
print(f"Median ROI: {clean_data['roi'].median().round(2)}")

# now we analyze the budget categories included in the plotù
low_budget = clean_data[clean_data['budget'] < 1e6]
mid_budget = clean_data[(clean_data['budget'] >= 1e6) & (clean_data['budget'] < 5e7)]
high_budget = clean_data[clean_data['budget'] >= 5e7]

print(f"\nLow budget films (<$1M): {len(low_budget)}, have an average revenue of: ${low_budget['revenue'].mean()/1e6} M")
print(f"Mid budget films ($1M - $50M): {len(mid_budget)}, average revenue: ${mid_budget['revenue'].mean()/1e6} M")
print(f"High budget films (> $50M): {len(high_budget)}, average revenue: ${high_budget['revenue'].mean()/1e6} M")

# 2) Which budget ranges are most profitable (Budget vs Profit - ROI perspective)? ----------------------------------------------------------------
plt.close('all')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# left plot: budget vs profit scatter plot
clean_profit_data = data[['budget', 'profit']].dropna()

# separate profitable and unprofitable movies for better visualization
profitable = clean_profit_data[clean_profit_data['profit'] > 0]
unprofitable = clean_profit_data[clean_profit_data['profit'] <= 0]

ax1.scatter(profitable['budget'], profitable['profit'],
            alpha = 0.5, color = 'seagreen', label = 'Profitable Movies', s = 20)

ax1.scatter(unprofitable['budget'], unprofitable['profit'],
            alpha = 0.5, color = 'red', label = 'Unprofitable Movies', s = 20)

# add trend line to the profit
log_budget_p = np.log10(profitable['budget'])
log_profit = np.log10(profitable['profit'])

z_profit = np.polyfit(log_budget_p, log_profit, 1)
p_profit = np.poly1d(z_profit)

budget_range_p = np.logspace(np.log10(clean_profit_data['budget'].min()),
                             np.log10(clean_profit_data['budget'].max()), 100)

trend_profit = 10 ** p_profit(np.log10(budget_range_p))

ax1.plot(budget_range_p, trend_profit, color = "red", linewidth = 2,
         linestyle = '--', label = f"Trend (slope = {z_profit[0]:.2f})")

# zero profit line
ax1.axhline (y= 0, color = 'black', linestyle = '-', linewidth = 1, alpha = 0.5, label = 'Break-even')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Budget ($)', fontsize = 12)
ax1.set_ylabel('Profit ($)', fontsize = 12)
ax1.set_title('Budget vs Profit: Which budgets are more profitable?',
              fontsize = 14, fontweight = 'bold')

ax1.legend(fontsize = 10, loc = 'upper left')
ax1.grid(True, alpha = 0.3)

# exlude extreme negatives for better visualization
ax1.set_ylim(1e3, 1e9)

# RIGHT PLOTS: ROI by budget category (by using a box plot)
budget_categories = []
roi_values = []

for _, row in clean_data.iterrows():
    if row['budget'] < 1e6:
        budget_categories.append('Low\n(<$1M)')
        roi_values.append(row['roi'])
    elif row['budget'] < 5e7:
        budget_categories.append('Mid\n($1M - $50M)')
        roi_values.append(row['roi'])
    else:
        budget_categories.append('High\n(>$50M)')
        roi_values.append(row['roi'])

roi_df = pd.DataFrame({'Budget Category': budget_categories, 'ROI (%)': roi_values})

# now we create the boxplot
sns.boxplot(x = 'Budget Category', y = 'ROI (%)', data = roi_df, ax = ax2,
            order = ['Low\n(<$1M)', 'Mid\n($1M - $50M)', 'High\n(>$50M)'],
            palette = ['lightcoral', 'lightblue', 'lightgreen'],
            showfliers = False)  # hide outliers for cleaner visualization

ax2.axhline (y = 0, color = 'red', linestyle = '--', linewidth = 1, alpha = 0.7, label = 'Break-even')
ax2.set_ylabel ('ROI (%)', fontsize = 12)
ax2.set_xlabel ('Budget Category', fontsize = 12)
ax2.set_title ('ROI distribution by Budget Category', fontsize = 14, fontweight = 'bold')
ax2.legend(fontsize = 10)
ax2.grid(True, alpha = 0.3, axis = 'y')

# the range of the y axis will be limited to clip extreme outliers just for visibility purposes
ax2.set_ylim (-100, 2000)

plt.tight_layout()
plt.show()

# There is no single "most profitable" budget range—each serves a different strategic purpose. The film industry thrives 
# on this diversity, balancing consistent mid-budget returns, the scale of blockbusters, and the occasional low-budget 
# phenomenon. The slope of 0.60 in the profit trend line captures this reality: spending more increases profit, but with 
# diminishing returns. Success depends not just on budget size, but on creative execution, marketing, timing, and often, luck.

# 3) Do better rated movies make more money? (VOTE AVERAGE (RATING) VS REVENUE) -----------------------------------------------------------------
plt.close(all)

 # so, first we prepare the data for the visualization
rating_revenue = data[['vote_average', 'revenue']].dropna()

# we first identify the data that would be considered outliers (for comparison purposes only)
Q1 = rating_revenue['revenue'].quantile(0.25)
Q3 = rating_revenue['revenue'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

# we then flag outliers for analysis purposes (but we will not remove them)
rating_revenue['is_outlier'] = rating_revenue['revenue'] > upper_bound
print (f"high - revenue outliers: {rating_revenue['is_outlier'].sum()} ({100 * rating_revenue['is_outlier'].sum()/len(rating_revenue):.1f}%)")

# now we create two datasets, one with all the data and one without the outliers
data_all = rating_revenue.copy()
data_filtered = rating_revenue[~rating_revenue['is_outlier']].copy()

# this is be the function that categorize the 
def categorize_ratings(rating):
    if rating < 5.0:
        return 'Poor\n(<5.0)'
    elif rating < 6.5:
        return 'Average\n(5.0 - 6.5)'
    elif rating < 7.5:
        return 'Good\n(6.5 - 7.5)'
    else:
        return 'Excellent\n(>= 7.5)'
    
# ----------- ANALYSIS WITH ALL THE DATA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# first we create the SCATTERPLOT --------- 
ax1.scatter(data_all['vote_average'],
            data_all['revenue'],
            alpha = 0.5,
            color = 'steelblue',
            s = 50,
            edgecolors = 'white',
            linewidth = 1,
            label = 'All movies')

# now we highlight outliers
outliers_only = data_all[data_all['is_outlier']]
ax1.scatter(outliers_only['vote_average'],
            outliers_only['revenue'],
            alpha = 0.7,
            color = 'red',
            s = 100,
            edgecolors = 'darkred',
            linewidth = 1,
            marker = '*',
            label = f"Blockbuster (n = {len(outliers_only)})")

# then we add a trend line inside the scatter plot
z_all = np.polyfit(data_all['vote_average'],
                   np.log10(data_all['revenue']), 1)
p_all = np.poly1d(z_all)

vote_range = np.linspace(0, 10, 100)
trend_all = 10 ** p_all(vote_range)

ax1.plot(vote_range, trend_all,
         color = 'darkgreen', linewidth = 2.5, linestyle = '--',
         label = f"Trend (slope = {z_all[0]:.2f})")

ax1.set_yscale('log')
ax1.set_xlabel('Vote Average (Rating)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Scatter plot: Rating vs revenue (all data)', fontsize = 13, fontweight = 'bold')
ax1.set_xlim(0, 10)
ax1.legend(fontsize = 10, loc = 'lower right')
ax1.grid(True, alpha = 0.3)

# calculate and display correlation
correlation_all = data_all['vote_average'].corr(data_all['revenue'])
ax1.text(0.05, 0.95, f"Correlation: {correlation_all:.3f}\n = {len(data_all)}",
         transform = ax1.transAxes, fontsize = 11,
         verticalalignment = 'top',
         bbox = dict(boxstyle = 'round', facecolor = 'lightblue', alpha = 0.8))

# then we plot on the same image also a BAR PLOT to show the relationship --------
data_all['category'] = data_all['vote_average'].apply(categorize_ratings)
category_order = ['Poor\n(<5.0)', 'Average\n(5.0 - 6.5)', 'Good\n(6.5 - 7.5)', 'Excellent\n(>= 7.5)']

avg_revenue_all = data_all.groupby('category')['revenue'].mean() / 1e6
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
    
fig.suptitle('Rating vs Revenue (data with outliers)',
             fontsize = 16, fontweight = 'bold', y = 0.98)

plt.tight_layout(rect = [0, 0, 1, 0.96])
plt.show()

# -------- ANALYSIS WITHOUT OUTLIERS  -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

# first we create the SCATTERPLOT --------- 
ax1.scatter(data_filtered['vote_average'],
            data_filtered['revenue'],
            alpha = 0.5,
            color = 'seagreen',
            s = 50,
            edgecolors = 'white',
            linewidth = 0.5,
            )

# then we add a trend line inside the scatter plot
z_filtered = np.polyfit(data_filtered['vote_average'],
                   np.log10(data_filtered['revenue']), 1)
p_filtered = np.poly1d(z_all)

trend_filtered = 10 ** p_filtered(vote_range)

ax1.plot(vote_range, trend_filtered,
         color = 'darkred', linewidth = 2.5, linestyle = '--',
         label = f"Trend (slope = {z_filtered[0]:.2f})")

ax1.set_yscale('log')
ax1.set_xlabel('Vote Average (Rating)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Revenue ($)', fontsize = 12, fontweight = 'bold')
ax1.set_title('Scatter plot: Rating vs revenue (filtered data)', fontsize = 13, fontweight = 'bold')
ax1.set_xlim(0, 10)
ax1.legend(fontsize = 10)
ax1.grid(True, alpha = 0.3)

# calculate and display correlation
correlation_filtered = data_filtered['vote_average'].corr(data_filtered['revenue'])
ax1.text(0.05, 0.95, f"Correlation: {correlation_filtered:.3f}\n = {len(data_filtered)}",
         transform = ax1.transAxes, fontsize = 11,
         verticalalignment = 'top',
         bbox = dict(boxstyle = 'round', facecolor = 'lightgreen', alpha = 0.8))

# then we plot on the same image also a BAR PLOT to show the relationship --------
data_filtered['category'] = data_filtered['vote_average'].apply(categorize_ratings)

avg_revenue_filtered = data_filtered.groupby('category')['revenue'].mean() / 1e6
avg_revenue_filtered = avg_revenue_filtered.reindex(category_order)

colors = ['red', 'orange', 'yellow', 'green']
bars = ax2.bar(category_order, avg_revenue_filtered, color = colors, alpha = 0.8,
               edgecolor = 'black', linewidth = 1.5)

# add the labels of the values on the bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.1f}M',
             ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    
fig.suptitle('Sensitivity analysis: Rating vs Revenue (without outliers)',
             fontsize = 16, fontweight = 'bold', y = 0.98)

plt.tight_layout(rect = [0, 0, 1, 0.96])
plt.show()

# RESULTS:
# this analysis examined the relationship between the vote average and the revenue. The question was: Does the audience vote translate
# into commercial success?

# Two main analysis were conducted:
# 1. with outliers: 
#    - SCATTERPLOT: in this the scatter plot revealed a weak positive correlation between rating and revenue with an almost flat line (slope = 0.02).
#    this indicates that while better rated films tend to earn more on average, the relationship is surprisingly weak.
#    - BOXPLOT: it provides a clearer aggregate view, which shows a progressive increase
# The weak correlation  is a significant finding, not a weakness in the analysis. It reveals that the film industry operates on factors beyond quality.

# 2. without outliers:
# When high-revenue outliers are removed using the IQR method (n=418 blockbusters excluded), the 
# results become COUNTER-INTUITIVE and reveal why outliers must be retained:

# WHY THIS HAPPENS - The Outlier Paradox:
# This counterintuitive result occurs because the "outliers" we removed ARE the successful 
# high-rated films (Avatar, Titanic, The Dark Knight, etc.). After removing them, what remains 
# in the "Excellent" category are predominantly:
#   - Art-house films with limited theatrical releases
#   - Independent films with strong critical praise but small audiences
#   - Foreign films with limited distribution
#   - Festival darlings that never achieved mainstream success

# so the outliers must be retained in our analysis

# So, in the end, quality has weak but positive effect on the revenue, meaning that it's not the only factor that contributes to the success of a film

# 4) Is there an optimal movie length for box office success? ----------------------------------------------------------------------------------------------------------------------------------------------
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

plt.tight_layout(rect = [0, 0, 1, 0.96])
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



