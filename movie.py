# first, libraries needed for the project are loaded into the environment
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from ast import literal_eval

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
                   'id', 'imdb_id', 'original_title', 'original_language', 'overview', 'status']
data = data.drop(columns = columns_to_drop, errors = 'ignore')

# then, the columns name are printed again to ensure that everything worked out correctly
data.columns.to_list()

# parse the dates before drop
data ['release_date'] = pd.to_datetime(data['release_date'], errors = 'coerce')

# drop the null rows for the 'title' and 'release_date' because they contain really few null values
data = data.dropna(subset = ['title', 'release_date'])

# we extract the year, in case it is needed in the next analysis
data['release_year'] = data['release_date'].dt.year

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

# budget and revenues zeros has also to be fixed
data[['budget', 'revenue']] = data[['budget', 'revenue']].replace(0, np.nan)

# now we remove unrealistic values
data = data[(data['budget'].isna()) | (data['budget'] >= 1000)]
data = data[(data['revenue'].isna()) | (data['revenue'] >= 1000)]

# now, we verify that there are no more missing values in the dataset. If everything worked correctly it should return 0
data.isna().sum().sum()

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
data[json_columns].applymap(type).head()

# and now the content of the parsed json columns
data[['genres', 'production_companies', 'production_countries']].head()

# now, simple genre list will be extract
data['genres_list'] = data['genres'].apply(lambda x: [d.get('name') for d in x if 'name' in d] if isinstance(x, list) else [])
data['primary_genre'] = data['genres_list'].apply(lambda x: x[0] if x else 'Unknown')

# now, i drop
# check the 0 values in the columns of the dataset, to see if they need some fixing
for col in numeric_columns:
    zero_count = (data[col] == 0).sum()
    zero_percent = 100 * zero_count / len (data)
    print(f"{col}: {zero_count} zeros ({zero_percent:.2f}%)")

# in the dataset the number of rows (observations) and columns remaining after the cleaning are (45434, 17)
data.shape

# now we show the list of all the variables retained in the data before the analysis 
data.columns.to_list()

# then it is better to save the cleaned data
data.to_csv('movies_cleaned.csv', index = False)

# --------------------------------------------- EDA (Exploratory data analysis) ------------------------------------------------------

