# first, libraries needed for the project are loaded into the environment
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------- Data cleaning and Preparation ---------------------------------------------------
# load the dataset movie_metadata.csv into the environment
data = pd.read_csv("movies_metadata.csv", low_memory = False, on_bad_lines = "skip")

# to visualize the composition of the dataset, the head of it has been printed (specifically the first 8 rows)
data.head(8)

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
columns_to_drop = ['adult','belongs_to_collection', 'homepage','poster_path', 'tagline', 'video', 'spoken_languages']
data = data.drop(columns = columns_to_drop, errors = 'ignore')

# then, the columns name are printed again to ensure that everything worked out correctly
data.columns.to_list()

# drop the null rows for the 'title' and 'release_date' because they contain really few null values
data = data.dropna(subset = ['title', 'release_date'])

# Now, in the remaining data, missing values will be handled
# numeric types data
numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors = 'coerce')

# separate the numerical and text columns 
numeric_columns = data.select_dtypes(include = [np.number]).columns
categorical_columns = data.select_dtypes(exclude = [np.number]).columns

for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_columns:
    data[col] = data[col].fillna('Unknown')

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

# check the 0 values in the columns of the dataset, to see if they need some fixing
for col in numeric_columns:
    zero_count = (data[col] == 0).sum()
    zero_percent = 100 * zero_count / len (data)
    print(f"{col}: {zero_count} zeros ({zero_percent:.2f}%)")

# the columns 'budget', 'revenue', contain a lot of zeros. It is better to substitute this values with NaN. Instead, since 'runtime'
# contains only 0.26% of zero values, we can substitute them with a median
columns_to_fix = ['budget', 'revenue', 'runtime']
data[columns_to_fix] = data [columns_to_fix].replace(0, np.nan)
data['runtime'] = data['runtime'].fillna(data['runtime'].median())

# in the dataset the number of rows (observations) and columns remaining after the cleaning are (45434, 17)
data.shape

# --------------------------------------------- EDA (Exploratory data analysis) ------------------------------------------------------

