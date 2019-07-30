# -*- coding: utf-8 -*-
"""
Python 3.7

Make sure that these libraries are in your Anaconda Virtual Environment
- pandas        (I ran with 0.24.2)

Make sure that the following data sets are in your working directory in Spyder
- winemag-data-130k-v2.csv (download from https://www.kaggle.com/zynicide/wine-reviews)
- gaming.csv
- movies.csv
- meets.csv
- openpowerlifting.csv

@author: Greg Loughnane
"""

import pandas as pd

#%%
#------------------------Creating, reading, and writing------------------------
print('------------------------Creating, reading, and writing------------------------')

# Create fruit and fruit sale data
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])
print(fruits)

fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],
                index=['2017 Sales', '2018 Sales'])
print(fruit_sales)
print('--------------------------------------------------')

# Create recipe data
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
recipe = pd.Series(quantities, index=items, name='Dinner')
print(recipe)
print('--------------------------------------------------')

# Create animal data and print to .csv
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(animals)
animals.to_csv("cows_and_goats.csv")

#%%
#---------------------Indexing, selecting, assigning reference-----------------
print('---------------------Indexing, selecting, assigning reference-----------------')

# Read in data
pd.set_option('max_rows', 5)
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

# Naive accessors
print()
print('---------------------Naive accessors---------------------')
print()
print(reviews)  # Print everything
print('--------------------------------------------------')
print(reviews.country)  # Print single column: country
print('--------------------------------------------------')
print(reviews['country'])  # Print single column: country

# Index-based selection (iloc)
print()
print('---------------------Index-based selection (iloc)--------------------')
print()
print(reviews['country'][0])
print('--------------------------------------------------')
print(reviews.iloc[0])              # First row
print('--------------------------------------------------')
print(reviews.iloc[:, 0])           # First Column
print('--------------------------------------------------')
print(reviews.iloc[:3, 0])          # Rows 1-3, Column 1
print('--------------------------------------------------')
print(reviews.iloc[1:3, 0])         # Rows 2-3, Column 1
print('--------------------------------------------------')
print(reviews.iloc[[0, 1, 2], 0])   # Rows 1-3, Column 1
print('--------------------------------------------------')
print(reviews.iloc[-5:])            # Rows 5-end, All columns

# Label-based selection (loc)
print()
print('---------------------Label-based selection (iloc)--------------------')
print()
print(reviews.loc[0, 'country'])
print('--------------------------------------------------')
print(reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']])
print('--------------------------------------------------')

# Manipulating the index
print()
print('---------------------Manipulating the index--------------------')
print()
print(reviews.set_index("title"))

# Conditional selection
print()
print('---------------------Conditional selection--------------------')
print()
print(reviews.country == 'Italy')
print('--------------------------------------------------')
print(reviews.loc[reviews.country == 'Italy'])
print('--------------------------------------------------')
print(reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)])
print('--------------------------------------------------')
print(reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)])
print('--------------------------------------------------')
print(reviews.loc[reviews.country.isin(['Italy', 'France'])])
print('--------------------------------------------------')
print(reviews.loc[reviews.price.notnull()])

# Assigning Data
print()
print('---------------------Assigning Data--------------------')
print()
reviews['critic'] = 'everyone'
print(reviews['critic'])
print('--------------------------------------------------')
reviews['index_backwards'] = range(len(reviews), 0, -1)
print(reviews['index_backwards'])

#%%
#--------------------------Summary functions and maps--------------------------
print('--------------------------Summary functions and maps--------------------------')

# Read in data (again, to overwrite previous manipulations)
pd.set_option('max_rows', 5)
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
print(reviews.head())
print('--------------------------------------------------')

# Calculate median of the "points" column
median_points = reviews.points.median()
print(median_points)
print('--------------------------------------------------')

# Find which countries are in the data set
countries = reviews.country.unique()
print(countries)
print('--------------------------------------------------')

# Count the number of times each country appears
reviews_per_country = reviews.country.value_counts()
print(reviews_per_country)
print('--------------------------------------------------')

# Calculate mean and subtract it from the price of each wine reviewed
centered_price = reviews.price - reviews.price.mean()
print(centered_price)
print('--------------------------------------------------')

# Find the bargain wine
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
print(bargain_wine)
print('--------------------------------------------------')

# Find "tropical" and "fruity" wines and count how many of them there are
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
print(descriptor_counts)
print('--------------------------------------------------')

# Define a star rating
def stars(row):
    if row.country == 'Canada': # all Canadian Vinteners Association wines are a 3
        return 3
    elif row.points >= 95:  # Greater than 95 rating is 3 star
        return 3
    elif row.points >= 85:  # Greater than 85 rating is 2 star
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')
print(star_ratings)

#%%
#-----------------------------Grouping and sorting-----------------------------
print('-----------------------------Grouping and sorting-----------------------------')

# Find the most common wines in the set
reviews_written = reviews.groupby('taster_twitter_handle').size()
print(reviews_written)
# or reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
print('--------------------------------------------------')

# Best wine for the money?
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
print(best_rating_per_price)
print('--------------------------------------------------')

# Find min and max prices for each "variety" of wine
price_extremes = reviews.groupby('variety').price.agg([min, max])
print(price_extremes)
print('--------------------------------------------------')

# Most expensive wine varieties
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
print(sorted_varieties)
print('--------------------------------------------------')

# Find the average rating from each reviewer
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
print(reviewer_mean_ratings)
# Check if there are significant differences
print(reviewer_mean_ratings.describe())
print('--------------------------------------------------')

# What combination of countries and varieties are the most common?
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
print(country_variety_counts)

#%%
#--------------------Data types and missing data reference---------------------
print('--------------------Data types and missing data reference---------------------')

# Check single column data type
print(reviews.price.dtype)
print('--------------------------------------------------')

# Check all column data types
print(reviews.dtypes)
print('--------------------------------------------------')

# Transform one column to float64
print(reviews.points.astype('float64'))
print('--------------------------------------------------')

# Can also check dtype of a DataFrame or Series' index
print(reviews.index.dtype)
print('--------------------------------------------------')

# Find missing data
print(reviews[reviews.country.isnull()])
print('--------------------------------------------------')

# Replace each NaN with "Unknown"
print(reviews.region_2.fillna("Unknown"))
print('--------------------------------------------------')

# Kerin O'Keefe changed her Twitter Handle - update it
print(reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino"))

#%%
#---------------------------Renaming and Combining-----------------------------
print('---------------------------Renaming and Combining-----------------------------')

# Read in data (again, to overwrite previous manipulations)
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
print(reviews.head())
print('--------------------------------------------------')

# Change region_1 to 'region' and region_2 to 'locale'
renamed = reviews.rename(columns=dict(region_1='region', region_2='locale'))
print(renamed)
print('--------------------------------------------------')

# Set the index of the dataset to wines
reindexed = reviews.rename_axis('wines', axis='rows')
print(reindexed)
print('--------------------------------------------------')

# Things on Reddit - read in and create DataFrames for gaming and movie products
gaming_products = pd.read_csv("gaming.csv")
print('--------------------------------------------------')
gaming_products['subreddit'] = "gaming"
print(gaming_products.head())
print('--------------------------------------------------')
movie_products = pd.read_csv("movies.csv")
print('--------------------------------------------------')
movie_products['subreddit'] = "movies"
print(movie_products.head())

combined_products = pd.concat([gaming_products, movie_products])
print(combined_products.head())

# Powerlifting Database - read in two DataFrames and combine them
powerlifting_meets = pd.read_csv("meets.csv")
print(powerlifting_meets.head())
print('--------------------------------------------------')
powerlifting_competitors = pd.read_csv("openpowerlifting.csv")
print(powerlifting_competitors.head())
print('--------------------------------------------------')

powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
print(powerlifting_combined.head())
