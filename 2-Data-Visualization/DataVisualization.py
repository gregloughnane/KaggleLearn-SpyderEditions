# -*- coding: utf-8 -*-
"""
Python 3.6

Make sure that these libraries are in your Anaconda Virtual Environment
- pandas        (I ran with 0.24.2)
- matplotlib    (I ran with 3.0.3)
- seaborn       (I ran with 0.9.0)

and...

Make sure that following data sets are in your working directory in Spyder
- fifa.csv
- spotify.csv
- flight_delays.csv
- insurance.csv
- iris.csv
- iris_setosa.csv
- iris_versicolor
- iris_virginica.csv

@author: Greg Loughnane
"""

#--------------------------------Hello Seaborn---------------------------------
print('--------------------------------Hello Seaborn---------------------------------')

# Get libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Choose your theme!
#sns.set_style("darkgrid")
#sns.set_style("whitegrid")
sns.set_style("white")
#sns.set_style("dark")
#sns.set_style("ticks")

# Read the file into a variable spotify_data
fifa_data = pd.read_csv('fifa.csv', index_col="Date", parse_dates=True)

# Print the first 5 rows of the data
print(fifa_data.head())

# Set the width and height of the figure
print('--------------------------------------------------')
plt.figure(figsize=(9,6))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)
plt.show() # Use this to make sure that plots are shown inline

#---------------------------------Line Charts----------------------------------
print('---------------------------------Line Charts----------------------------------')

# Read the file into a variable spotify_data
spotify_data = pd.read_csv('spotify.csv', index_col="Date", parse_dates=True)

# Print the first and last 5 rows of the data
print(spotify_data.head())
print('--------------------------------------------------')
print(spotify_data.tail())

# Set the width and height of the figure
plt.figure(figsize=(9,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)

# Plot subset of the data
list(spotify_data.columns)

# Set the width and height of the figure
plt.figure(figsize=(9,6))

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
plt.xlabel("Date")
plt.show()

#----------------------------------Bar Charts----------------------------------
print('----------------------------------Bar Charts----------------------------------')

# Read the file into a variable flight_data
flight_data = pd.read_csv('flight_delays.csv', index_col="Month")
print(flight_data)

# Set the width and height of the figure
plt.figure(figsize=(9,6))

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
plt.ylabel("Arrival delay (in minutes)")
plt.show()

#----------------------------------Heatmaps------------------------------------
print('----------------------------------Heatmaps------------------------------------')

# Set the width and height of the figure
plt.figure(figsize=(9,6))

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.title("Average Arrival Delay for Each Airline, by Month")
plt.xlabel("Airline")
plt.show()

#--------------------------------Scatter Plots---------------------------------
print('--------------------------------Scatter Plots---------------------------------')

# Read the file into a variable insurance_data
insurance_data = pd.read_csv('insurance.csv')
print(insurance_data.head())

# Simple scatterplot
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# Add regression line
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

# How about a scatterplot with regression lines, color-coded with information?
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.show()

# Swarm plots are pretty cool too (a.k.a. categorical scatter plots)
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
plt.show()

#---------------------------------Distributions--------------------------------
print('---------------------------------Distributions--------------------------------')

# Read the file into a variable insurance_data
iris_data = pd.read_csv('iris.csv', index_col="Id")
print(iris_data.head())

# Histogram 
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
plt.show()

# KDE plot 
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
plt.show()

# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
plt.show()

# Read the files into variables 
iris_set_data = pd.read_csv('iris_setosa.csv', index_col="Id")
iris_ver_data = pd.read_csv('iris_versicolor.csv', index_col="Id")
iris_vir_data = pd.read_csv('iris_virginica.csv', index_col="Id")

# Print the first 5 rows of the Iris versicolor data
print(iris_ver_data.head())

# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()
plt.show()

# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")
plt.show()
