# -*- coding: utf-8 -*-
"""
Python 3.6

Make sure that these libraries are in your Anaconda Virtual Environment
- pandas        (I ran with 0.24.2)
- scikit-learn  (I ran with 0.20.3)

and...

Make sure that the melb_data.csv is in your working directory in Spyder

@author: Greg Loughnane
"""


# read the data and store data in DataFrame titled melbourne_data
import pandas as pd
melbourne_data = pd.read_csv('melb_data.csv') 
# print a summary of the data in Melbourne data
print(melbourne_data.describe())
print()
print(melbourne_data.columns)
print()

# The Melbourne data has some missing values 
# (some houses for which some variables weren't recorded.)
# dropna drops missing home values from the data (na = "not available")
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
melbourne_features = ['Rooms', 
                      'Bathroom', 
                      'Landsize', 
                      'Lattitude', 
                      'Longtitude']

X = melbourne_data[melbourne_features]
print(X.describe())
print()
print(X.head())
print()

#-------------------------------Simple ML Model-------------------------------    
from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)
print(melbourne_model)
print()

# Fit model
melbourne_model.fit(X, y)

# Practice predicting house prices
print("Making predictions for the following 5 houses:")
print()
print(X.head())
print()
print("The predictions are")
print(melbourne_model.predict(X.head()))
print()

# Predict house prices
predicted_home_prices = melbourne_model.predict(X)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, predicted_home_prices)
print("The predicted home prices are:", predicted_home_prices )
print("The mean absolute errors in each prediction are are:", mae)
print()

#----------------------------Hyperparameter Tuning-----------------------------  
# split data into training and validation data, for both features and target
# The split is based on a random number generator.
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
print()

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, 
                                  random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" \
          %(max_leaf_nodes, my_mae))
    
#-------------------------------Better ML Model--------------------------------    
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
