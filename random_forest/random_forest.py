# -*- coding: utf-8 -*-
"""
Created on Thu 26 15:37:56 2020

@author: edvonaldo
"""

import os
import csv
import math
import random
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

path = 'G:/Meu Drive/UFAL/2020.1/Aprendizado de Máquina/Databases/temps.csv'

features = pd.read_csv(path)

anomalies = features.describe()

years = features['year']
months = features['month']
days = features['day']

# List and then convert to datetime object
new_features = [str(int(year)) + '/' + str(int(months)) + '/' + str(int(days))
                for year, months, days in zip(years, months, days)]

new_features = [dt.datetime.strptime(new_feature, '%Y/%m/%d')
                for new_feature in new_features]

plt.style.use('fivethirtyeight')

#%%

# Set up the plotting layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (15,10))
fig.autofmt_xdate(rotation = 45)

# Actual max temperature measurement
ax1.plot(new_features, features['actual'])
ax1.set_xlabel(''); 
ax1.set_ylabel('Temperature (F)');
ax1.set_title('Max Temp')

# Temperature from 1 day ago
ax2.plot(new_features, features['temp_1'])
ax2.set_xlabel(''); 
ax2.set_ylabel('Temperature (F)'); 
ax2.set_title('Prior Max Temp')

# Temperature from 2 days ago
ax3.plot(new_features, features['temp_2'])
ax3.set_xlabel('Date'); 
ax3.set_ylabel('Temperature (F)'); 
ax3.set_title('Two Days Prior Max Temp')

# Friend Estimate
ax4.plot(new_features, features['friend'])
ax4.set_xlabel('Date'); 
ax4.set_ylabel('Temperature (F)'); 
ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)

#%% Prepare to data

features = pd.get_dummies(features)

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features axis 1 refers to the columns
features = features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

#%% Split data
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#%% Base line
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#%% Training data

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

#%% Making predictions
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#%% Performance metrics
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#%% Vizualizing One Tree Decision
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')