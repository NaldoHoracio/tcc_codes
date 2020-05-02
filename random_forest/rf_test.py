# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:56:32 2020

@author: edvonaldo
"""


# Pandas is used for data manipulation
import pandas as pd

path = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/random_forest/temps.csv'

# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv(path)

#%%
# One-hot encode categorical features
features = pd.get_dummies(features)

print('Shape of features after one-hot encoding:', features.shape)

#%%
# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)