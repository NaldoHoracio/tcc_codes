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

# Saving feature names for later use
feature_list = list(features.columns)
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
features = features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list_get_dummies = list(features.columns)

# Convert to numpy array
features = np.array(features)

#%% K-Fold CV
# Import the model we are using
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees

scores = []

importance_fields = 0.0
importance_fields_aux = []

# Instance RF
rf = RandomForestRegressor(n_estimators=500, random_state = 0)

kf_cv = KFold(n_splits=12, random_state=None, shuffle=False)

for train_index, test_index in kf_cv.split(features):
    #print("Train index: ", np.min(train_index), '- ', np.max(train_index))
    print("Test index: ", np.min(test_index), '-', np.max(test_index))
    
    train_features = features[train_index]
    test_features = features[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
    
    rf.fit(train_features, train_labels)
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    # Erro
    errors = abs(predictions - test_labels)
    
    # Accuracy
    accuracy = 100 - mean_absolute_error(test_labels, predictions)
    
    #print('Fields: ', importance_fields)
    
    # Variable importances
    importance_fields_aux = rf.feature_importances_
    importance_fields += importance_fields_aux
    
    #print('Fields aux: ', importance_fields_aux)
    
    # Append
    scores.append(accuracy)

#%% Scores
importance_fields_t = importance_fields/12
print('Accuracy: ', round(np.mean(scores), 2), '%.')

print('Total: ', round(np.sum(importance_fields_t),2))

#%% Variable importances
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 8)) for feature, importance in zip(feature_list_get_dummies, importance_fields_t)]

# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#%% Visualization of Variable Importances
import matplotlib.pyplot as plt

year_to_temp1 = importance_fields_t[0:5]
average_to_friend = importance_fields_t[5:10]
fri_to_wed = importance_fields_t[10:17]

# year to temp1
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
x1 = ['year', 'month', 'day', 'temp_2', 'temp_1']
y1 = [year_to_temp1[0],year_to_temp1[1],year_to_temp1[2],year_to_temp1[3],year_to_temp1[4]]
ax.bar(x1,y1)
plt.ylabel('Importance'); 
plt.xlabel('Variable'); 
plt.savefig('VI_P1.png', dpi=450, bbox_inches='tight');

#%%
import matplotlib.pyplot as plt

# list of x locations for plotting
x_values = list(range(len(importance_fields_t)))

# Make a bar chart
plt.bar(x_values, importance_fields_t, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list_get_dummies, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable'); 
plt.title('Variable Importances');
plt.savefig('VI.png', dpi=450, bbox_inches='tight')