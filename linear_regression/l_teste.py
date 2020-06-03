# Load the diabetes dataset
# Pandas is used for data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/data_test/temps_extended.csv'

# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv(path)

# Saving feature names for later use
feature_list = list(features.columns)

#%% One-hot encode
# One-hot encode categorical features
features = pd.get_dummies(features)

print('Shape of features after one-hot encoding:', features.shape)

#%% Saving labels and features
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
from sklearn import linear_model


scores = []
coefs = []

importance_fields = 0.0
importance_fields_aux = []

# Instance RF
l_regression = linear_model.LinearRegression()

kf_cv = KFold(n_splits=2191, random_state=None, shuffle=False)

for train_index, test_index in kf_cv.split(features):
    #print("Train index: ", np.min(train_index), '- ', np.max(train_index))
    print("Test index: ", np.min(test_index), '-', np.max(test_index))
    
    train_features = features[train_index]
    test_features = features[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
    
    l_regression.fit(train_features, train_labels)
    
    # Use the forest's predict method on the test data
    predictions = l_regression.predict(test_features)
    
    # Erro
    errors = abs(predictions - test_labels)
    
    # Accuracy
    accuracy = 100 - mean_absolute_error(test_labels, predictions)
    
    #print('Fields: ', importance_fields)
    
    # Variable importances
    #importance_fields_aux = l_regression.feature_importances_
    #importance_fields += importance_fields_aux
    
    #print('Fields aux: ', importance_fields_aux)
    
    # Append
    scores.append(accuracy)
    coefs.append(l_regression.coef_)

    
#%% Scores
#importance_fields_t = importance_fields/347
print('Acurácia: ', round(np.mean(scores), 2), '%.')
print('Min: ', round(np.min(scores), 2))
print('Max: ', round(np.max(scores), 2))

#print('Total: ', round(np.sum(importance_fields_t),2))
