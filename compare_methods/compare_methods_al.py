# -*- coding: utf-8 -*-
"""
Título: Comparação de métodos de KDD em dados de Alagoas

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

# Funções úteis
# Tempo de execução
def seconds_transform(seconds_time):
  hours = int(seconds_time/3600)
  rest_1 = seconds_time%3600
  minutes = int(rest_1/60)
  seconds = rest_1 - 60*minutes
  #print(seconds)
  print(" ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")
  return hours, minutes, round(seconds,2)

# Controle de build
def version_file(name_file, fields, rows_version):
    rows_aux = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file)
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value += 1
        rows_version['Version'] = value
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the data rows  
            csvwriter.writerows(rows_aux) 
    else:
        file_version_py = name_file
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the fields
            csvwriter.writeheader()
            # writing the data rows 
            csvwriter.writerows(rows_aux)
            #print ("File not exist")

#%% PREPARANDO OS DADOS

data_al2014 = pd.read_csv(r'tcc_data/AL_2014.csv')
data_al2015 = pd.read_csv(r'tcc_data/AL_2015.csv')
data_al2016 = pd.read_csv(r'tcc_data/AL_2016.csv')
data_al2017 = pd.read_csv(r'tcc_data/AL_2017.csv')
data_al2018 = pd.read_csv(r'tcc_data/AL_2018.csv')

labels_al = [] # Labels
features_al = [] # Features
features_al_list = [] # Guardando as variáveis das features
features_al_list_oh = [] # Variáveis das features com one-hot

#%% Pré-processamento e enriquecimento

def processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018):
    #% 2.1 - Limpeza
    del data_al2014['Unnamed: 0']
    del data_al2015['Unnamed: 0']
    del data_al2016['Unnamed: 0']
    del data_al2017['Unnamed: 0']
    del data_al2018['Unnamed: 0']

    # Escolhendo apenas as colunas de interesse
    data_al2014 = data_al2014.loc[:,'NT_GER':'QE_I26']
    data_al2015 = data_al2015.loc[:,'NT_GER':'QE_I26']
    data_al2016 = data_al2016.loc[:,'NT_GER':'QE_I26']
    data_al2017 = data_al2017.loc[:,'NT_GER':'QE_I26']
    data_al2018 = data_al2018.loc[:,'NT_GER':'QE_I26']

    data_al2014 = data_al2014.drop(data_al2014.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2015 = data_al2015.drop(data_al2015.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2016 = data_al2016.drop(data_al2016.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2017 = data_al2017.drop(data_al2017.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2018 = data_al2018.drop(data_al2018.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

    data_al2014 = data_al2014.drop(data_al2014.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2015 = data_al2015.drop(data_al2015.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2016 = data_al2016.drop(data_al2016.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2017 = data_al2017.drop(data_al2017.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2018 = data_al2018.drop(data_al2018.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    
    # MERGE NOS DADOS: data al
    frames = [data_al2014, data_al2015, data_al2016, data_al2017, data_al2018];
    data_al = pd.concat(frames);

    # Enriquecimento
    data_al['NT_GER'] = data_al['NT_GER'].str.replace(',','.')
    data_al['NT_GER'] = data_al['NT_GER'].astype(float)

    data_al_media = round(data_al['NT_GER'].mean(),2)
    
    data_al['NT_GER'] = data_al['NT_GER'].fillna(data_al_media)
    
    describe_al = data_al.describe()
    
    # 3 - Transformação
    labels_al = np.array(data_al['NT_GER'])

    # Removendo as features de notas
    data_al = data_al.drop(['NT_GER'], axis = 1)
    
    features_al_list = list(data_al.columns)


    # One hot encoding - QE_I01 a QE_I26
    features_al = pd.get_dummies(data=data_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
    # Salvando os nomes das colunas (features) com os dados para uso posterior
    # depois de codificar
    features_al_list_oh = list(features_al.columns)
    #
    # Convertendo para numpy
    features_al = np.array(features_al)
    
    return features_al, labels_al, features_al_list_oh

#%% Aplicando o pré-processamento

features_al, labels_al, features_al_list_oh = processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018)

#%% BIBLIOTECAS
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import time

n_cv = int(5);

train_x_al, test_x_al, train_y_al, test_y_al = train_test_split(features_al, labels_al, test_size=0.33, random_state=42)

#%% Cross Validation - Árvore de decisão

dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200, random_state=42)

time_dt_al_cv = time.time() # Time start DT CV
# min_samples_split = 320; min_samples_leaf = 200; max_features= log2
accuracy_al_dt_cv = cross_val_score(dt_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')
sec_dt_al_cv = (time.time() - time_dt_al_cv) # Time end DT CV

print('Accuracy DT CV: ', round(np.mean(accuracy_al_dt_cv), 4))
seconds_transform(sec_dt_al_cv)

#%% Escrevendo em Arquivo - DT
fields_al_dt_cv = ['Version','Metodo', 'Split', 'Leaf', 'Acc', 'Acc medio', 'Tempo (h,min,s)', 'n_cv']

rows_al_dt_cv = {'Version':0,'Metodo':'DT', 
                 'Split': 320, 'Leaf':200, 
                 'Acc': accuracy_al_dt_cv, 'Acc medio': accuracy_al_dt_cv.mean(),
                 'Tempo (h,min,s)':seconds_transform(sec_dt_al_cv), 'n_cv':n_cv}

file_al_dt_cv = "../tcc_codes/compare_methods/Logs/CV/DT_CV_AL.csv"

version_file(file_al_dt_cv, fields_al_dt_cv, rows_al_dt_cv)

#%% Cross Validation - RF

# min_samples_split=40, min_samples_leaf=20
rf_al = RandomForestRegressor(n_estimators=1000, min_samples_split=40, min_samples_leaf=20, random_state=42)

time_rf_al_cv = time.time()
accuracy_al_rf_cv = cross_val_score(rf_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')

sec_rf_al_cv = (time.time() - time_rf_al_cv)

print('Accuracy RF CV: ', round(np.mean(accuracy_al_rf_cv), 4))
seconds_transform(sec_rf_al_cv)

#%% Escrevendo em Arquivo - RF
fields_al_rf_cv = ['Version', 'Metodo', 'N_tree', 'Split', 'Leaf', 'Acc', 'Acc medio', 
                   'Tempo (h,min,s)', 'n_cv']

rows_al_rf_cv = {'Version':0,'Metodo':'RF',
                 'N_tree':'1000', 'Split':40, 'Leaf':20, 
                 'Acc':accuracy_al_rf_cv, 'Acc medio':accuracy_al_rf_cv.mean(), 
                 'Tempo (h,min,s)':seconds_transform(sec_rf_al_cv), 'n_cv':n_cv}

file_al_rf_cv = "../tcc_codes/compare_methods/Logs/CV/RF_CV_AL.csv"

version_file(file_al_rf_cv, fields_al_rf_cv, rows_al_rf_cv)

#%% LASSO

ls_al = linear_model.Lasso(alpha=0.005, positive=True, random_state=42)

time_ls_al_cv = time.time()
accuracy_al_ls_cv = cross_val_score(ls_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')
sec_ls_al_cv = (time.time() - time_ls_al_cv)

print('Accuracy LS CV: ', round(np.mean(accuracy_al_ls_cv), 4))
seconds_transform(sec_ls_al_cv)

#%% Escrevendo arquivo - LS
fields_al_ls_cv = ['Version','Metodo', 'Alfa', 'Acc','Acc medio', 'Tempo (h,min,s)', 'n_cv']

rows_al_ls_cv = {'Version':0,'Metodo':'LS',
                 'Alfa':0.005, 'Acc':accuracy_al_ls_cv, 
                 'Acc medio':accuracy_al_ls_cv.mean(),
                 'Tempo (h,min,s)':seconds_transform(sec_ls_al_cv), 
                 'n_cv':n_cv}

file_al_ls_cv = "../tcc_codes/compare_methods/Logs/CV/LS_CV_AL.csv"

version_file(file_al_ls_cv, fields_al_ls_cv, rows_al_ls_cv)

#%% Treinando os modelos

scores_al_rf = []
scores_al_dt_mae = [];
scores_al_dt_mse = [];

scores_al_dt = []
scores_al_rf_mae = [];
scores_al_rf_mse = [];


scores_al_ls = []
scores_al_ls_mae = [];
scores_al_ls_mse = [];

importance_fields_al_rf = 0.0
importance_fields_aux_al_rf = []

importance_fields_al_dt = 0.0
importance_fields_aux_al_dt = []

importance_fields_al_ls = 0.0
importance_fields_aux_al_ls = []

dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200, random_state=42)
rf_al = RandomForestRegressor(n_estimators=1000, min_samples_split=40, min_samples_leaf=20, random_state=42)
lasso_al = linear_model.Lasso(alpha=0.005, positive=True, random_state=42)

kf_cv_al = KFold(n_splits=n_cv, random_state=42, shuffle=False) # n_splits

#%% Treinando dados - DT_AL

time_dt_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    # Método 1 - Árvore de decisão
    
    dt_al.fit(train_features_al, train_labels_al)
    
    predictions_al_dt = dt_al.predict(test_features_al)
    
    accuracy_al_dt = dt_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_dt = mean_absolute_error(test_labels_al, predictions_al_dt)
    
    accuracy_mse_al_dt = mean_squared_error(test_labels_al, predictions_al_dt)
    
    # Importância de variável
    importance_fields_aux_al_dt = dt_al.feature_importances_
    importance_fields_al_dt += importance_fields_aux_al_dt
    
    # Append em cada valor médio
    scores_al_dt.append(accuracy_al_dt)
    
    scores_al_dt_mae.append(accuracy_mae_al_dt)
    
    scores_al_dt_mse.append(accuracy_mse_al_dt)

sec_dt_al = (time.time() - time_dt_al) # Time end dt loop

seconds_transform(sec_dt_al)

#%% Treino dos dados - RF_AL

time_rf_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    # Método 2 - Random Forest
    
    rf_al.fit(train_features_al, train_labels_al)
    
    predictions_al_rf = rf_al.predict(test_features_al)
    
    accuracy_al_rf = rf_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_rf = mean_absolute_error(test_labels_al, predictions_al_rf)
    
    accuracy_mse_al_rf = mean_squared_error(test_labels_al, predictions_al_rf)
     
    # Importância de variável
    importance_fields_aux_al_rf = rf_al.feature_importances_
    importance_fields_al_rf += importance_fields_aux_al_rf
    
    # Append em cada valor médio
    scores_al_rf.append(accuracy_al_rf)
    
    scores_al_rf_mae.append(accuracy_mae_al_rf)
    
    scores_al_rf_mse.append(accuracy_mse_al_rf)

sec_rf_al = (time.time() - time_rf_al) # Time end dt loop

seconds_transform(sec_rf_al)

#%% Treino dos dados - LS_AL

lasso_al = linear_model.Lasso(alpha=0.005, positive=True, random_state=42)

time_ls_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    
    # Método 3 - Lasso
    
    lasso_al.fit(train_features_al, train_labels_al)
    
    predictions_al_ls = lasso_al.predict(test_features_al)
    
    accuracy_al_ls = lasso_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_ls = mean_absolute_error(test_labels_al, predictions_al_ls)
    
    accuracy_mse_al_ls = mean_squared_error(test_labels_al, predictions_al_ls)    
    
    # Importância das variáveis
    importance_fields_aux_al_ls = lasso_al.coef_
    importance_fields_al_ls += importance_fields_aux_al_ls
    
    # Append em cada valor médio
    scores_al_ls.append(accuracy_al_ls)
    
    scores_al_ls_mae.append(accuracy_mae_al_ls)
    
    scores_al_ls_mse.append(accuracy_mse_al_ls)

sec_ls_al = (time.time() - time_ls_al) # Time end dt loop

seconds_transform(sec_ls_al)

#%% Testando - RF
predictions_al_rf = rf_al.predict(test_x_al)
    
accuracy_al_rf_f = rf_al.score(test_x_al, test_y_al)

accuracy_mae_al_rf_f = mean_absolute_error(test_y_al, predictions_al_rf)
    
accuracy_mse_al_rf_f = mean_squared_error(test_y_al, predictions_al_rf)

print('Accuracy AL RF: ', round(accuracy_al_rf_f, 4))
print('Accuracy MAE AL RF: ', round(accuracy_mae_al_rf_f, 4))
print('Accuracy MSE AL RF: ', round(accuracy_mse_al_rf_f, 4))

#%% Escrevendo em arquivo - RF
fields_al_rf = ['Version','Metodo', 'R2', 'MAE', 'MSE', 'Tempo (h,min,s)']

rows_al_rf = {'Version':0,'Metodo':'RF', 'R2':round(accuracy_al_rf_f,4), 'MAE':round(accuracy_mae_al_rf_f,4),
               'MSE':round(accuracy_mse_al_rf_f,4), 'Tempo (h,min,s)':seconds_transform(sec_rf_al)}

file_al_rf = "../tcc_codes/compare_methods/Logs/METRICS_EVALUATE/RF_AL.csv"

version_file(file_al_rf, fields_al_rf, rows_al_rf)

#%% Testando - DT
predictions_al_dt = dt_al.predict(test_x_al)
    
accuracy_al_dt_f = dt_al.score(test_x_al, test_y_al)

accuracy_mae_al_dt_f = mean_absolute_error(test_y_al, predictions_al_dt)
    
accuracy_mse_al_dt_f = mean_squared_error(test_y_al, predictions_al_dt)

print('Final Accuracy AL DT: ', round(accuracy_al_dt_f, 4))
print('Final Accuracy MAE AL DT: ', round(accuracy_mae_al_dt_f, 4))
print('Final Accuracy MSE AL DT: ', round(accuracy_mse_al_dt_f, 4))

#%% Escrevendo em arquivo - DT
fields_al_dt = ['Version','Metodo', 'R2', 'MAE', 'MSE', 'Tempo (h,min,s)']

rows_al_dt = {'Version':0,'Metodo':'DT', 
              'R2':round(accuracy_al_dt_f,4), 
              'MAE':round(accuracy_mae_al_dt_f,4),
              'MSE':round(accuracy_mse_al_dt_f,4), 
              'Tempo (h,min,s)':seconds_transform(sec_dt_al)}

file_al_dt = "../tcc_codes/compare_methods/Logs/METRICS_EVALUATE/DT_AL.csv"

version_file(file_al_dt, fields_al_dt, rows_al_dt)
    
#%% Testando LS
predictions_al_ls = lasso_al.predict(test_x_al)
    
accuracy_al_ls_f = lasso_al.score(test_x_al, test_y_al)

accuracy_mae_al_ls_f = mean_absolute_error(test_y_al, predictions_al_ls)
    
accuracy_mse_al_ls_f = mean_squared_error(test_y_al, predictions_al_ls)

print('Final Accuracy AL LS: ', round(accuracy_al_ls_f, 4))
print('Final Accuracy MAE AL LS: ', round(accuracy_mae_al_ls_f, 4))
print('Final Accuracy MSE AL LS: ', round(accuracy_mse_al_ls_f, 4))

#%% Escrevendo em arquivo - LS
fields_al_ls = ['Version','Metodo', 'R2', 'MAE', 'MSE', 'Tempo (h,min,s)']

rows_al_ls = {'Version':0, 'Metodo':'LS', 
              'R2':round(accuracy_al_ls_f,4), 
              'MAE':round(accuracy_mae_al_ls_f,4),
              'MSE':round(accuracy_mse_al_ls_f,4), 
              'Tempo (h,min,s)':seconds_transform(sec_ls_al)}

file_al_ls = "../tcc_codes/compare_methods/Logs/METRICS_EVALUATE/LS_AL.csv"

version_file(file_al_ls, fields_al_ls, rows_al_ls)

#%% Acurácia AL

importance_fields_al_rf_t = importance_fields_al_rf/n_cv
importance_fields_al_dt_t = importance_fields_al_dt/n_cv
importance_fields_al_ls_t = importance_fields_al_ls/n_cv

print('VImp Total AL RF: ', round(np.sum(importance_fields_al_rf_t),2));
print('VImp Total AL DT: ', round(np.sum(importance_fields_al_dt_t),2));
print('VImp Total AL LS: ', round(np.sum(importance_fields_al_ls_t),2));

#%% Importancia das variáveis
# Lista de tupla com as variáveis de importância - Random Forest
feature_importances_al_rf = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_rf_t)]

# Print out the feature and importances
[print('Variable RF: {:20} Importance RF: {}'.format(*pair)) for pair in feature_importances_al_rf];

print("\n")

# Lista de tupla com as variáveis de importância - Árvore de decisão
feature_importances_al_dt = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_dt_t)]

# Print out the feature and importances
[print('Variable DT: {:20} Importance DT: {}'.format(*pair)) for pair in feature_importances_al_dt];

#%% Lista de tupla com as variáveis de importância - Lasso
feature_importances_al_ls = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_ls_t)]

# Print out the feature and importances
[print('Variable LS: {:20} Importance LS: {}'.format(*pair)) for pair in feature_importances_al_ls];

#%% Separando os valores
# Random Forest
I01_AL_RF = importance_fields_al_rf_t[0:5]; I02_AL_RF = importance_fields_al_rf_t[5:11]; 

I03_AL_RF = importance_fields_al_rf_t[11:14]; I04_AL_RF = importance_fields_al_rf_t[14:20]; 

I05_AL_RF = importance_fields_al_rf_t[20:26]; I06_AL_RF = importance_fields_al_rf_t[26:32];

I07_AL_RF = importance_fields_al_rf_t[32:40]; I08_AL_RF = importance_fields_al_rf_t[40:47]; 

I09_AL_RF = importance_fields_al_rf_t[47:53]; I10_AL_RF = importance_fields_al_rf_t[53:58]; 

I11_AL_RF = importance_fields_al_rf_t[58:69]; I12_AL_RF = importance_fields_al_rf_t[69:75];

I13_AL_RF = importance_fields_al_rf_t[75:81]; I14_AL_RF = importance_fields_al_rf_t[81:87]; 

I15_AL_RF = importance_fields_al_rf_t[87:93]; I16_AL_RF = importance_fields_al_rf_t[93:94]; 

I17_AL_RF = importance_fields_al_rf_t[94:100]; I18_AL_RF = importance_fields_al_rf_t[100:105]; 

I19_AL_RF = importance_fields_al_rf_t[105:112]; I20_AL_RF = importance_fields_al_rf_t[112:123]; 

I21_AL_RF = importance_fields_al_rf_t[123:125]; I22_AL_RF = importance_fields_al_rf_t[125:130]; 

I23_AL_RF = importance_fields_al_rf_t[130:135]; I24_AL_RF = importance_fields_al_rf_t[135:140];

I25_AL_RF = importance_fields_al_rf_t[140:148]; I26_AL_RF = importance_fields_al_rf_t[148:157];

# Decision Tree
I01_AL_DT = importance_fields_al_dt_t[0:5]; I02_AL_DT = importance_fields_al_dt_t[5:11]; 

I03_AL_DT = importance_fields_al_dt_t[11:14]; I04_AL_DT = importance_fields_al_dt_t[14:20]; 

I05_AL_DT = importance_fields_al_dt_t[20:26]; I06_AL_DT = importance_fields_al_dt_t[26:32];

I07_AL_DT = importance_fields_al_dt_t[32:40]; I08_AL_DT = importance_fields_al_dt_t[40:47]; 

I09_AL_DT = importance_fields_al_dt_t[47:53]; I10_AL_DT = importance_fields_al_dt_t[53:58]; 

I11_AL_DT = importance_fields_al_dt_t[58:69]; I12_AL_DT = importance_fields_al_dt_t[69:75];

I13_AL_DT = importance_fields_al_dt_t[75:81]; I14_AL_DT = importance_fields_al_dt_t[81:87]; 

I15_AL_DT = importance_fields_al_dt_t[87:93]; I16_AL_DT = importance_fields_al_dt_t[93:94]; 

I17_AL_DT = importance_fields_al_dt_t[94:100]; I18_AL_DT = importance_fields_al_dt_t[100:105]; 

I19_AL_DT = importance_fields_al_dt_t[105:112]; I20_AL_DT = importance_fields_al_dt_t[112:123]; 

I21_AL_DT = importance_fields_al_dt_t[123:125]; I22_AL_DT = importance_fields_al_dt_t[125:130]; 

I23_AL_DT = importance_fields_al_dt_t[130:135]; I24_AL_DT = importance_fields_al_dt_t[135:140];

I25_AL_DT = importance_fields_al_dt_t[140:148]; I26_AL_DT = importance_fields_al_dt_t[148:157];

#%% Lasso
I01_AL_LS = importance_fields_al_ls_t[0:5]; I02_AL_LS = importance_fields_al_ls_t[5:11]; 

I03_AL_LS = importance_fields_al_ls_t[11:14]; I04_AL_LS = importance_fields_al_ls_t[14:20]; 

I05_AL_LS = importance_fields_al_ls_t[20:26]; I06_AL_LS = importance_fields_al_ls_t[26:32];

I07_AL_LS = importance_fields_al_ls_t[32:40]; I08_AL_LS = importance_fields_al_ls_t[40:47]; 

I09_AL_LS = importance_fields_al_ls_t[47:53]; I10_AL_LS = importance_fields_al_ls_t[53:58]; 

I11_AL_LS = importance_fields_al_ls_t[58:69]; I12_AL_LS = importance_fields_al_ls_t[69:75];

I13_AL_LS = importance_fields_al_ls_t[75:81]; I14_AL_LS = importance_fields_al_ls_t[81:87]; 

I15_AL_LS = importance_fields_al_ls_t[87:93]; I16_AL_LS = importance_fields_al_ls_t[93:94]; 

I17_AL_LS = importance_fields_al_ls_t[94:100]; I18_AL_LS = importance_fields_al_ls_t[100:105]; 

I19_AL_LS = importance_fields_al_ls_t[105:112]; I20_AL_LS = importance_fields_al_ls_t[112:123]; 

I21_AL_LS = importance_fields_al_ls_t[123:125]; I22_AL_LS = importance_fields_al_ls_t[125:130]; 

I23_AL_LS = importance_fields_al_ls_t[130:135]; I24_AL_LS = importance_fields_al_ls_t[135:140];

I25_AL_LS = importance_fields_al_ls_t[140:148]; I26_AL_LS = importance_fields_al_ls_t[148:157];


#%% Lasso percentual (normalizando)

sum_variables_al = np.sum(importance_fields_al_ls_t)

I01_AL_LS = I01_AL_LS/sum_variables_al; I02_AL_LS = I02_AL_LS/sum_variables_al; 

I03_AL_LS = I03_AL_LS/sum_variables_al; I04_AL_LS = I04_AL_LS/sum_variables_al; 

I05_AL_LS = I05_AL_LS/sum_variables_al; I06_AL_LS = I06_AL_LS/sum_variables_al;

I07_AL_LS = I07_AL_LS/sum_variables_al; I08_AL_LS = I08_AL_LS/sum_variables_al; 

I09_AL_LS = I09_AL_LS/sum_variables_al; I10_AL_LS = I10_AL_LS/sum_variables_al; 

I11_AL_LS = I11_AL_LS/sum_variables_al; I12_AL_LS = I12_AL_LS/sum_variables_al;

I13_AL_LS = I13_AL_LS/sum_variables_al; I14_AL_LS = I14_AL_LS/sum_variables_al; 

I15_AL_LS = I15_AL_LS/sum_variables_al; I16_AL_LS = I16_AL_LS/sum_variables_al; 

I17_AL_LS = I17_AL_LS/sum_variables_al; I18_AL_LS = I18_AL_LS/sum_variables_al; 

I19_AL_LS = I19_AL_LS/sum_variables_al; I20_AL_LS = I20_AL_LS/sum_variables_al; 

I21_AL_LS = I21_AL_LS/sum_variables_al; I22_AL_LS = I22_AL_LS/sum_variables_al; 

I23_AL_LS = I23_AL_LS/sum_variables_al; I24_AL_LS = I24_AL_LS/sum_variables_al;

I25_AL_LS = I25_AL_LS/sum_variables_al; I26_AL_LS = I26_AL_LS/sum_variables_al;

'''

sum_ = np.sum(I01_AL_LS) + np.sum(I02_AL_LS) + np.sum(I03_AL_LS) + np.sum(I04_AL_LS)+ \
    np.sum(I05_AL_LS) + np.sum(I06_AL_LS)+ np.sum(I07_AL_LS) + np.sum(I08_AL_LS) + \
        np.sum(I09_AL_LS)+ np.sum(I10_AL_LS)+ np.sum(I11_AL_LS)+ np.sum(I12_AL_LS)+ \
            np.sum(I13_AL_LS)+ np.sum(I14_AL_LS)+ np.sum(I15_AL_LS)+ np.sum(I16_AL_LS)+ \
                np.sum(I17_AL_LS)+ np.sum(I18_AL_LS)+ np.sum(I19_AL_LS)+ np.sum(I20_AL_LS)+ \
                    np.sum(I21_AL_LS)+ np.sum(I22_AL_LS)+ np.sum(I23_AL_LS)+ np.sum(I24_AL_LS)+ \
                        np.sum(I25_AL_LS)+ np.sum(I26_AL_LS)

print(sum_)

'''

#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1_rf = [I01_AL_RF[0],I01_AL_RF[1],I01_AL_RF[2],I01_AL_RF[3],I01_AL_RF[4]];
y1_rf = list(map(lambda t:t*100, y1_rf))
y1_dt = [I01_AL_DT[0],I01_AL_DT[1],I01_AL_DT[2],I01_AL_DT[3],I01_AL_DT[4]];
y1_dt = list(map(lambda t:t*100, y1_dt))
y1_ls = [I01_AL_LS[0],I01_AL_LS[1],I01_AL_LS[2],I01_AL_LS[3],I01_AL_LS[4]];
y1_ls = list(map(lambda t:t*100, y1_ls))

# Configurando a posição no eixo x
axis1 = np.arange(len(y1_rf))
y11 = [x + bar_width for x in axis1]
y12 = [x + bar_width for x in y11]
y13 = [x + bar_width for x in y12]

# Fazendo o plot
plt.bar(y11, y1_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y12, y1_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y13, y1_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y1_rf))], \
           ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'],\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)');
plt.xlabel('Variável (AL)');
plt.title('Estado civil');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I01_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
#bar_width = 0.1;

x2 = ['Branca','Preta','Amarela','Parda','Indígena','Não quero declarar'];
y2_rf = [I02_AL_RF[0],I02_AL_RF[1],I02_AL_RF[2],I02_AL_RF[3],I02_AL_RF[4],I02_AL_RF[5]];
y2_rf = list(map(lambda t:t*100, y2_rf))
y2_dt = [I02_AL_DT[0],I02_AL_DT[1],I02_AL_DT[2],I02_AL_DT[3],I02_AL_DT[4],I02_AL_DT[5]];
y2_dt = list(map(lambda t:t*100, y2_dt))
y2_ls = [I02_AL_LS[0],I02_AL_LS[1],I02_AL_LS[2],I02_AL_LS[3],I02_AL_LS[4],I02_AL_LS[5]];
y2_ls = list(map(lambda t:t*100, y2_ls))

# Configurando a posição no eixo x
axis2 = np.arange(len(y2_rf))
y21 = [x + bar_width for x in axis2]
y22 = [x + bar_width for x in y21]
y23 = [x + bar_width for x in y22]

# Fazendo o plot
plt.bar(y21, y2_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y22, y2_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y23, y2_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y2_rf))], \
           x2,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Cor/raça');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I02_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
#bar_width = 0.1;

x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3_rf = [I03_AL_RF[0],I03_AL_RF[1],I03_AL_RF[2]];
y3_rf = list(map(lambda t:t*100, y3_rf))
y3_dt = [I03_AL_DT[0],I03_AL_DT[1],I03_AL_DT[2]];
y3_dt = list(map(lambda t:t*100, y3_dt))
y3_ls = [I03_AL_LS[0],I03_AL_LS[1],I03_AL_LS[2]];
y3_ls = list(map(lambda t:t*100, y3_ls))

# Configurando a posição no eixo x
axis3 = np.arange(len(y3_rf))
y31 = [x + bar_width for x in axis3]
y32 = [x + bar_width for x in y31]
y33 = [x + bar_width for x in y32]

# Fazendo o plot
plt.bar(y31, y3_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y32, y3_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y33, y3_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y3_rf))], \
           x3,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Nacionalidade');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I03_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
#bar_width = 0.1;

x4 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y4_rf = [I04_AL_RF[0],I04_AL_RF[1],I04_AL_RF[2],I04_AL_RF[3],I04_AL_RF[4],I04_AL_RF[5]];
y4_rf = list(map(lambda t:t*100, y4_rf))
y4_dt = [I04_AL_DT[0],I04_AL_DT[1],I04_AL_DT[2],I04_AL_DT[3],I04_AL_DT[4],I04_AL_DT[5]];
y4_dt = list(map(lambda t:t*100, y4_dt));
y4_ls = [I04_AL_LS[0],I04_AL_LS[1],I04_AL_LS[2],I04_AL_LS[3],I04_AL_LS[4],I04_AL_LS[5]];
y4_ls = list(map(lambda t:t*100, y4_ls));

# Configurando a posição no eixo x
axis4 = np.arange(len(y4_rf))
y41 = [x + bar_width for x in axis4]
y42 = [x + bar_width for x in y41]
y43 = [x + bar_width for x in y42]


# Fazendo o plot
plt.bar(y41, y4_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y42, y4_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y43, y4_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y4_rf))], \
           x4,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Escolarização da pai');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I04_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
#bar_width = 0.1;

x5 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y5_rf = [I05_AL_RF[0],I05_AL_RF[1],I05_AL_RF[2],I05_AL_RF[3],I05_AL_RF[4],I05_AL_RF[5]];
y5_rf = list(map(lambda t:t*100, y5_rf))
y5_dt = [I05_AL_DT[0],I05_AL_DT[1],I05_AL_DT[2],I05_AL_DT[3],I05_AL_DT[4],I05_AL_DT[5]];
y5_dt = list(map(lambda t:t*100, y5_dt));
y5_ls = [I05_AL_LS[0],I05_AL_LS[1],I05_AL_LS[2],I05_AL_LS[3],I05_AL_LS[4],I05_AL_LS[5]];
y5_ls = list(map(lambda t:t*100, y5_ls));


# Configurando a posição no eixo x
axis5 = np.arange(len(y5_rf))
y51 = [x + bar_width for x in axis5]
y52 = [x + bar_width for x in y51]
y53 = [x + bar_width for x in y52]

# Fazendo o plot
plt.bar(y51, y5_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y52, y5_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y53, y5_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y5_rf))], \
           x5,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Escolarização da mãe');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I05_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
#bar_width = 0.1;

x6 = ['Casa/apartamento(sozinho)','Casa/apartamento(pais/parentes)',
      'Casa/apartamento(cônjugue/filhos)','Casa/apartamento(outras pessoas)',
      'Alojamento univ na IES','Outro'];
y6_rf = [I06_AL_RF[0],I06_AL_RF[1],I06_AL_RF[2],I06_AL_RF[3],I06_AL_RF[4],I06_AL_RF[5]];
y6_rf = list(map(lambda t:t*100, y6_rf));
y6_dt = [I06_AL_DT[0],I06_AL_DT[1],I06_AL_DT[2],I06_AL_DT[3],I06_AL_DT[4],I06_AL_DT[5]];
y6_dt = list(map(lambda t:t*100, y6_dt));
y6_ls = [I06_AL_LS[0],I06_AL_LS[1],I06_AL_LS[2],I06_AL_LS[3],I06_AL_LS[4],I06_AL_LS[5]];
y6_ls = list(map(lambda t:t*100, y6_ls));

# Configurando a posição no eixo x
axis6 = np.arange(len(y6_rf))
y61 = [x + bar_width for x in axis6]
y62 = [x + bar_width for x in y61]
y63 = [x + bar_width for x in y62]

# Fazendo o plot
plt.bar(y61, y6_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y62, y6_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y63, y6_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y6_rf))], \
           x6,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Onde e com quem moro');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I06_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
#bar_width = 0.1;

x7 = ['Nenhuma','Uma','Duas','Três','Quatro','Cinco','Seis','Sete ou mais'];
y7_rf = [I07_AL_RF[0],I07_AL_RF[1],I07_AL_RF[2],I07_AL_RF[3],
         I07_AL_RF[4],I07_AL_RF[5],I07_AL_RF[6],I07_AL_RF[7]];
y7_rf = list(map(lambda t:t*100, y7_rf));
y7_dt = [I07_AL_DT[0],I07_AL_DT[1],I07_AL_DT[2],I07_AL_DT[3],
         I07_AL_DT[4],I07_AL_DT[5],I07_AL_DT[6],I07_AL_DT[7]];
y7_dt = list(map(lambda t:t*100, y7_dt));
y7_ls = [I07_AL_LS[0],I07_AL_LS[1],I07_AL_LS[2],I07_AL_LS[3],
         I07_AL_LS[4],I07_AL_LS[5],I07_AL_LS[6],I07_AL_LS[7]];
y7_ls = list(map(lambda t:t*100, y7_ls));

# Configurando a posição no eixo x
axis7 = np.arange(len(y7_rf))
y71 = [x + bar_width for x in axis7]
y72 = [x + bar_width for x in y71]
y73 = [x + bar_width for x in y72]

# Fazendo o plot
plt.bar(y71, y7_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y72, y7_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y73, y7_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y7_rf))], \
           x7,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Quantos moram com o estudante');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I07_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
#bar_width = 0.1;

x8 = ['Até 1,5 sál mín','1 a 3 sál mín.','3 a 4,5 sál mín',
      '4,5 a 6 sál mín','6 a 10 sál mín','30 a 10 sál mín',
      'Acima de 30 sál mín'];
y8_rf = [I08_AL_RF[0],I08_AL_RF[1],I08_AL_RF[2],I08_AL_RF[3],
         I08_AL_RF[4],I08_AL_RF[5],I08_AL_RF[6]];
y8_rf = list(map(lambda t:t*100, y8_rf));
y8_dt = [I08_AL_DT[0],I08_AL_DT[1],I08_AL_DT[2],I08_AL_DT[3],
         I08_AL_DT[4],I08_AL_DT[5],I08_AL_DT[6]];
y8_dt = list(map(lambda t:t*100, y8_dt));
y8_ls = [I08_AL_LS[0],I08_AL_LS[1],I08_AL_LS[2],I08_AL_LS[3],
         I08_AL_LS[4],I08_AL_LS[5],I08_AL_LS[6]];
y8_ls = list(map(lambda t:t*100, y8_ls));

# Configurando a posição no eixo x
axis8 = np.arange(len(y8_rf))
y81 = [x + bar_width for x in axis8]
y82 = [x + bar_width for x in y81]
y83 = [x + bar_width for x in y82]

# Fazendo o plot
plt.bar(y81, y8_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y82, y8_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y83, y8_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')

    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y8_rf))], \
           x8,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Renda total');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I08_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I09
fig9 = plt.figure();
ax9 = fig9.add_axes([0,0,1,1]);
#bar_width = 0.1;

x9 = ['Sem renda (fonte: governamental)','Sem renda (fonte: família/outros)',
      'Tenho renda; recebo ajuda (família/colegas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a família',
      'Sou o provedor da família'];
y9_rf = [I09_AL_RF[0],I09_AL_RF[1],I09_AL_RF[2],I09_AL_RF[3],
         I09_AL_RF[4],I09_AL_RF[5]];
y9_rf = list(map(lambda t:t*100, y9_rf));
y9_dt = [I09_AL_DT[0],I07_AL_DT[1],I09_AL_DT[2],I07_AL_DT[3],
         I09_AL_DT[4],I09_AL_DT[5]];
y9_dt = list(map(lambda t:t*100, y9_dt));
y9_ls = [I09_AL_LS[0],I07_AL_LS[1],I09_AL_LS[2],I07_AL_LS[3],
         I09_AL_LS[4],I09_AL_LS[5]];
y9_ls = list(map(lambda t:t*100, y9_ls))

# Configurando a posição no eixo x
axis9 = np.arange(len(y9_rf))
y91 = [x + bar_width for x in axis9]
y92 = [x + bar_width for x in y91]
y93 = [x + bar_width for x in y92]

# Fazendo o plot
plt.bar(y91, y9_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y92, y9_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y93, y9_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y9_rf))], \
           x9,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Situação financeira');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I09_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
#bar_width = 0.1;

x10 = ['Não trabalho','Trabalho eventualmente','Trablho (até 20h/sem)',
       'Trabalho (21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10_rf = [I10_AL_RF[0],I10_AL_RF[1],I10_AL_RF[2],I10_AL_RF[3],
         I10_AL_RF[4]];
y10_rf = list(map(lambda t:t*100, y10_rf));
y10_dt = [I10_AL_DT[0],I10_AL_DT[1],I10_AL_DT[2],I10_AL_DT[3],
         I10_AL_DT[4]];
y10_dt = list(map(lambda t:t*100, y10_dt));
y10_ls = [I10_AL_LS[0],I10_AL_LS[1],I10_AL_LS[2],I10_AL_LS[3],
         I10_AL_LS[4]];
y10_ls = list(map(lambda t:t*100, y10_ls));


# Configurando a posição no eixo x
axis10 = np.arange(len(y10_rf))
y101 = [x + bar_width for x in axis10]
y102 = [x + bar_width for x in y101]
y103 = [x + bar_width for x in y102]

# Fazendo o plot
plt.bar(y101, y10_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y102, y10_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y103, y10_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y10_rf))], \
           x10,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Situação de trabalho');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I10_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I11
fig11 = plt.figure();
ax11 = fig11.add_axes([0,0,1,1]);
#bar_width = 0.1;

x11 = ['Nenhum (curso gratuito)','Nenhum (mas não gratuito)','ProUni integral',
       'ProUni parcial;apenas','FIES;apenas','ProUni e FIES',
       'Bolsa do governo (E/D/MUN)',
       'Bolsa pela IES','Bolsa por outra entidade','Financiamento pela IES',
       'Financiamento bancário'];
y11_rf = [I11_AL_RF[0],I11_AL_RF[1],I11_AL_RF[2],I11_AL_RF[3], I11_AL_RF[4],
          I11_AL_RF[5],I11_AL_RF[6],I11_AL_RF[7],I11_AL_RF[8], I11_AL_RF[9], I11_AL_RF[10]];
y11_rf = list(map(lambda t:t*100, y11_rf));
y11_dt = [I11_AL_DT[0],I11_AL_DT[1],I11_AL_DT[2],I11_AL_DT[3],I11_AL_DT[4],
          I11_AL_DT[5],I11_AL_DT[6],I11_AL_DT[7],I11_AL_DT[8], I11_AL_DT[9], I11_AL_DT[10]];
y11_dt = list(map(lambda t:t*100, y11_dt));
y11_ls = [I11_AL_LS[0],I11_AL_LS[1],I11_AL_LS[2],I11_AL_LS[3],I11_AL_LS[4],
          I11_AL_LS[5],I11_AL_LS[6],I11_AL_LS[7],I11_AL_LS[8], I11_AL_LS[9], I11_AL_LS[10]];
y11_ls = list(map(lambda t:t*100, y11_ls));

# Configurando a posição no eixo x
axis11 = np.arange(len(y11_rf))
y111 = [x + bar_width for x in axis11]
y112 = [x + bar_width for x in y111]
y113 = [x + bar_width for x in y112]

# Fazendo o plot
plt.bar(y111, y11_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y112, y11_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y113, y11_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y11_rf))], \
           x11,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I11_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
#bar_width = 0.1;

x12 = ['Nenhum','Moradia','Alimentação','Moradia e alimentação', 'Permanência','Outros'];
y12_rf = [I12_AL_RF[0],I12_AL_RF[1],I12_AL_RF[2],I12_AL_RF[3], I12_AL_RF[4],
          I12_AL_RF[5]];
y12_rf = list(map(lambda t:t*100, y12_rf));
y12_dt = [I12_AL_DT[0],I12_AL_DT[1],I12_AL_DT[2],I12_AL_DT[3],I12_AL_DT[4],
          I12_AL_DT[5]];
y12_dt = list(map(lambda t:t*100, y12_dt));
y12_ls = [I12_AL_LS[0],I12_AL_LS[1],I12_AL_LS[2],I12_AL_LS[3],I12_AL_LS[4],
          I12_AL_LS[5]];
y12_ls = list(map(lambda t:t*100, y12_ls));

# Configurando a posição no eixo x
axis12 = np.arange(len(y12_rf))
y121 = [x + bar_width for x in axis12]
y122 = [x + bar_width for x in y121]
y123 = [x + bar_width for x in y122]

# Fazendo o plot
plt.bar(y121, y12_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y122, y12_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y123, y12_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y12_rf))], \
           x12,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Auxílio permanência');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I12_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
#bar_width = 0.1;

x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extensão','Monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13_rf = [I13_AL_RF[0],I13_AL_RF[1],I13_AL_RF[2],I13_AL_RF[3], I13_AL_RF[4],
          I13_AL_RF[5]];
y13_rf = list(map(lambda t:t*100, y13_rf));
y13_dt = [I13_AL_DT[0],I13_AL_DT[1],I13_AL_DT[2],I13_AL_DT[3],I13_AL_DT[4],
          I13_AL_DT[5]];
y13_dt = list(map(lambda t:t*100, y13_dt));
y13_ls = [I13_AL_LS[0],I13_AL_LS[1],I13_AL_LS[2],I13_AL_LS[3],I13_AL_LS[4],
          I13_AL_LS[5]];
y13_ls = list(map(lambda t:t*100, y13_ls));


# Configurando a posição no eixo x
axis13 = np.arange(len(y13_rf))
y131 = [x + bar_width for x in axis13]
y132 = [x + bar_width for x in y131]
y133 = [x + bar_width for x in y132]

# Fazendo o plot
plt.bar(y131, y13_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y132, y13_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y133, y13_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y13_rf))], \
           x13,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Tipo de bolsa recebido');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I13_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
#bar_width = 0.1;

x14 = ['Não','Sim, Ciências sem Fronteiras', 'Sim, intercâmbio pelo Gov Fed',
       'Sim, intercâmbio pelo Gov Est', 'Sim, intercâmbio pela minha IES',
       'Sim, intercâmbio não institucional'];
y14_rf = [I14_AL_RF[0],I14_AL_RF[1],I14_AL_RF[2],I14_AL_RF[3], I14_AL_RF[4],
          I14_AL_RF[5]];
y14_rf = list(map(lambda t:t*100, y14_rf));
y14_dt = [I14_AL_DT[0],I14_AL_DT[1],I14_AL_DT[2],I14_AL_DT[3],I14_AL_DT[4],
          I14_AL_DT[5]];
y14_dt = list(map(lambda t:t*100, y14_dt));
y14_ls = [I14_AL_LS[0],I14_AL_LS[1],I14_AL_LS[2],I14_AL_LS[3],I14_AL_LS[4],
          I14_AL_LS[5]];
y14_ls = list(map(lambda t:t*100, y14_ls));

# Configurando a posição no eixo x
axis14 = np.arange(len(y14_rf))
y141 = [x + bar_width for x in axis14]
y142 = [x + bar_width for x in y141]
y143 = [x + bar_width for x in y142]

# Fazendo o plot
plt.bar(y141, y14_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y142, y14_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y143, y14_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y14_rf))], \
           x14,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Programas de atividade no exterior');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I14_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1]);
#bar_width = 0.1;

x15 = ['Não','Sim, étnico-racial','Sim, renda', 'Sim, esc púb ou part (c/ bolsa)',
       'Sim, combina dois ou mais', 'Sim, outra'];
y15_rf = [I15_AL_RF[0],I15_AL_RF[1],I15_AL_RF[2],I15_AL_RF[3], I15_AL_RF[4],
          I15_AL_RF[5]];
y15_rf = list(map(lambda t:t*100, y15_rf));
y15_dt = [I15_AL_DT[0],I15_AL_DT[1],I15_AL_DT[2],I15_AL_DT[3],I15_AL_DT[4],
          I15_AL_DT[5]];
y15_dt = list(map(lambda t:t*100, y15_dt));
y15_ls = [I15_AL_LS[0],I15_AL_LS[1],I15_AL_LS[2],I15_AL_LS[3],I15_AL_LS[4],
          I15_AL_LS[5]];
y15_ls = list(map(lambda t:t*100, y15_ls));

# Configurando a posição no eixo x
axis15 = np.arange(len(y15_rf))
y151 = [x + bar_width for x in axis15]
y152 = [x + bar_width for x in y151]
y153 = [x + bar_width for x in y152]

# Fazendo o plot
plt.bar(y151, y15_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y152, y15_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y153, y15_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y15_rf))], \
           x15,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Ingresso por cota');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I15_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
#bar_width = 0.1;

x16 = ['AL'];
y16_rf = [I16_AL_RF[0]];
y16_rf = list(map(lambda t:t*100, y16_rf));
y16_dt = [I16_AL_DT[0]];
y16_dt = list(map(lambda t:t*100, y16_dt));
y16_ls = [I16_AL_LS[0]];
y16_ls = list(map(lambda t:t*100, y16_ls));

# Configurando a posição no eixo x
axis16 = np.arange(len(y16_rf))
y161 = [x + bar_width for x in axis16]
y162 = [x + bar_width for x in y161]
y163 = [x + bar_width for x in y162]

# Fazendo o plot
plt.bar(y161, y16_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y162, y16_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y163, y16_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y16_rf))], \
           x16,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('UF que concluiu o médio');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I16_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
#bar_width = 0.1;

x17 = ['Todo em esc púb', 'Todo em esc priv','Todo no ext',
       'Maior parte em esc púb','Maior parte em esc priv',
       'Parte no Brasil'];
y17_rf = [I17_AL_RF[0],I17_AL_RF[1],I17_AL_RF[2],I17_AL_RF[3], I17_AL_RF[4],
          I17_AL_RF[5]];
y17_rf = list(map(lambda t:t*100, y17_rf));
y17_dt = [I17_AL_DT[0],I17_AL_DT[1],I17_AL_DT[2],I17_AL_DT[3],I17_AL_DT[4],
          I17_AL_DT[5]];
y17_dt = list(map(lambda t:t*100, y17_dt));
y17_ls = [I17_AL_LS[0],I17_AL_LS[1],I17_AL_LS[2],I17_AL_LS[3],I17_AL_LS[4],
          I17_AL_LS[5]];
y17_ls = list(map(lambda t:t*100, y17_ls));

# Configurando a posição no eixo x
axis17 = np.arange(len(y17_rf))
y171 = [x + bar_width for x in axis17]
y172 = [x + bar_width for x in y171]
y173 = [x + bar_width for x in y172]

# Fazendo o plot
plt.bar(y171, y17_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y172, y17_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y173, y17_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y17_rf))], \
           x17,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Tipo de escola no médio');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I17_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
#bar_width = 0.1;

x18 = ['Tradicional', 'Prof técnico', 'Prof magistério (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18_rf = [I18_AL_RF[0],I18_AL_RF[1],I18_AL_RF[2],I18_AL_RF[3], I18_AL_RF[4]];
y18_rf = list(map(lambda t:t*100, y18_rf));
y18_dt = [I18_AL_DT[0],I18_AL_DT[1],I18_AL_DT[2],I18_AL_DT[3],I18_AL_DT[4]];
y18_dt = list(map(lambda t:t*100, y18_dt));
y18_ls = [I18_AL_LS[0],I18_AL_LS[1],I18_AL_LS[2],I18_AL_LS[3],I18_AL_LS[4]];
y18_ls = list(map(lambda t:t*100, y18_ls));

# Configurando a posição no eixo x
axis18 = np.arange(len(y18_rf))
y181 = [x + bar_width for x in axis18]
y182 = [x + bar_width for x in y181]
y183 = [x + bar_width for x in y182]

# Fazendo o plot
plt.bar(y181, y18_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y182, y18_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y183, y18_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y18_rf))], \
           x18,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Modalidade do Ensino Médio');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I18_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
#bar_width = 0.1;

x19 = ['Ninguém', 'Pais', 'Outros membros (excluindo pais)', 'Profs', 
       'Líder ou representante religioso', 'Colegas/amigos', 'Outras pessoas'];
y19_rf = [I19_AL_RF[0],I19_AL_RF[1],I19_AL_RF[2],I19_AL_RF[3], 
          I19_AL_RF[4], I19_AL_RF[5], I19_AL_RF[6]];
y19_rf = list(map(lambda t:t*100, y19_rf));
y19_dt = [I19_AL_DT[0],I19_AL_DT[1],I19_AL_DT[2],I19_AL_DT[3],
          I19_AL_DT[4], I19_AL_DT[5], I19_AL_DT[6]];
y19_dt = list(map(lambda t:t*100, y19_dt));
y19_ls = [I19_AL_LS[0],I19_AL_LS[1],I19_AL_LS[2],I19_AL_LS[3],
          I19_AL_LS[4], I19_AL_LS[5], I19_AL_LS[6]];
y19_ls = list(map(lambda t:t*100, y19_ls));

# Configurando a posição no eixo x
axis19 = np.arange(len(y19_rf))
y191 = [x + bar_width for x in axis19]
y192 = [x + bar_width for x in y191]
y193 = [x + bar_width for x in y192]

# Fazendo o plot
plt.bar(y191, y19_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y192, y19_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y193, y19_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y19_rf))], \
           x19,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Maior incentivo para cursar a graduação');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I19_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
#bar_width = 0.1;

x20 = ['Não tive dificuldade', 'Não recebi apoio', 'Pais', 'Avós', 'Irmãos, primos ou tios',
       'Líder ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do serviço de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20_rf = [I20_AL_RF[0],I20_AL_RF[1],I20_AL_RF[2],I20_AL_RF[3], I20_AL_RF[4], I20_AL_RF[5], 
          I20_AL_RF[6], I20_AL_RF[7], I20_AL_RF[8], I20_AL_RF[9], I20_AL_RF[10]];
y20_rf = list(map(lambda t:t*100, y20_rf));
y20_dt = [I20_AL_DT[0],I20_AL_DT[1],I20_AL_DT[2],I20_AL_DT[3],I20_AL_DT[4], I20_AL_DT[5],
          I20_AL_DT[6],I20_AL_DT[7], I20_AL_DT[8], I20_AL_DT[9], I20_AL_DT[10]];
y20_dt = list(map(lambda t:t*100, y20_dt));
y20_ls = [I20_AL_LS[0],I20_AL_LS[1],I20_AL_LS[2],I20_AL_LS[3],I20_AL_LS[4], I20_AL_LS[5],
          I20_AL_LS[6],I20_AL_LS[7], I20_AL_LS[8], I20_AL_LS[9], I20_AL_LS[10]];
y20_ls = list(map(lambda t:t*100, y20_ls));

# Configurando a posição no eixo x
axis20 = np.arange(len(y20_rf))
y201 = [x + bar_width for x in axis20]
y202 = [x + bar_width for x in y201]
y203 = [x + bar_width for x in y202]

# Fazendo o plot
plt.bar(y201, y20_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y202, y20_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y203, y20_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y20_rf))], \
           x20,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e concluí-lo');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I20_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
#bar_width = 0.1;

x21 = ['Sim', 'Não'];
y21_rf = [I21_AL_RF[0],I21_AL_RF[1]];
y21_rf = list(map(lambda t:t*100, y21_rf));
y21_dt = [I21_AL_DT[0],I21_AL_DT[1]];
y21_dt = list(map(lambda t:t*100, y21_dt));
y21_ls = [I21_AL_LS[0],I21_AL_LS[1]];
y21_ls = list(map(lambda t:t*100, y21_ls));

# Configurando a posição no eixo x
axis21 = np.arange(len(y21_rf))
y211 = [x + bar_width for x in axis21]
y212 = [x + bar_width for x in y211]
y213 = [x + bar_width for x in y212]

# Fazendo o plot
plt.bar(y211, y21_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y212, y21_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y213, y21_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y21_rf))], \
           x21,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Alguém da família concluiu curso superior');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I21_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
#bar_width = 0.1;

x22 = ['Nenhum  ', 'Um ou dois', 'Três a cinco', 'Seis a oito', 'Mais de oito'];
y22_rf = [I22_AL_RF[0],I22_AL_RF[1],I22_AL_RF[2],I22_AL_RF[3], I22_AL_RF[4]];
y22_rf = list(map(lambda t:t*100, y22_rf));
y22_dt = [I22_AL_DT[0],I22_AL_DT[1],I22_AL_DT[2],I22_AL_DT[3],I22_AL_DT[4]];
y22_dt = list(map(lambda t:t*100, y22_dt));
y22_ls = [I22_AL_LS[0],I22_AL_LS[1],I22_AL_LS[2],I22_AL_LS[3],I22_AL_LS[4]];
y22_ls = list(map(lambda t:t*100, y22_ls));

# Configurando a posição no eixo x
axis22 = np.arange(len(y22_rf))
y221 = [x + bar_width for x in axis22]
y222 = [x + bar_width for x in y221]
y223 = [x + bar_width for x in y222]

# Fazendo o plot
plt.bar(y221, y22_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y222, y22_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y223, y22_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y22_rf))], \
           x22,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Livros lido no ano (excluindo da biografia do curso)');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I22_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
#bar_width = 0.1;

x23 = ['Nenhuma', 'De uma a três', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23_rf = [I23_AL_RF[0],I23_AL_RF[1],I23_AL_RF[2],I23_AL_RF[3],I23_AL_RF[4]];
y23_rf = list(map(lambda t:t*100, y23_rf));
y23_dt = [I23_AL_DT[0],I23_AL_DT[1],I23_AL_DT[2],I23_AL_DT[3],I23_AL_DT[4]];
y23_dt = list(map(lambda t:t*100, y23_dt));
y23_ls = [I23_AL_LS[0],I23_AL_LS[1],I23_AL_LS[2],I23_AL_LS[3],I23_AL_LS[4]];
y23_ls = list(map(lambda t:t*100, y23_ls));

# Configurando a posição no eixo x
axis23 = np.arange(len(y23_rf))
y231 = [x + bar_width for x in axis23]
y232 = [x + bar_width for x in y231]
y233 = [x + bar_width for x in y232]

# Fazendo o plot
plt.bar(y231, y23_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y232, y23_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y233, y23_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y23_rf))], \
           x23,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I23_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
#bar_width = 0.1;

x24 = ['Sim, presencial', 'Sim, semipresencial', 
       'Sim, presencial e semipresencial', 'Sim, EAD', 'Não'];
y24_rf = [I24_AL_RF[0],I24_AL_RF[1],I24_AL_RF[2],I24_AL_RF[3], I24_AL_RF[4]];
y24_rf = list(map(lambda t:t*100, y24_rf));
y24_dt = [I24_AL_DT[0],I24_AL_DT[1],I24_AL_DT[2],I24_AL_DT[3],I24_AL_DT[4]];
y24_dt = list(map(lambda t:t*100, y24_dt));
y24_ls = [I24_AL_LS[0],I24_AL_LS[1],I24_AL_LS[2],I24_AL_LS[3],I24_AL_LS[4]];
y24_ls = list(map(lambda t:t*100, y24_ls));

# Configurando a posição no eixo x
axis24 = np.arange(len(y24_rf))
y241 = [x + bar_width for x in axis24]
y242 = [x + bar_width for x in y241]
y243 = [x + bar_width for x in y242]

# Fazendo o plot
plt.bar(y241, y24_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y242, y24_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y243, y24_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y24_rf))], \
           x24,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I24_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
#bar_width = 0.1;

x25 = ['Inserção no merc de trab', 'Influência familiar','Valorização profissional',
       'Prestígio social', 'Vocação', 'Oferecido na modalidade EAD',
       'Baixa concorrência', 'Outro motivo'];
y25_rf = [I25_AL_RF[0],I25_AL_RF[1],I25_AL_RF[2],I25_AL_RF[3], 
          I25_AL_RF[4], I25_AL_RF[5], I25_AL_RF[6], I25_AL_RF[7]];
y25_rf = list(map(lambda t:t*100, y25_rf));
y25_dt = [I25_AL_DT[0],I25_AL_DT[1],I25_AL_DT[2],I25_AL_DT[3],
          I25_AL_DT[4], I25_AL_DT[5], I25_AL_DT[6],I25_AL_DT[7]];
y25_dt = list(map(lambda t:t*100, y25_dt));
y25_ls = [I25_AL_LS[0],I25_AL_LS[1],I25_AL_LS[2],I25_AL_LS[3],
          I25_AL_LS[4], I25_AL_LS[5], I25_AL_LS[6],I25_AL_LS[7]];
y25_ls = list(map(lambda t:t*100, y25_ls));

# Configurando a posição no eixo x
axis25 = np.arange(len(y25_rf))
y251 = [x + bar_width for x in axis25]
y252 = [x + bar_width for x in y251]
y253 = [x + bar_width for x in y252]


# Fazendo o plot
plt.bar(y251, y25_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y252, y25_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y253, y25_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y25_rf))], \
           x25,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Porque escolhi o curso');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I25_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
#bar_width = 0.1;

x26 = ['Gratuidade', 'Preço da mensalidade', 'Prox à residência', 'Prox ao trab', 
       'Facilidade de acesso', 'Qualidade/reputação', 'Única op de aprovação',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26_rf = [I26_AL_RF[0],I26_AL_RF[1],I26_AL_RF[2],I26_AL_RF[3], I26_AL_RF[4], I26_AL_RF[5], 
          I26_AL_RF[6], I26_AL_RF[7], I26_AL_RF[8]];
y26_rf = list(map(lambda t:t*100, y26_rf));
y26_dt = [I26_AL_DT[0],I26_AL_DT[1],I26_AL_DT[2],I26_AL_DT[3],I26_AL_DT[4], I26_AL_DT[5],
          I26_AL_DT[6],I26_AL_DT[7], I26_AL_DT[8]];
y26_dt = list(map(lambda t:t*100, y26_dt));
y26_ls = [I26_AL_LS[0],I26_AL_LS[1],I26_AL_LS[2],I26_AL_LS[3],I26_AL_LS[4], I26_AL_LS[5],
          I26_AL_LS[6],I26_AL_LS[7], I26_AL_LS[8]];
y26_ls = list(map(lambda t:t*100, y26_ls));

# Configurando a posição no eixo x
axis26 = np.arange(len(y26_rf))
y261 = [x + bar_width for x in axis26]
y262 = [x + bar_width for x in y261]
y263 = [x + bar_width for x in y262]

# Fazendo o plot
plt.bar(y261, y26_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y262, y26_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y263, y26_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y26_rf))], \
           x26,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (AL)');
plt.title('Porque escolhi essa IES');
plt.legend();
plt.savefig('compare_methods/AL/DETAILS_VAR_AL/QE_I26_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27a
fig27a = plt.figure();
ax27aa = fig27a.add_axes([0,0,1,1]);
#bar_width = 0.1;

ax27a = ['Estado civil', 'Cor;raça', 'Nacionalidade', 'Escolarização;pai', 'Escolarização;mãe', 
         'Onde;com quem;moro', 'Qtde;moram;comigo', 'Renda;total;família', 'Situação;financeira;atual', 
         'Situação;atual;trabalho', 'Fonte;bolsa;mensalidade', 'Aux;permanência', 
         'Bolsa;acadêmica;graduação'];
y27a_rf = [np.sum(I01_AL_RF),np.sum(I02_AL_RF),np.sum(I03_AL_RF),np.sum(I04_AL_RF),
          np.sum(I05_AL_RF),np.sum(I06_AL_RF),np.sum(I07_AL_RF),np.sum(I08_AL_RF),
          np.sum(I09_AL_RF),np.sum(I10_AL_RF),np.sum(I11_AL_RF),np.sum(I12_AL_RF),
          np.sum(I13_AL_RF)];
y27a_rf = list(map(lambda t:t*100, y27a_rf));
y27a_dt = [np.sum(I01_AL_DT),np.sum(I02_AL_DT),np.sum(I03_AL_DT),np.sum(I04_AL_DT),
          np.sum(I05_AL_DT),np.sum(I06_AL_DT),np.sum(I07_AL_DT),np.sum(I08_AL_DT),
          np.sum(I09_AL_DT),np.sum(I10_AL_DT),np.sum(I11_AL_DT),np.sum(I12_AL_DT),
          np.sum(I13_AL_DT)];
y27a_dt = list(map(lambda t:t*100, y27a_dt));
y27a_ls = [np.sum(I01_AL_LS),np.sum(I02_AL_LS),np.sum(I03_AL_LS),np.sum(I04_AL_LS),
          np.sum(I05_AL_LS),np.sum(I06_AL_LS),np.sum(I07_AL_LS),np.sum(I08_AL_LS),
          np.sum(I09_AL_LS),np.sum(I10_AL_LS),np.sum(I11_AL_LS),np.sum(I12_AL_LS),
          np.sum(I13_AL_LS)];

y27a_ls = list(map(lambda t:t*100, y27a_ls));


# Configurando a posição no eixo x
axis27a = np.arange(len(y27a_rf))
y27a1 = [x + bar_width for x in axis27a]
y27a2 = [x + bar_width for x in y27a1]
y27a3 = [x + bar_width for x in y27a2]

# Fazendo o plot
plt.bar(y27a1, y27a_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y27a2, y27a_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y27a3, y27a_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27a_rf))], \
           ax27a,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)');
plt.xlabel('Variável (AL)');
plt.title('Categorias QE_I01 a QE_I13');
plt.legend();
plt.savefig('compare_methods/AL/QE_I27a_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27b
fig27b = plt.figure();
ax27ab = fig27b.add_axes([0,0,1,1]);
#bar_width = 0.1;

ax27b = ['Atividade;exterior', 'Ingresso;cota', 'UF;medio', 'Tipo;escola;medio', 'Modalidade;medio', 
         'Quem;incentivo;curso',         'Grupo;força;curso', 'Quem;família;superior', 
         'Quantos;livros;ano', 'Horas;estudo;semana', 'Oportunidade;idioma;estrang', 'Por que;curso', 
         'Por que;IES'];
y27b_rf = [np.sum(I14_AL_RF),np.sum(I15_AL_RF),np.sum(I16_AL_RF),np.sum(I17_AL_RF),
          np.sum(I18_AL_RF),np.sum(I19_AL_RF),np.sum(I19_AL_RF),np.sum(I20_AL_RF),
          np.sum(I21_AL_RF),np.sum(I22_AL_RF),np.sum(I23_AL_RF),np.sum(I24_AL_RF),
          np.sum(I13_AL_RF)];
y27b_rf = list(map(lambda t:t*100, y27b_rf));
y27b_dt =  [np.sum(I14_AL_DT),np.sum(I15_AL_DT),np.sum(I16_AL_DT),np.sum(I17_AL_DT),
          np.sum(I18_AL_DT),np.sum(I19_AL_DT),np.sum(I19_AL_DT),np.sum(I20_AL_DT),
          np.sum(I21_AL_DT),np.sum(I22_AL_DT),np.sum(I23_AL_DT),np.sum(I24_AL_DT),
          np.sum(I13_AL_DT)];
y27b_dt = list(map(lambda t:t*100, y27b_dt));
y27b_ls =  [np.sum(I14_AL_LS),np.sum(I15_AL_LS),np.sum(I16_AL_LS),np.sum(I17_AL_LS),
          np.sum(I18_AL_LS),np.sum(I19_AL_LS),np.sum(I20_AL_LS), np.sum(I21_AL_LS),
          np.sum(I22_AL_LS),np.sum(I23_AL_LS),np.sum(I24_AL_LS),np.sum(I25_AL_LS),
          np.sum(I26_AL_LS)];
y27b_ls = list(map(lambda t:t*100, y27b_ls));

# Configurando a posição no eixo x
axis27b = np.arange(len(y27b_rf))
y27b1 = [x + bar_width for x in axis27b]
y27b2 = [x + bar_width for x in y27b1]
y27b3 = [x + bar_width for x in y27b2]

# Fazendo o plot
plt.bar(y27b1, y27b_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y27b2, y27b_dt, color='green', width=bar_width, edgecolor='white', label='Decision Tree')
plt.bar(y27b3, y27b_ls, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27b_rf))], \
           ax27b,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável (BR)');
plt.title('Categorias QE_I14 a QE_I26');
plt.legend();
plt.savefig('compare_methods/AL/QE_I27b_AL_CP.png', dpi=450, bbox_inches='tight');

#%% Arquivo de registro AL DT
fields_vimp_al_dt = ['Version','Metodo', 
                     'I01_AL', 'I02_AL', 'I03_AL', 'I04_AL', 'I05_AL', 'I06_AL',
                     'I07_AL', 'I08_AL', 'I09_AL', 'I10_AL', 'I11_AL', 'I12_AL', 
                     'I13_AL', 'I14_AL', 'I15_AL', 'I16_AL', 'I17_AL', 'I18_AL', 
                     'I19_AL', 'I20_AL', 'I21_AL', 'I22_AL', 'I23_AL', 'I24_AL',
                     'I25_AL', 'I26_AL']

rows_vimp_al_dt = {'Version':0,'Metodo':'DT_AL',
                   'I01_AL':round(np.sum(I01_AL_DT),6),
                   'I02_AL':round(np.sum(I02_AL_DT),6), 'I03_AL':round(np.sum(I03_AL_DT),6),
                   'I04_AL':round(np.sum(I04_AL_DT),6), 'I05_AL':round(np.sum(I05_AL_DT),6),
                   'I06_AL':round(np.sum(I06_AL_DT),6), 'I07_AL':round(np.sum(I07_AL_DT),6),
                   'I08_AL':round(np.sum(I08_AL_DT),6), 'I09_AL':round(np.sum(I09_AL_DT),6),
                   'I10_AL':round(np.sum(I10_AL_DT),6), 'I11_AL':round(np.sum(I11_AL_DT),6),
                   'I12_AL':round(np.sum(I12_AL_DT),6), 'I13_AL':round(np.sum(I13_AL_DT),6),
                   'I14_AL':round(np.sum(I14_AL_DT),6), 'I15_AL':round(np.sum(I15_AL_DT),6),
                   'I16_AL':round(np.sum(I16_AL_DT),6), 'I17_AL':round(np.sum(I17_AL_DT),6),
                   'I18_AL':round(np.sum(I18_AL_DT),6), 'I19_AL':round(np.sum(I19_AL_DT),6),
                   'I20_AL':round(np.sum(I20_AL_DT),6), 'I21_AL':round(np.sum(I21_AL_DT),6),
                   'I22_AL':round(np.sum(I22_AL_DT),6), 'I23_AL':round(np.sum(I23_AL_DT),6),
                   'I24_AL':round(np.sum(I24_AL_DT),6), 'I25_AL':round(np.sum(I25_AL_DT),6),
                   'I26_AL': round(np.sum(I26_AL_DT),6)}

file_vimp_al_dt = "../tcc_codes/compare_methods/Logs/VIMPS/VIMP_DT_AL.csv"

version_file(file_vimp_al_dt, fields_vimp_al_dt, rows_vimp_al_dt)

#%% Arquivo de registro AL RF
fields_vimp_al_rf = ['Version','Metodo', 
                     'I01_AL', 'I02_AL', 'I03_AL', 'I04_AL', 'I05_AL', 'I06_AL',
                     'I07_AL', 'I08_AL', 'I09_AL', 'I10_AL', 'I11_AL', 'I12_AL', 
                     'I13_AL', 'I14_AL', 'I15_AL', 'I16_AL', 'I17_AL', 'I18_AL', 
                     'I19_AL', 'I20_AL', 'I21_AL', 'I22_AL', 'I23_AL', 'I24_AL',
                     'I25_AL', 'I26_AL']

rows_vimp_al_rf = {'Version':0,'Metodo':'RF_AL',
                   'I01_AL':round(np.sum(I01_AL_RF),6),
                   'I02_AL':round(np.sum(I02_AL_RF),6), 'I03_AL':round(np.sum(I03_AL_RF),6),
                   'I04_AL':round(np.sum(I04_AL_RF),6), 'I05_AL':round(np.sum(I05_AL_RF),6),
                   'I06_AL':round(np.sum(I06_AL_RF),6), 'I07_AL':round(np.sum(I07_AL_RF),6),
                   'I08_AL':round(np.sum(I08_AL_RF),6), 'I09_AL':round(np.sum(I09_AL_RF),6),
                   'I10_AL':round(np.sum(I10_AL_RF),6), 'I11_AL':round(np.sum(I11_AL_RF),6),
                   'I12_AL':round(np.sum(I12_AL_RF),6), 'I13_AL':round(np.sum(I13_AL_RF),6),
                   'I14_AL':round(np.sum(I14_AL_RF),6), 'I15_AL':round(np.sum(I15_AL_RF),6),
                   'I16_AL':round(np.sum(I16_AL_RF),6), 'I17_AL':round(np.sum(I17_AL_RF),6),
                   'I18_AL':round(np.sum(I18_AL_RF),6), 'I19_AL':round(np.sum(I19_AL_RF),6),
                   'I20_AL':round(np.sum(I20_AL_RF),6), 'I21_AL':round(np.sum(I21_AL_RF),6),
                   'I22_AL':round(np.sum(I22_AL_RF),6), 'I23_AL':round(np.sum(I23_AL_RF),6),
                   'I24_AL':round(np.sum(I24_AL_RF),6), 'I25_AL':round(np.sum(I25_AL_RF),6),
                   'I26_AL': round(np.sum(I26_AL_RF),6)}

file_vimp_al_rf = "../tcc_codes/compare_methods/Logs/VIMPS/VIMP_RF_AL.csv"

version_file(file_vimp_al_rf, fields_vimp_al_rf, rows_vimp_al_rf)
    
#%% Arquivo de registro AL LS
fields_vimp_al_ls = ['Version','Metodo', 
                     'I01_AL', 'I02_AL', 'I03_AL', 'I04_AL', 'I05_AL', 'I06_AL',
                     'I07_AL', 'I08_AL', 'I09_AL', 'I10_AL', 'I11_AL', 'I12_AL', 
                     'I13_AL', 'I14_AL', 'I15_AL', 'I16_AL', 'I17_AL', 'I18_AL', 
                     'I19_AL', 'I20_AL', 'I21_AL', 'I22_AL', 'I23_AL', 'I24_AL',
                     'I25_AL', 'I26_AL']

rows_vimp_al_ls = {'Version':0,'Metodo':'LS_AL',
                   'I01_AL':round(np.sum(I01_AL_LS),6),
                   'I02_AL':round(np.sum(I02_AL_LS),6), 'I03_AL':round(np.sum(I03_AL_LS),6),
                   'I04_AL':round(np.sum(I04_AL_LS),6), 'I05_AL':round(np.sum(I05_AL_LS),6),
                   'I06_AL':round(np.sum(I06_AL_LS),6), 'I07_AL':round(np.sum(I07_AL_LS),6),
                   'I08_AL':round(np.sum(I08_AL_LS),6), 'I09_AL':round(np.sum(I09_AL_LS),6),
                   'I10_AL':round(np.sum(I10_AL_LS),6), 'I11_AL':round(np.sum(I11_AL_LS),6),
                   'I12_AL':round(np.sum(I12_AL_LS),6), 'I13_AL':round(np.sum(I13_AL_LS),6),
                   'I14_AL':round(np.sum(I14_AL_LS),6), 'I15_AL':round(np.sum(I15_AL_LS),6),
                   'I16_AL':round(np.sum(I16_AL_LS),6), 'I17_AL':round(np.sum(I17_AL_LS),6),
                   'I18_AL':round(np.sum(I18_AL_LS),6), 'I19_AL':round(np.sum(I19_AL_LS),6),
                   'I20_AL':round(np.sum(I20_AL_LS),6), 'I21_AL':round(np.sum(I21_AL_LS),6),
                   'I22_AL':round(np.sum(I22_AL_LS),6), 'I23_AL':round(np.sum(I23_AL_LS),6),
                   'I24_AL':round(np.sum(I24_AL_LS),6), 'I25_AL':round(np.sum(I25_AL_LS),6),
                   'I26_AL': round(np.sum(I26_AL_LS),6)}

file_vimp_al_ls = "../tcc_codes/compare_methods/Logs/VIMPS/VIMP_LS_AL.csv"

version_file(file_vimp_al_ls, fields_vimp_al_ls, rows_vimp_al_ls)


