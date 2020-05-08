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

# PREPRANDO OS DADOS

path_al = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/tcc_data/AL_data.csv'

features_al = pd.read_csv(path_al)

del features_al['Unnamed: 0']


# Choosing only the columns of interest
features_al = features_al.loc[:,'NT_GER':'QE_I26']
features_al = features_al.drop(features_al.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

#%% Observando os dados
print('O formato dos dados é: ', features_al.shape)

describe_al = features_al.describe()

print('Descrição para as colunas: ', describe_al)
print(describe_al.columns)

#%% Números que são strings para float
# Colunas NT_GER a NT_DIS_FG ^ NT_CE a NT_DIS_CE
features_al['NT_GER'] = features_al['NT_GER'].str.replace(',','.')
features_al['NT_GER'] = features_al['NT_GER'].astype(float)

features_al['NT_FG'] = features_al['NT_FG'].str.replace(',','.')
features_al['NT_FG'] = features_al['NT_FG'].astype(float)

features_al['NT_OBJ_FG'] = features_al['NT_OBJ_FG'].str.replace(',','.')
features_al['NT_OBJ_FG'] = features_al['NT_OBJ_FG'].astype(float)

features_al['NT_DIS_FG'] = features_al['NT_DIS_FG'].str.replace(',','.')
features_al['NT_DIS_FG'] = features_al['NT_DIS_FG'].astype(float)

features_al['NT_CE'] = features_al['NT_CE'].str.replace(',','.')
features_al['NT_CE'] = features_al['NT_CE'].astype(float)

features_al['NT_OBJ_CE'] = features_al['NT_OBJ_CE'].str.replace(',','.')
features_al['NT_OBJ_CE'] = features_al['NT_OBJ_CE'].astype(float)

features_al['NT_DIS_CE'] = features_al['NT_DIS_CE'].str.replace(',','.')
features_al['NT_DIS_CE'] = features_al['NT_DIS_CE'].astype(float)
#%% Substituindo valores nan pela mediana (medida resistente) e 0 por 1

features_al_median = features_al.iloc[:,0:16].median()

features_al.iloc[:,0:16] = features_al.iloc[:,0:16].fillna(features_al.iloc[:,0:16].median())

features_al.iloc[:,0:16] = features_al.iloc[:,0:16].replace(to_replace = 0, value = 1)
#%% Observando os dados
print('O formato dos dados é: ', features_al.shape)

describe_al = features_al.describe()

print('Descrição para as colunas: ', describe_al)
print(describe_al.columns)

#%% One hot encoding - QE_I01 a QE_I26
features_al = pd.get_dummies(data=features_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
#%% Convertendo os labels de predição para arrays numpy
#labels_to_predict = np.array(features_al.loc[:,'NT_GER':'NT_CE_D3'])
labels_to_predict = np.array(features_al['NT_GER'])
print('Media das labels: %.2f' %(labels_to_predict.mean()) )
#%%
# Removendo as features de notas
features_al = features_al.drop(['NT_GER','NT_FG','NT_OBJ_FG','NT_DIS_FG',
                               'NT_FG_D1','NT_FG_D1_PT','NT_FG_D1_CT',
                               'NT_FG_D2','NT_FG_D2_PT','NT_FG_D2_CT',
                               'NT_CE','NT_OBJ_CE','NT_DIS_CE',
                               'NT_CE_D1','NT_CE_D2','NT_CE_D3'], axis = 1)
#%% Salvando e convertendo
# Salvando os nomes das colunas (features) com os dados para uso posterior
features_al_list = list(features_al.columns)

# Convertendo para numpy
features_al = np.array(features_al)

#%% DIVIDINDO OS DADOS EM TREINO E TESTE
# Conjunto de Treino e de Teste
from sklearn.model_selection import train_test_split

train_features_al, test_features_al, train_labels_al, test_labels_al = train_test_split(features_al, 
                                                                         labels_to_predict, 
                                                                         test_size = 0.25, 
                                                                         random_state = 0)
#%% Forma dos recursos de Treino e Teste
print('Forma (recursos de treino): ', train_features_al.shape)
print('Forma (valores-alvo de treino): ', train_labels_al.shape)
print('Forma (recursos de teste):', test_features_al.shape)
print('Forma (valores-alvo de teste): ', test_labels_al.shape)


#%% TREINANDO O MODELO
from sklearn.ensemble import RandomForestRegressor

# Instanciando o modelo
rf_al = RandomForestRegressor(n_estimators = 1000, random_state = 0)

# Treinando os modelos com os dados de treinamento
rf_al.fit(train_features_al, train_labels_al)

#%% TESTANDO O MODELO
# Usando o RF nos dados
predictions_al = rf_al.predict(test_features_al)

print('Min: %.2f' %(np.min(predictions_al)))
print('Max: %.2f' %(np.max(predictions_al)))
print('Media: ', round(np.mean(predictions_al), 2))

# Calculando os erros
errors_al = abs(predictions_al - test_labels_al)

print('Erro absoluto medio: %.2f' %(np.mean(errors_al)), 'u.m.')

#%% Acurácia
from sklearn.metrics import mean_squared_error

accuracy_al = 100 - mean_squared_error(test_labels_al, predictions_al)

print('Accuracy: ', round(accuracy_al, 2), '%.')