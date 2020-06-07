# -*- coding: utf-8 -*-
"""
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

#%%

del features_al['Unnamed: 0']

# Escolhendo apenas as colunas de interesse
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
#%% Observando os dados
print('O formato dos dados é: ', features_al.shape)

describe_al = features_al.describe()

print('Descrição para as colunas: ', describe_al)
print(describe_al.columns)

#%% Convertendo os labels de predição para arrays numpy
#labels_to_predict = np.array(features_al.loc[:,'NT_GER':'NT_CE_D3'])
labels_al = np.array(features_al['NT_GER'])
print('Media das labels: %.2f' %(labels_al.mean()) )
#%%
# Removendo as features de notas
features_al = features_al.drop(['NT_GER','NT_FG','NT_OBJ_FG','NT_DIS_FG',
                               'NT_FG_D1','NT_FG_D1_PT','NT_FG_D1_CT',
                               'NT_FG_D2','NT_FG_D2_PT','NT_FG_D2_CT',
                               'NT_CE','NT_OBJ_CE','NT_DIS_CE',
                               'NT_CE_D1','NT_CE_D2','NT_CE_D3'], axis = 1)
#%% Salvando e convertendo
# Salvando os nomes das colunas (features) com os dados para uso posterior
# antes de codificar
features_al_list = list(features_al.columns)


# One hot encoding - QE_I01 a QE_I26
features_al = pd.get_dummies(data=features_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
# Salvando os nomes das colunas (features) com os dados para uso posterior
# depois de codificar
features_al_list_oh = list(features_al.columns)
#%%
# Convertendo para numpy
features_al = np.array(features_al)

#%% MÉTODOS KDD
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

scores_al_rf = []
scores_al_dt = []

importance_fields_al_rf = 0.0
importance_fields_aux_al_rf = []

importance_fields_al_dt = 0.0
importance_fields_aux_al_dt = []

rf_al = RandomForestRegressor(n_estimators = 500, random_state=0)
dt_al = DecisionTreeClassifier(random_state = 0)

kf_cv_al = KFold(n_splits=11, random_state=None, shuffle=False) # n_splits: divisores de 7084 ^ memory

for train_index_al, test_index_al in kf_cv_al.split(features_al):
    print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = features_al[train_index_al]
    test_features_al = features_al[test_index_al]
    train_labels_al = labels_al[train_index_al]
    test_labels_al = labels_al[test_index_al]
    
    # Ajustando cada features e label com RF e DT
    #rf_al.fit(train_features_al, train_labels_al)
    dt_al.fit(train_features_al, train_labels_al)
    
    # Usando o RF e DT para predição dos dados
    #predictions_al_rf = rf_al.predict(test_features_al)
    predictions_al_dt = dt_al.predict(test_features_al)
    
    # Erro
    #errors_al_rf = abs(predictions_al_rf - test_labels_al)
    errors_al_dt = abs(predictions_al_dt - test_labels_al)
    
    # Acurácia
    #accuracy_al_rf = 100 - mean_absolute_error(test_labels_al, predictions_al_rf)
    accuracy_al_dt = 100 - mean_absolute_error(test_labels_al, predictions_al_dt)
    
    # Importância das variáveis
    #importance_fields_aux_al_rf = rf_al.feature_importances_
    #importance_fields_al_rf += importance_fields_aux_al_rf
    
    importance_fields_aux_al_dt = dt_al.feature_importances_
    importance_fields_al_dt += importance_fields_aux_al_dt
    
    # Append em cada valor médio
    #scores_al_rf.append(accuracy_al_rf)
    scores_al_dt.append(accuracy_al_dt)

#%% Acurácia AL
print('Accuracy RF: ', round(np.average(scores_al_rf), 2), "%.")
print('Accuracy DT: ', round(np.average(scores_al_dt), 2), "%.")

importance_fields_al_rf_t = importance_fields_al_rf/11
importance_fields_al_dt_t = importance_fields_al_dt/11

print('Total RF: ', round(np.sum(importance_fields_al_rf_t),2))
print('Total DT: ', round(np.sum(importance_fields_al_dt_t),2))

#%% Importancia das variáveis
# Lista de tupla com as variáveis de importância - Random Forest
feature_importances_al_rf = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_rf_t)]

# Print out the feature and importances
[print('Variable RF: {:20} Importance RF: {}'.format(*pair)) for pair in feature_importances_al_rf];

# Lista de tupla com as variáveis de importância - Árvore de decisão
feature_importances_al_dt = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_dt_t)]

# Print out the feature and importances
[print('Variable DT: {:20} Importance DT: {}'.format(*pair)) for pair in feature_importances_al_dt];

#%% Separando os valores
# RF
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

# DT
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

#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
bar_width = 0.1;

x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1_rf = [I01_AL_RF[0],I01_AL_RF[1],I01_AL_RF[2],I01_AL_RF[3],I01_AL_RF[4]];
y1_dt = [I01_AL_DT[0],I01_AL_DT[1],I01_AL_DT[2],I01_AL_DT[3],I01_AL_DT[4]];

# Configurando a posição no eixo x
axis1 = np.arange(len(y1_rf))
y11 = [x + bar_width for x in axis1]
y12 = [x + bar_width for x in y11]

# Fazendo o plot
plt.bar(y11, y1_rf, color='red', width=bar_width, edgecolor='white', \
        label='Random Forest', rotation=90, fontsize=8)
plt.bar(y12, y1_rf, color='green', width=bar_width, edgecolor='white', \
        label='Decision Tree', rotation=90, fontsize=8)
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y1_rf))], x1, rotation=90, fontsize=8)
plt.ylabel('Importância'); 
plt.xlabel('Variável');
plt.title('Estado civil');
plt.legend();
plt.savefig('QE_I01_AL_CP.png', dpi=450, bbox_inches='tight');

#%%
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
bar_width = 0.1
 
# set height of bar
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]
bars4 = [44.3, 44.5, 80, 150, 15]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
 
# Make the plot
plt.bar(r1, bars1, color='red', width=bar_width, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='green', width=bar_width, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='blue', width=bar_width, edgecolor='white', label='var3')
plt.bar(r4, bars4, color='yellow', width=bar_width, edgecolor='white', label='var4')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'],rotation=90, fontsize=8)
 
# Create legend & Show graphic
plt.ylabel('Importância'); 
plt.xlabel('Variável');
plt.title('QE_I14 a QE_I26');
plt.legend()
plt.show()
