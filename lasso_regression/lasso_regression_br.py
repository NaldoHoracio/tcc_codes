"""
Tí­tulo: Lasso Regression aplicado a dados no Brasil (excluindo Alagoas)

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

path_br = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/tcc_data/BR_data.csv'

features_br = pd.read_csv(path_br)

#%%
del features_br['Unnamed: 0']

#%%

# Escolhendo apenas as colunas de interesse
features_br = features_br.loc[:,'NT_GER':'QE_I26']
features_br = features_br.drop(features_br.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

#%% Observando os dados
print('O formato dos dados é: ', features_br.shape)

describe_br = features_br.describe()

print('Descrição para as colunas: ', describe_br)
print(describe_br.columns)

#%% Números que são strings para float
# Colunas NT_GER a NT_DIS_FG ^ NT_CE a NT_DIS_CE
features_br['NT_GER'] = features_br['NT_GER'].str.replace(',','.')
features_br['NT_GER'] = features_br['NT_GER'].astype(float)

features_br['NT_FG'] = features_br['NT_FG'].str.replace(',','.')
features_br['NT_FG'] = features_br['NT_FG'].astype(float)

features_br['NT_OBJ_FG'] = features_br['NT_OBJ_FG'].str.replace(',','.')
features_br['NT_OBJ_FG'] = features_br['NT_OBJ_FG'].astype(float)

features_br['NT_DIS_FG'] = features_br['NT_DIS_FG'].str.replace(',','.')
features_br['NT_DIS_FG'] = features_br['NT_DIS_FG'].astype(float)

# NT_CE
features_br['NT_CE'] = features_br['NT_CE'].str.replace(',','.')
features_br['NT_CE'] = features_br['NT_CE'].astype(float)

# NT_OBJ_CE
features_br['NT_OBJ_CE'] = features_br['NT_OBJ_CE'].str.replace(',','.')
features_br['NT_OBJ_CE'] = features_br['NT_OBJ_CE'].astype(float)

# NT_DIS_CE
features_br['NT_DIS_CE'] = features_br['NT_DIS_CE'].str.replace(',','.')
features_br['NT_DIS_CE'] = features_br['NT_DIS_CE'].astype(float)
#%% Substituindo valores nan pela mediana (medida resistente) e 0 por 1

features_br_median = features_br.iloc[:,0:16].median()

features_br.iloc[:,0:16] = features_br.iloc[:,0:16].fillna(features_br.iloc[:,0:16].median())

features_br.iloc[:,0:16] = features_br.iloc[:,0:16].replace(to_replace = 0, value = 1)
#%% Observando os dados
print('O formato dos dados é: ', features_br.shape)

describe_br = features_br.describe()

print('Descrição para as colunas: ', describe_br)
print(describe_br.columns)
#%% Convertendo os labels de predição para arrays numpy
#labels_to_predict = np.array(features_al.loc[:,'NT_GER':'NT_CE_D3'])
labels_br = np.array(features_br['NT_GER'])
print('Media das labels: %.2f' %(labels_br.mean()) )
#%%
# Removendo as features de notas
features_br = features_br.drop(['NT_GER','NT_FG','NT_OBJ_FG','NT_DIS_FG',
                               'NT_FG_D1','NT_FG_D1_PT','NT_FG_D1_CT',
                               'NT_FG_D2','NT_FG_D2_PT','NT_FG_D2_CT',
                               'NT_CE','NT_OBJ_CE','NT_DIS_CE',
                               'NT_CE_D1','NT_CE_D2','NT_CE_D3'], axis = 1)
#%% Salvando e convertendo
# Salvando os nomes das colunas (features) com os dados para uso posterior
# antes de codificar
features_br_list = list(features_br.columns)


# One hot encoding - QE_I01 a QE_I26
features_br = pd.get_dummies(data=features_br, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
# Salvando os nomes das colunas (features) com os dados para uso posterior
# depois de codificar
features_br_list_oh = list(features_br.columns)
#%%
# Convertendo para numpy
features_br = np.array(features_br)
#%% K-Fold CV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn import linear_model

number_splits = int(11)

scores_br = []

importance_fields_br = 0.0
importance_fields_aux_br = []

lasso_br = linear_model.Lasso(alpha=0.1, positive=True)

kf_cv_br = KFold(n_splits=number_splits, random_state=None, shuffle=False) # n_splits: divisores de 7084 ^ memory

for train_index_br, test_index_br in kf_cv_br.split(features_br):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_br), '-', np.max(test_index_br))
    
    # Dividindo nas features e labels
    train_features_br = features_br[train_index_br]
    test_features_br = features_br[test_index_br]
    train_labels_br = labels_br[train_index_br]
    test_labels_br = labels_br[test_index_br]
    
    # Ajustando cada features e label com RF
    lasso_br.fit(train_features_br, train_labels_br)
    
    # Usando o Random Forest para predição dos dados
    predictions_br = lasso_br.predict(test_features_br)
    
    # Erro
    errors_br = abs(predictions_br - test_labels_br)
    
    # Acurácia
    accuracy_br = 100 - mean_absolute_error(test_labels_br, predictions_br)
    
    # Importânncia das variáveis
    importance_fields_aux_br = lasso_br.coef_
    importance_fields_br += importance_fields_aux_br
    
    
    # Append em cada valor médio
    scores_br.append(accuracy_br)

#%% - Acurácia BR
print('Accuracy: ', round(np.average(scores_br), 2), "%.")

importance_fields_br_t = importance_fields_br/number_splits

#%% Importancia das variáveis
# List of tuples with variable and importance
feature_importances_br = [(feature, round(importance, 8)) for feature, importance in zip(features_br_list_oh, importance_fields_br_t)]

# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_br];


#%% Separando os valores
I01_BR = importance_fields_br_t[0:5]; I02_BR = importance_fields_br_t[5:11]; 

I03_BR = importance_fields_br_t[11:14]; I04_BR = importance_fields_br_t[14:20]; 

I05_BR = importance_fields_br_t[20:26]; I06_BR = importance_fields_br_t[26:32];

I07_BR = importance_fields_br_t[32:40]; I08_BR = importance_fields_br_t[40:47]; 

I09_BR = importance_fields_br_t[47:53]; I10_BR = importance_fields_br_t[53:58]; 

I11_BR = importance_fields_br_t[58:69]; I12_BR = importance_fields_br_t[69:75];

I13_BR = importance_fields_br_t[75:81]; I14_BR = importance_fields_br_t[81:87]; 

I15_BR = importance_fields_br_t[87:93]; I16_BR = importance_fields_br_t[93:120]; 

I17_BR = importance_fields_br_t[120:126]; I18_BR = importance_fields_br_t[126:131]; 

I19_BR = importance_fields_br_t[131:138]; I20_BR = importance_fields_br_t[138:149]; 

I21_BR = importance_fields_br_t[149:151]; I22_BR = importance_fields_br_t[151:156]; 

I23_BR = importance_fields_br_t[156:161]; I24_BR = importance_fields_br_t[161:166];

I25_BR = importance_fields_br_t[166:174]; I26_BR = importance_fields_br_t[174:183];
#%% Convertendo para percentual

sum_variables = np.sum(importance_fields_br_t)

I01_BR = I01_BR/sum_variables; I02_BR = I02_BR/sum_variables; 

I03_BR = I03_BR/sum_variables; I04_BR = I04_BR/sum_variables; 

I05_BR = I05_BR/sum_variables; I06_BR = I06_BR/sum_variables;

I07_BR = I07_BR/sum_variables; I08_BR = I08_BR/sum_variables; 

I09_BR = I09_BR/sum_variables; I10_BR = I10_BR/sum_variables; 

I11_BR = I11_BR/sum_variables; I12_BR = I12_BR/sum_variables;

I13_BR = I13_BR/sum_variables; I14_BR = I14_BR/sum_variables; 

I15_BR = I15_BR/sum_variables; I16_BR = I16_BR/sum_variables; 

I17_BR = I17_BR/sum_variables; I18_BR = I18_BR/sum_variables; 

I19_BR = I19_BR/sum_variables; I20_BR = I20_BR/sum_variables; 

I21_BR = I21_BR/sum_variables; I22_BR = I22_BR/sum_variables; 

I23_BR = I23_BR/sum_variables; I24_BR = I24_BR/sum_variables;

I25_BR = I25_BR/sum_variables; I26_BR = I26_BR/sum_variables;

#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1 = [I01_BR[0],I01_BR[1],I01_BR[2],I01_BR[3],I01_BR[4]];
y1 = list(map(lambda t:t*100, y1))

# Configurando a posição no eixo x
axis1 = np.arange(len(y1))
y11 = [x + bar_width for x in axis1]

# Fazendo o plot
plt.bar(y11, y1, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y1))], \
           ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'],\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)');
plt.xlabel('Variável');
plt.title('Estado civil');
plt.legend();
plt.savefig('QE_I01_LS.png', dpi=450, bbox_inches='tight');

#%% VisuBRization of Variable Importances
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
bar_width = 0.3;

x2 = ['Branca','Preta','Amarela','Parda','Indígena','Não quero declarar'];
y2 = [I02_BR[0],I02_BR[1],I02_BR[2],I02_BR[3],I02_BR[4],I02_BR[5]];
y2 = list(map(lambda t:t*100, y2))

# Configurando a posição no eixo x
axis2 = np.arange(len(y2))
y21 = [x + bar_width for x in axis2]

# Fazendo o plot
plt.bar(y21, y2, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y2))], \
           x2,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Cor/raça');
plt.legend();
plt.savefig('QE_I02_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
bar_width = 0.3;

x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3_rf = [I03_BR_RF[0],I03_BR_RF[1],I03_BR_RF[2]];
y3_rf = list(map(lambda t:t*100, y3_rf))
y3_dt = [I03_BR_DT[0],I03_BR_DT[1],I03_BR_DT[2]];
y3_dt = list(map(lambda t:t*100, y3_dt))

# Configurando a posição no eixo x
axis3 = np.arange(len(y3_rf))
y31 = [x + bar_width for x in axis3]
y32 = [x + bar_width for x in y31]

# Fazendo o plot
plt.bar(y31, y3_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y32, y3_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y3_rf))], \
           x3,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Nacionalidade');
plt.legend();
plt.savefig('QE_I03_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
bar_width = 0.3;

x4 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y4_rf = [I04_BR_RF[0],I04_BR_RF[1],I04_BR_RF[2],I04_BR_RF[3],I04_BR_RF[4],I04_BR_RF[5]];
y4_rf = list(map(lambda t:t*100, y4_rf))
y4_dt = [I04_BR_DT[0],I04_BR_DT[1],I04_BR_DT[2],I04_BR_DT[3],I04_BR_DT[4],I04_BR_DT[5]];
y4_dt = list(map(lambda t:t*100, y4_dt));

# Configurando a posição no eixo x
axis4 = np.arange(len(y4_rf))
y41 = [x + bar_width for x in axis4]
y42 = [x + bar_width for x in y41]

# Fazendo o plot
plt.bar(y41, y4_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y42, y4_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y4_rf))], \
           x4,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Escolarização da pai');
plt.legend();
plt.savefig('QE_I04_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
bar_width = 0.3;

x5 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y5_rf = [I05_BR_RF[0],I05_BR_RF[1],I05_BR_RF[2],I05_BR_RF[3],I05_BR_RF[4],I05_BR_RF[5]];
y5_rf = list(map(lambda t:t*100, y5_rf))
y5_dt = [I05_BR_DT[0],I05_BR_DT[1],I05_BR_DT[2],I05_BR_DT[3],I05_BR_DT[4],I05_BR_DT[5]];
y5_dt = list(map(lambda t:t*100, y5_dt));

# Configurando a posição no eixo x
axis5 = np.arange(len(y5_rf))
y51 = [x + bar_width for x in axis5]
y52 = [x + bar_width for x in y51]

# Fazendo o plot
plt.bar(y51, y5_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y52, y5_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y5_rf))], \
           x5,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Escolarização da mãe');
plt.legend();
plt.savefig('QE_I05_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
bar_width = 0.3;

x6 = ['Casa/apartamento (sozinho)','Casa/apartamento (pais/parentes)',
      'Casa/apartamento (cônjugue/filhos)','Casa/apartamento (outras pessoas)',
      'Alojamento univ. na própria IES','Outro'];
y6_rf = [I06_BR_RF[0],I06_BR_RF[1],I06_BR_RF[2],I06_BR_RF[3],I06_BR_RF[4],I06_BR_RF[5]];
y6_rf = list(map(lambda t:t*100, y6_rf));
y6_dt = [I06_BR_DT[0],I06_BR_DT[1],I06_BR_DT[2],I06_BR_DT[3],I06_BR_DT[4],I06_BR_DT[5]];
y6_dt = list(map(lambda t:t*100, y6_dt));

# Configurando a posição no eixo x
axis6 = np.arange(len(y6_rf))
y61 = [x + bar_width for x in axis6]
y62 = [x + bar_width for x in y61]

# Fazendo o plot
plt.bar(y61, y6_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y62, y6_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y6_rf))], \
           x6,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Onde e com quem moro');
plt.legend();
plt.savefig('QE_I06_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
bar_width = 0.3;

x7 = ['Nenhuma','Uma','Duas','Três','Quatro','Cinco','Seis','Sete ou mais'];
y7_rf = [I07_BR_RF[0],I07_BR_RF[1],I07_BR_RF[2],I07_BR_RF[3],
         I07_BR_RF[4],I07_BR_RF[5],I07_BR_RF[6],I07_BR_RF[7]];
y7_rf = list(map(lambda t:t*100, y7_rf));
y7_dt = [I07_BR_DT[0],I07_BR_DT[1],I07_BR_DT[2],I07_BR_DT[3],
         I07_BR_DT[4],I07_BR_DT[5],I07_BR_DT[6],I07_BR_DT[7]];
y7_dt = list(map(lambda t:t*100, y7_dt));

# Configurando a posição no eixo x
axis7 = np.arange(len(y7_rf))
y71 = [x + bar_width for x in axis7]
y72 = [x + bar_width for x in y71]

# Fazendo o plot
plt.bar(y71, y7_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y72, y7_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y7_rf))], \
           x7,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Quantos moram com o estudante');
plt.legend();
plt.savefig('QE_I07_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
bar_width = 0.3;

x8 = ['Até 1,5 sál. mín','De 1 a 3 sál. mín.','De 3 a 4,5 sál. mín.',
      'De 4,5 a 6 sál. mín','De 6 a 10 sál. mín.','De 30 a 10 sál. mín',
      'Acima de 30 sál. mín.'];
y8_rf = [I08_BR_RF[0],I08_BR_RF[1],I08_BR_RF[2],I08_BR_RF[3],
         I08_BR_RF[4],I08_BR_RF[5],I08_BR_RF[6]];
y8_rf = list(map(lambda t:t*100, y8_rf));
y8_dt = [I08_BR_DT[0],I08_BR_DT[1],I08_BR_DT[2],I08_BR_DT[3],
         I08_BR_DT[4],I08_BR_DT[5],I08_BR_DT[6]];
y8_dt = list(map(lambda t:t*100, y8_dt));

# Configurando a posição no eixo x
axis8 = np.arange(len(y8_rf))
y81 = [x + bar_width for x in axis8]
y82 = [x + bar_width for x in y81]

# Fazendo o plot
plt.bar(y81, y8_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y82, y8_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y8_rf))], \
           x8,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Renda total');
plt.legend();
plt.savefig('QE_I08_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I09
fig9 = plt.figure();
ax9 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x9 = ['Sem renda (financiamento governamental)','Sem renda (financ. por família/outros)',
      'Tenho renda, mas recebo ajuda (família/outras pessoas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a família',
      'Sou o principal a ajudar a família'];
y9_rf = [I09_BR_RF[0],I09_BR_RF[1],I09_BR_RF[2],I09_BR_RF[3],
         I09_BR_RF[4],I09_BR_RF[5]];
y9_rf = list(map(lambda t:t*100, y9_rf));
y9_dt = [I09_BR_DT[0],I07_BR_DT[1],I09_BR_DT[2],I07_BR_DT[3],
         I09_BR_DT[4],I09_BR_DT[5]];
y9_dt = list(map(lambda t:t*100, y9_dt));

# Configurando a posição no eixo x
axis9 = np.arange(len(y9_rf))
y91 = [x + bar_width for x in axis9]
y92 = [x + bar_width for x in y91]

# Fazendo o plot
plt.bar(y91, y9_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y92, y9_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y9_rf))], \
           x9,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Situação financeira');
plt.legend();
plt.savefig('QE_I09_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
bar_width = 0.3;

x10 = ['Não estou trabalhando','Trabalho eventualmente','Trablho (até 20h/sem)',
       'Trabalho (de 21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10_rf = [I10_BR_RF[0],I10_BR_RF[1],I10_BR_RF[2],I10_BR_RF[3],
         I10_BR_RF[4]];
y10_rf = list(map(lambda t:t*100, y10_rf));
y10_dt = [I10_BR_DT[0],I10_BR_DT[1],I10_BR_DT[2],I10_BR_DT[3],
         I10_BR_DT[4]];
y10_dt = list(map(lambda t:t*100, y10_dt));

# Configurando a posição no eixo x
axis10 = np.arange(len(y10_rf))
y101 = [x + bar_width for x in axis10]
y102 = [x + bar_width for x in y101]

# Fazendo o plot
plt.bar(y101, y10_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y102, y10_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y10_rf))], \
           x10,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Situação de trabalho');
plt.legend();
plt.savefig('QE_I10_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I11
fig11 = plt.figure();
ax11 = fig11.add_axes([0,0,1,1]);
bar_width = 0.3;

x11 = ['Nenhum (curso gratuito)','Nenhum (mas não gratuito)','ProUni integral',
       'ProUni parcial, apenas','FIES, apenas','ProUni parcial e FIES',
       'Bolsa do governo (estadual/distrital/municipal)',
       'Bolsa pela IES','Bolsa por outra entidade','Financiamento pela IES',
       'Financiamento bancário'];
y11_rf = [I11_BR_RF[0],I11_BR_RF[1],I11_BR_RF[2],I11_BR_RF[3], I11_BR_RF[4],
          I11_BR_RF[5],I11_BR_RF[6],I11_BR_RF[7],I11_BR_RF[8], I11_BR_RF[9], I11_BR_RF[10]];
y11_rf = list(map(lambda t:t*100, y11_rf));
y11_dt = [I11_BR_DT[0],I11_BR_DT[1],I11_BR_DT[2],I11_BR_DT[3],I11_BR_DT[4],
          I11_BR_DT[5],I11_BR_DT[6],I11_BR_DT[7],I11_BR_DT[8], I11_BR_DT[9], I11_BR_DT[10]];
y11_dt = list(map(lambda t:t*100, y11_dt));

# Configurando a posição no eixo x
axis11 = np.arange(len(y11_rf))
y111 = [x + bar_width for x in axis11]
y112 = [x + bar_width for x in y111]

# Fazendo o plot
plt.bar(y111, y11_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y112, y11_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y11_rf))], \
           x11,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.legend();
plt.savefig('QE_I11_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
bar_width = 0.3;

x12 = ['Nenhum','Moradia','Alimentação','Moradia e alimentação', 'Permanência','Outros'];
y12_rf = [I12_BR_RF[0],I12_BR_RF[1],I12_BR_RF[2],I12_BR_RF[3], I12_BR_RF[4],
          I12_BR_RF[5]];
y12_rf = list(map(lambda t:t*100, y12_rf));
y12_dt = [I12_BR_DT[0],I12_BR_DT[1],I12_BR_DT[2],I12_BR_DT[3],I12_BR_DT[4],
          I12_BR_DT[5]];
y12_dt = list(map(lambda t:t*100, y12_dt));

# Configurando a posição no eixo x
axis12 = np.arange(len(y12_rf))
y121 = [x + bar_width for x in axis12]
y122 = [x + bar_width for x in y121]

# Fazendo o plot
plt.bar(y121, y12_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y122, y12_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y12_rf))], \
           x12,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Auxílio permanência');
plt.legend();
plt.savefig('QE_I12_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
bar_width = 0.3;

x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extensão','Bolsa monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13_rf = [I13_BR_RF[0],I13_BR_RF[1],I13_BR_RF[2],I13_BR_RF[3], I13_BR_RF[4],
          I13_BR_RF[5]];
y13_rf = list(map(lambda t:t*100, y13_rf));
y13_dt = [I13_BR_DT[0],I13_BR_DT[1],I13_BR_DT[2],I13_BR_DT[3],I13_BR_DT[4],
          I13_BR_DT[5]];
y13_dt = list(map(lambda t:t*100, y13_dt));

# Configurando a posição no eixo x
axis13 = np.arange(len(y13_rf))
y131 = [x + bar_width for x in axis13]
y132 = [x + bar_width for x in y131]

# Fazendo o plot
plt.bar(y131, y13_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y132, y13_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y13_rf))], \
           x13,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Tipo de bolsa recebido');
plt.legend();
plt.savefig('QE_I13_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
bar_width = 0.3;

x14 = ['Não','Sim, Ciências sem Fronteiras', 'Sim, intercâmbio pelo Governo Federal',
       'Sim, intercâmbio pelo Governo Estadual', 'Sim, intercâmbio pela minha IES',
       'Sim, intercâmbio não institucional'];
y14_rf = [I14_BR_RF[0],I14_BR_RF[1],I14_BR_RF[2],I14_BR_RF[3], I14_BR_RF[4],
          I14_BR_RF[5]];
y14_rf = list(map(lambda t:t*100, y14_rf));
y14_dt = [I14_BR_DT[0],I14_BR_DT[1],I14_BR_DT[2],I14_BR_DT[3],I14_BR_DT[4],
          I14_BR_DT[5]];
y14_dt = list(map(lambda t:t*100, y14_dt));

# Configurando a posição no eixo x
axis14 = np.arange(len(y14_rf))
y141 = [x + bar_width for x in axis14]
y142 = [x + bar_width for x in y141]

# Fazendo o plot
plt.bar(y141, y14_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y142, y14_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y14_rf))], \
           x14,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Programas de atividade no exterior');
plt.legend();
plt.savefig('QE_I14_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1]);
bar_width = 0.3;

x15 = ['Não','Sim, étnico-racial','Sim, renda', 'Sim, escola pública ou particular (com bolsa)',
       'Sim, combina dois mais', 'Sim, outra'];
y15_rf = [I15_BR_RF[0],I15_BR_RF[1],I15_BR_RF[2],I15_BR_RF[3], I15_BR_RF[4],
          I15_BR_RF[5]];
y15_rf = list(map(lambda t:t*100, y15_rf));
y15_dt = [I15_BR_DT[0],I15_BR_DT[1],I15_BR_DT[2],I15_BR_DT[3],I15_BR_DT[4],
          I15_BR_DT[5]];
y15_dt = list(map(lambda t:t*100, y15_dt));

# Configurando a posição no eixo x
axis15 = np.arange(len(y15_rf))
y151 = [x + bar_width for x in axis15]
y152 = [x + bar_width for x in y151]

# Fazendo o plot
plt.bar(y151, y15_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y152, y15_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y15_rf))], \
           x15,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Ingresso por cota');
plt.legend();
plt.savefig('QE_I15_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
bar_width = 0.3;

x16 = ['RO','AC','AM','RR','PA','AP','TO','MA','PI','CE','RN','PB','PE',
       'SE','BA','MG','ES','RJ','SP','PR','SC','RS','MS','MT','GO','DF','Outro'];
y16_rf = [I16_BR_RF[0],I16_BR_RF[1],I16_BR_RF[2],I16_BR_RF[3],I16_BR_RF[4],I16_BR_RF[5],
          I16_BR_RF[6],I16_BR_RF[7],I16_BR_RF[8],I16_BR_RF[9],I16_BR_RF[10],I16_BR_RF[11],
          I16_BR_RF[12],I16_BR_RF[13],I16_BR_RF[14],I16_BR_RF[15],I16_BR_RF[16],I16_BR_RF[17],
          I16_BR_RF[18],I16_BR_RF[19],I16_BR_RF[20],I16_BR_RF[21],I16_BR_RF[22],I16_BR_RF[23],
          I16_BR_RF[24],I16_BR_RF[25],I16_BR_RF[26]];
y16_rf = list(map(lambda t:t*100, y16_rf));
y16_dt = [I16_BR_DT[0],I16_BR_DT[1],I16_BR_DT[2],I16_BR_DT[3],I16_BR_DT[4],I16_BR_DT[5],
          I16_BR_DT[6],I16_BR_DT[7],I16_BR_DT[8],I16_BR_DT[9],I16_BR_DT[10],I16_BR_DT[11],
          I16_BR_DT[12],I16_BR_DT[13],I16_BR_DT[14],I16_BR_DT[15],I16_BR_DT[16],I16_BR_DT[17],
          I16_BR_DT[18],I16_BR_DT[19],I16_BR_DT[20],I16_BR_DT[21],I16_BR_DT[22],I16_BR_DT[23],
          I16_BR_DT[24],I16_BR_DT[25],I16_BR_DT[26]];
y16_dt = list(map(lambda t:t*100, y16_dt));

# Configurando a posição no eixo x
axis16 = np.arange(len(y16_rf))
y161 = [x + bar_width for x in axis16]
y162 = [x + bar_width for x in y161]

# Fazendo o plot
plt.bar(y161, y16_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y162, y16_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y16_rf))], \
           x16,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('UF que concluiu o médio');
plt.legend();
plt.savefig('QE_I16_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
#bar_width = 0.1;

x17 = ['Todo em escola pública', 'Todo em escola privada','Todo no exterior',
       'Maior parte em escola pública','Maior parte em escola privada',
       'Parte no Brasil e parte no exterior'];
y17_rf = [I17_BR_RF[0],I17_BR_RF[1],I17_BR_RF[2],I17_BR_RF[3], I17_BR_RF[4],
          I17_BR_RF[5]];
y17_rf = list(map(lambda t:t*100, y17_rf));
y17_dt = [I17_BR_DT[0],I17_BR_DT[1],I17_BR_DT[2],I17_BR_DT[3],I17_BR_DT[4],
          I17_BR_DT[5]];
y17_dt = list(map(lambda t:t*100, y17_dt));

# Configurando a posição no eixo x
axis17 = np.arange(len(y17_rf))
y171 = [x + bar_width for x in axis17]
y172 = [x + bar_width for x in y171]

# Fazendo o plot
plt.bar(y171, y17_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y172, y17_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y17_rf))], \
           x17,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Tipo de escola no médio');
plt.legend();
plt.savefig('QE_I17_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
#bar_width = 0.1;

x18 = ['Tradicional', 'Prof. técnico', 'Prof. magistério (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18_rf = [I18_BR_RF[0],I18_BR_RF[1],I18_BR_RF[2],I18_BR_RF[3], I18_BR_RF[4]];
y18_rf = list(map(lambda t:t*100, y18_rf));
y18_dt = [I18_BR_DT[0],I18_BR_DT[1],I18_BR_DT[2],I18_BR_DT[3],I18_BR_DT[4]];
y18_dt = list(map(lambda t:t*100, y18_dt));

# Configurando a posição no eixo x
axis18 = np.arange(len(y18_rf))
y181 = [x + bar_width for x in axis18]
y182 = [x + bar_width for x in y181]

# Fazendo o plot
plt.bar(y181, y18_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y182, y18_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y18_rf))], \
           x18,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Modalidade do Ensino Médio');
plt.legend();
plt.savefig('QE_I18_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
#bar_width = 0.1;

x19 = ['Ninguém', 'Pais', 'Outros membros (excluindo os pais)', 'Professores', 
       'Líder ou representante religioso', 'Colegas/amigos', 'Outras pessoas'];
y19_rf = [I19_BR_RF[0],I19_BR_RF[1],I19_BR_RF[2],I19_BR_RF[3], 
          I19_BR_RF[4], I19_BR_RF[5], I19_BR_RF[6]];
y19_rf = list(map(lambda t:t*100, y19_rf));
y19_dt = [I19_BR_DT[0],I19_BR_DT[1],I19_BR_DT[2],I19_BR_DT[3],
          I19_BR_DT[4], I19_BR_DT[5], I19_BR_DT[6]];
y19_dt = list(map(lambda t:t*100, y19_dt));

# Configurando a posição no eixo x
axis19 = np.arange(len(y19_rf))
y191 = [x + bar_width for x in axis19]
y192 = [x + bar_width for x in y191]

# Fazendo o plot
plt.bar(y191, y19_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y192, y19_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y19_rf))], \
           x19,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Maior incentivo para cursar a graduação');
plt.legend();
plt.savefig('QE_I19_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
#bar_width = 0.1;

x20 = ['Não tive dificuldade', 'Não recebi apoio', 'Pais', 'Avós', 'Irmãos, primos ou tios',
       'Líder ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do serviço de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20_rf = [I20_BR_RF[0],I20_BR_RF[1],I20_BR_RF[2],I20_BR_RF[3], I20_BR_RF[4], I20_BR_RF[5], 
          I20_BR_RF[6], I20_BR_RF[7], I20_BR_RF[8], I20_BR_RF[9], I20_BR_RF[10]];
y20_rf = list(map(lambda t:t*100, y20_rf));
y20_dt = [I20_BR_DT[0],I20_BR_DT[1],I20_BR_DT[2],I20_BR_DT[3],I20_BR_DT[4], I20_BR_DT[5],
          I20_BR_DT[6],I20_BR_DT[7], I20_BR_DT[8], I20_BR_DT[9], I20_BR_DT[10]];
y20_dt = list(map(lambda t:t*100, y20_dt));

# Configurando a posição no eixo x
axis20 = np.arange(len(y20_rf))
y201 = [x + bar_width for x in axis20]
y202 = [x + bar_width for x in y201]

# Fazendo o plot
plt.bar(y201, y20_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y202, y20_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y20_rf))], \
           x20,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e concluí-lo');
plt.legend();
plt.savefig('QE_I20_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
#bar_width = 0.1;

x21 = ['Sim', 'Não'];
y21_rf = [I21_BR_RF[0],I21_BR_RF[1]];
y21_rf = list(map(lambda t:t*100, y21_rf));
y21_dt = [I21_BR_DT[0],I21_BR_DT[1]];
y21_dt = list(map(lambda t:t*100, y21_dt));

# Configurando a posição no eixo x
axis21 = np.arange(len(y21_rf))
y211 = [x + bar_width for x in axis21]
y212 = [x + bar_width for x in y211]

# Fazendo o plot
plt.bar(y211, y21_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y212, y21_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y21_rf))], \
           x21,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Alguém da família concluiu curso superior');
plt.legend();
plt.savefig('QE_I21_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
#bar_width = 0.1;

x22 = ['Nenhum  ', 'Um ou dois', 'Três a cinco', 'Seis a oito', 'Mais de oito'];
y22_rf = [I22_BR_RF[0],I22_BR_RF[1],I22_BR_RF[2],I22_BR_RF[3], I22_BR_RF[4]];
y22_rf = list(map(lambda t:t*100, y22_rf));
y22_dt = [I22_BR_DT[0],I22_BR_DT[1],I22_BR_DT[2],I22_BR_DT[3],I22_BR_DT[4]];
y22_dt = list(map(lambda t:t*100, y22_dt));
# Configurando a posição no eixo x
axis22 = np.arange(len(y22_rf))
y221 = [x + bar_width for x in axis22]
y222 = [x + bar_width for x in y221]

# Fazendo o plot
plt.bar(y221, y22_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y222, y22_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y22_rf))], \
           x22,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Livros lido no ano (excluindo da Biografia do curso)');
plt.legend();
plt.savefig('QE_I22_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
#bar_width = 0.1;

x23 = ['Nenhuma', 'De uma a três', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23_rf = [I23_BR_RF[0],I23_BR_RF[1],I23_BR_RF[2],I23_BR_RF[3],I23_BR_RF[4]];
y23_rf = list(map(lambda t:t*100, y23_rf));
y23_dt = [I23_BR_DT[0],I23_BR_DT[1],I23_BR_DT[2],I23_BR_DT[3],I23_BR_DT[4]];
y23_dt = list(map(lambda t:t*100, y23_dt));

# Configurando a posição no eixo x
axis23 = np.arange(len(y23_rf))
y231 = [x + bar_width for x in axis23]
y232 = [x + bar_width for x in y231]

# Fazendo o plot
plt.bar(y231, y23_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y232, y23_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y23_rf))], \
           x23,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.legend();
plt.savefig('QE_I23_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
#bar_width = 0.1;

x24 = ['Sim, apenas presencial', 'Sim, apenas semipresencial', 
       'Sim, parte presencial e parte semipresencial', 'Sim, EAD', 'Não'];
y24_rf = [I24_BR_RF[0],I24_BR_RF[1],I24_BR_RF[2],I24_BR_RF[3], I24_BR_RF[4]];
y24_rf = list(map(lambda t:t*100, y24_rf));
y24_dt = [I24_BR_DT[0],I24_BR_DT[1],I24_BR_DT[2],I24_BR_DT[3],I24_BR_DT[4]];
y24_dt = list(map(lambda t:t*100, y24_dt));

# Configurando a posição no eixo x
axis24 = np.arange(len(y24_rf))
y241 = [x + bar_width for x in axis24]
y242 = [x + bar_width for x in y241]

# Fazendo o plot
plt.bar(y241, y24_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y242, y24_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y24_rf))], \
           x24,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.legend();
plt.savefig('QE_I24_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
#bar_width = 0.1;

x25 = ['Inserção no mercado de trabalho', 'Influência familiar','Valorização profissional',
       'Prestígio social', 'Vocação', 'Oferecido na modalidade EAD',
       'Baixa concorrência', 'Outro motivo'];
y25_rf = [I25_BR_RF[0],I25_BR_RF[1],I25_BR_RF[2],I25_BR_RF[3], 
          I25_BR_RF[4], I25_BR_RF[5], I25_BR_RF[6], I25_BR_RF[7]];
y25_rf = list(map(lambda t:t*100, y25_rf));
y25_dt = [I25_BR_DT[0],I25_BR_DT[1],I25_BR_DT[2],I25_BR_DT[3],
          I25_BR_DT[4], I25_BR_DT[5], I25_BR_DT[6],I25_BR_DT[7]];
y25_dt = list(map(lambda t:t*100, y25_dt));

# Configurando a posição no eixo x
axis25 = np.arange(len(y25_rf))
y251 = [x + bar_width for x in axis25]
y252 = [x + bar_width for x in y251]

# Fazendo o plot
plt.bar(y251, y25_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y252, y25_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y25_rf))], \
           x25,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Porque escolhi o curso');
plt.legend();
plt.savefig('QE_I25_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
#bar_width = 0.1;

x26 = ['Gratuidade', 'Preço da mensalidade', 'Prox. a residência', 'Prox. ao trabalho', 
       'Facilidade de acesso', 'Qualidade/reputação', 'Única opção de aprovação',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26_rf = [I26_BR_RF[0],I26_BR_RF[1],I26_BR_RF[2],I26_BR_RF[3], I26_BR_RF[4], I26_BR_RF[5], 
          I26_BR_RF[6], I26_BR_RF[7], I26_BR_RF[8]];
y26_rf = list(map(lambda t:t*100, y26_rf));
y26_dt = [I26_BR_DT[0],I26_BR_DT[1],I26_BR_DT[2],I26_BR_DT[3],I26_BR_DT[4], I26_BR_DT[5],
          I26_BR_DT[6],I26_BR_DT[7], I26_BR_DT[8]];
y26_dt = list(map(lambda t:t*100, y26_dt));

# Configurando a posição no eixo x
axis26 = np.arange(len(y26_rf))
y261 = [x + bar_width for x in axis26]
y262 = [x + bar_width for x in y261]

# Fazendo o plot
plt.bar(y261, y26_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y262, y26_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y26_rf))], \
           x26,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Porque escolhi essa IES');
plt.legend();
plt.savefig('QE_I26_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27a
fig27a = plt.figure();
ax27aa = fig27a.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27a = ['QE_I01_BR', 'QE_I02_BR', 'QE_I03_BR', 'QE_I04_BR', 'QE_I05_BR', 'QE_I06_BR',
         'QE_I07_BR', 'QE_I08_BR', 'QE_I09_BR', 'QE_I10_BR', 'QE_I11_BR', 'QE_I12_BR', 
         'QE_I13_BR'];
y27a_rf = [np.sum(I01_BR_RF),np.sum(I02_BR_RF),np.sum(I03_BR_RF),np.sum(I04_BR_RF),
          np.sum(I05_BR_RF),np.sum(I06_BR_RF),np.sum(I07_BR_RF),np.sum(I08_BR_RF),
          np.sum(I09_BR_RF),np.sum(I10_BR_RF),np.sum(I11_BR_RF),np.sum(I12_BR_RF),
          np.sum(I13_BR_RF)];
y27a_rf = list(map(lambda t:t*100, y27a_rf));
y27a_dt = [np.sum(I01_BR_DT),np.sum(I02_BR_DT),np.sum(I03_BR_DT),np.sum(I04_BR_DT),
          np.sum(I05_BR_DT),np.sum(I06_BR_DT),np.sum(I07_BR_DT),np.sum(I08_BR_DT),
          np.sum(I09_BR_DT),np.sum(I10_BR_DT),np.sum(I11_BR_DT),np.sum(I12_BR_DT),
          np.sum(I13_BR_DT)];
y27a_dt = list(map(lambda t:t*100, y27a_dt));

# Configurando a posição no eixo x
axis27a = np.arange(len(y27a_rf))
y27a1 = [x + bar_width for x in axis27a]
y27a2 = [x + bar_width for x in y27a1]

# Fazendo o plot
plt.bar(y27a1, y27a_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y27a2, y27a_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27a_rf))], \
           ax27a,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)');
plt.xlabel('Variável');
plt.title('QE_I01 a QE_I13');
plt.legend();
plt.savefig('QE_I27a_BR_CP.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27b
fig27b = plt.figure();
ax27ab = fig27b.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27b = ['QE_I14_BR', 'QE_I15_BR', 'QE_I16_BR', 'QE_I17_BR', 'QE_I18_BR', 'QE_I19_BR',
         'QE_I20_BR', 'QE_I21_BR', 'QE_I22_BR', 'QE_I23_BR', 'QE_I24_BR', 'QE_I25_BR', 
         'QE_I26_BR'];
y27b_rf = [np.sum(I14_BR_RF),np.sum(I15_BR_RF),np.sum(I16_BR_RF),np.sum(I17_BR_RF),
          np.sum(I18_BR_RF),np.sum(I19_BR_RF),np.sum(I20_BR_RF), np.sum(I21_BR_RF),
          np.sum(I22_BR_RF),np.sum(I23_BR_RF),np.sum(I24_BR_RF),
          np.sum(I25_BR_RF),np.sum(I26_BR_RF)];
y27b_rf = list(map(lambda t:t*100, y27b_rf));
y27b_dt =  [np.sum(I14_BR_DT),np.sum(I15_BR_DT),np.sum(I16_BR_DT),np.sum(I17_BR_DT),
          np.sum(I18_BR_DT),np.sum(I19_BR_DT),np.sum(I20_BR_DT), np.sum(I21_BR_DT),
          np.sum(I22_BR_DT),np.sum(I23_BR_DT),np.sum(I24_BR_DT),np.sum(I25_BR_DT),
          np.sum(I26_BR_DT)];
y27b_dt = list(map(lambda t:t*100, y27b_dt));

# Configurando a posição no eixo x
axis27b = np.arange(len(y27b_rf))
y27b1 = [x + bar_width for x in axis27b]
y27b2 = [x + bar_width for x in y27b1]

# Fazendo o plot
plt.bar(y27b1, y27b_rf, color='red', width=bar_width, edgecolor='white', label='Random Forest')
plt.bar(y27b2, y27b_dt, color='blue', width=bar_width, edgecolor='white', label='Decision Tree')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27b_rf))], \
           ax27b,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('QE_I14 a QE_I26');
plt.legend();
plt.savefig('QE_I27b_BR_CP.png', dpi=450, bbox_inches='tight');