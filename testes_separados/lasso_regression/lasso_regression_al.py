"""
Tí­tulo: Lasso Regression aplicado a dados em Alagoas

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

#%%

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

# NT_CE
features_al['NT_CE'] = features_al['NT_CE'].str.replace(',','.')
features_al['NT_CE'] = features_al['NT_CE'].astype(float)

# NT_OBJ_CE
features_al['NT_OBJ_CE'] = features_al['NT_OBJ_CE'].str.replace(',','.')
features_al['NT_OBJ_CE'] = features_al['NT_OBJ_CE'].astype(float)

# NT_DIS_CE
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
#%% K-Fold CV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn import linear_model

number_splits = int(11)

scores_al = []

importance_fields_al = 0.0
importance_fields_aux_al = []

lasso_al = linear_model.Lasso(alpha=0.1, positive=True)

kf_cv_al = KFold(n_splits=number_splits, random_state=None, shuffle=False) # n_splits: divisores de 7084 ^ memory

for train_index_al, test_index_al in kf_cv_al.split(features_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = features_al[train_index_al]
    test_features_al = features_al[test_index_al]
    train_labels_al = labels_al[train_index_al]
    test_labels_al = labels_al[test_index_al]
    
    # Ajustando cada features e label com RF
    lasso_al.fit(train_features_al, train_labels_al)
    
    # Usando o Random Forest para predição dos dados
    predictions_al = lasso_al.predict(test_features_al)
    
    # Erro
    errors_al = abs(predictions_al - test_labels_al)
    
    # Acurácia
    accuracy_al = 100 - mean_absolute_error(test_labels_al, predictions_al)
    
    # Importânncia das variáveis
    importance_fields_aux_al = lasso_al.coef_
    importance_fields_al += importance_fields_aux_al
    
    
    # Append em cada valor médio
    scores_al.append(accuracy_al)

#%% - Acurácia AL
print('Accuracy: ', round(np.average(scores_al), 2), "%.")

importance_fields_al_t = importance_fields_al/number_splits

#%% Importancia das variáveis
# List of tuples with variable and importance
feature_importances_al = [(feature, round(importance, 8)) for feature, importance in zip(features_al_list_oh, importance_fields_al_t)]

# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_al];


#%% Separando os valores
I01_AL = importance_fields_al_t[0:5]; I02_AL = importance_fields_al_t[5:11]; 

I03_AL = importance_fields_al_t[11:14]; I04_AL = importance_fields_al_t[14:20]; 

I05_AL = importance_fields_al_t[20:26]; I06_AL = importance_fields_al_t[26:32];

I07_AL = importance_fields_al_t[32:40]; I08_AL = importance_fields_al_t[40:47]; 

I09_AL = importance_fields_al_t[47:53]; I10_AL = importance_fields_al_t[53:58]; 

I11_AL = importance_fields_al_t[58:69]; I12_AL = importance_fields_al_t[69:75];

I13_AL = importance_fields_al_t[75:81]; I14_AL = importance_fields_al_t[81:87]; 

I15_AL = importance_fields_al_t[87:93]; I16_AL = importance_fields_al_t[93:94]; 

I17_AL = importance_fields_al_t[94:100]; I18_AL = importance_fields_al_t[100:105]; 

I19_AL = importance_fields_al_t[105:112]; I20_AL = importance_fields_al_t[112:123]; 

I21_AL = importance_fields_al_t[123:125]; I22_AL = importance_fields_al_t[125:130]; 

I23_AL = importance_fields_al_t[130:135]; I24_AL = importance_fields_al_t[135:140];

I25_AL = importance_fields_al_t[140:148]; I26_AL = importance_fields_al_t[148:157];
#%% Convertendo para percentual

sum_variables = np.sum(importance_fields_al_t)

I01_AL = I01_AL/sum_variables; I02_AL = I02_AL/sum_variables; 

I03_AL = I03_AL/sum_variables; I04_AL = I04_AL/sum_variables; 

I05_AL = I05_AL/sum_variables; I06_AL = I06_AL/sum_variables;

I07_AL = I07_AL/sum_variables; I08_AL = I08_AL/sum_variables; 

I09_AL = I09_AL/sum_variables; I10_AL = I10_AL/sum_variables; 

I11_AL = I11_AL/sum_variables; I12_AL = I12_AL/sum_variables;

I13_AL = I13_AL/sum_variables; I14_AL = I14_AL/sum_variables; 

I15_AL = I15_AL/sum_variables; I16_AL = I16_AL/sum_variables; 

I17_AL = I17_AL/sum_variables; I18_AL = I18_AL/sum_variables; 

I19_AL = I19_AL/sum_variables; I20_AL = I20_AL/sum_variables; 

I21_AL = I21_AL/sum_variables; I22_AL = I22_AL/sum_variables; 

I23_AL = I23_AL/sum_variables; I24_AL = I24_AL/sum_variables;

I25_AL = I25_AL/sum_variables; I26_AL = I26_AL/sum_variables;

#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1 = [I01_AL[0],I01_AL[1],I01_AL[2],I01_AL[3],I01_AL[4]];
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
plt.savefig('QE_I01_AL_LS.png', dpi=450, bbox_inches='tight');

#%% VisuALization of Variable Importances
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
bar_width = 0.3;

x2 = ['Branca','Preta','Amarela','Parda','Indígena','Não quero declarar'];
y2 = [I02_AL[0],I02_AL[1],I02_AL[2],I02_AL[3],I02_AL[4],I02_AL[5]];
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
plt.savefig('QE_I02_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
bar_width = 0.3;

x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3 = [I03_AL[0],I03_AL[1],I03_AL[2]];
y3 = list(map(lambda t:t*100, y3))
y3_dt = [I03_AL[0],I03_AL[1],I03_AL[2]];
y3_dt = list(map(lambda t:t*100, y3_dt))

# Configurando a posição no eixo x
axis3 = np.arange(len(y3))
y31 = [x + bar_width for x in axis3]
y32 = [x + bar_width for x in y31]

# Fazendo o plot
plt.bar(y31, y3, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y3))], \
           x3,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Nacionalidade');
plt.legend();
plt.savefig('QE_I03_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
bar_width = 0.3;

x4 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y4 = [I04_AL[0],I04_AL[1],I04_AL[2],I04_AL[3],I04_AL[4],I04_AL[5]];
y4 = list(map(lambda t:t*100, y4))
# Configurando a posição no eixo x
axis4 = np.arange(len(y4))
y41 = [x + bar_width for x in axis4]
y42 = [x + bar_width for x in y41]

# Fazendo o plot
plt.bar(y41, y4, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y4))], \
           x4,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Escolarização da pai');
plt.legend();
plt.savefig('QE_I04_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
bar_width = 0.3;

x5 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y5 = [I05_AL[0],I05_AL[1],I05_AL[2],I05_AL[3],I05_AL[4],I05_AL[5]];
y5 = list(map(lambda t:t*100, y5))

# Configurando a posição no eixo x
axis5 = np.arange(len(y5))
y51 = [x + bar_width for x in axis5]
y52 = [x + bar_width for x in y51]

# Fazendo o plot
plt.bar(y51, y5, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y5))], \
           x5,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Escolarização da mãe');
plt.legend();
plt.savefig('QE_I05_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
bar_width = 0.3;

x6 = ['Casa/apartamento (sozinho)','Casa/apartamento (pais/parentes)',
      'Casa/apartamento (cônjugue/filhos)','Casa/apartamento (outras pessoas)',
      'Alojamento univ. na própria IES','Outro'];
y6 = [I06_AL[0],I06_AL[1],I06_AL[2],I06_AL[3],I06_AL[4],I06_AL[5]];
y6 = list(map(lambda t:t*100, y6));

# Configurando a posição no eixo x
axis6 = np.arange(len(y6))
y61 = [x + bar_width for x in axis6]
y62 = [x + bar_width for x in y61]

# Fazendo o plot
plt.bar(y61, y6, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y6))], \
           x6,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Onde e com quem moro');
plt.legend();
plt.savefig('QE_I06_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
bar_width = 0.3;

x7 = ['Nenhuma','Uma','Duas','Três','Quatro','Cinco','Seis','Sete ou mais'];
y7 = [I07_AL[0],I07_AL[1],I07_AL[2],I07_AL[3],
         I07_AL[4],I07_AL[5],I07_AL[6],I07_AL[7]];
y7 = list(map(lambda t:t*100, y7));

# Configurando a posição no eixo x
axis7 = np.arange(len(y7))
y71 = [x + bar_width for x in axis7]
y72 = [x + bar_width for x in y71]

# Fazendo o plot
plt.bar(y71, y7, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y7))], \
           x7,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Quantos moram com o estudante');
plt.legend();
plt.savefig('QE_I07_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
bar_width = 0.3;

x8 = ['Até 1,5 sál. mín','De 1 a 3 sál. mín.','De 3 a 4,5 sál. mín.',
      'De 4,5 a 6 sál. mín','De 6 a 10 sál. mín.','De 30 a 10 sál. mín',
      'Acima de 30 sál. mín.'];
y8 = [I08_AL[0],I08_AL[1],I08_AL[2],I08_AL[3],
         I08_AL[4],I08_AL[5],I08_AL[6]];
y8 = list(map(lambda t:t*100, y8));

# Configurando a posição no eixo x
axis8 = np.arange(len(y8))
y81 = [x + bar_width for x in axis8]
y82 = [x + bar_width for x in y81]

# Fazendo o plot
plt.bar(y81, y8, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y8))], \
           x8,\
           rotation=90, fontsize=8)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Renda total');
plt.legend();
plt.savefig('QE_I08_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I09
fig9 = plt.figure();
ax9 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x9 = ['Sem renda (financiamento governamental)','Sem renda (financ. por família/outros)',
      'Tenho renda, mas recebo ajuda (família/outras pessoas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a família',
      'Sou o principal a ajudar a família'];
y9 = [I09_AL[0],I09_AL[1],I09_AL[2],I09_AL[3],
         I09_AL[4],I09_AL[5]];
y9 = list(map(lambda t:t*100, y9));

# Configurando a posição no eixo x
axis9 = np.arange(len(y9))
y91 = [x + bar_width for x in axis9]
y92 = [x + bar_width for x in y91]

# Fazendo o plot
plt.bar(y91, y9, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y9))], \
           x9,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Situação financeira');
plt.legend();
plt.savefig('QE_I09_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
bar_width = 0.3;

x10 = ['Não estou trabalhando','Trabalho eventualmente','Trablho (até 20h/sem)',
       'Trabalho (de 21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10 = [I10_AL[0],I10_AL[1],I10_AL[2],I10_AL[3],
         I10_AL[4]];
y10 = list(map(lambda t:t*100, y10));

# Configurando a posição no eixo x
axis10 = np.arange(len(y10))
y101 = [x + bar_width for x in axis10]
y102 = [x + bar_width for x in y101]

# Fazendo o plot
plt.bar(y101, y10, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y10))], \
           x10,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Situação de trabalho');
plt.legend();
plt.savefig('QE_I10_AL_LS.png', dpi=450, bbox_inches='tight');

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
y11 = [I11_AL[0],I11_AL[1],I11_AL[2],I11_AL[3], I11_AL[4],
          I11_AL[5],I11_AL[6],I11_AL[7],I11_AL[8], I11_AL[9], I11_AL[10]];
y11 = list(map(lambda t:t*100, y11));

# Configurando a posição no eixo x
axis11 = np.arange(len(y11))
y111 = [x + bar_width for x in axis11]
y112 = [x + bar_width for x in y111]

# Fazendo o plot
plt.bar(y111, y11, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y11))], \
           x11,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.legend();
plt.savefig('QE_I11_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
bar_width = 0.3;

x12 = ['Nenhum','Moradia','Alimentação','Moradia e alimentação', 'Permanência','Outros'];
y12 = [I12_AL[0],I12_AL[1],I12_AL[2],I12_AL[3], I12_AL[4],
          I12_AL[5]];
y12 = list(map(lambda t:t*100, y12));

# Configurando a posição no eixo x
axis12 = np.arange(len(y12))
y121 = [x + bar_width for x in axis12]
y122 = [x + bar_width for x in y121]

# Fazendo o plot
plt.bar(y121, y12, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y12))], \
           x12,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Auxílio permanência');
plt.legend();
plt.savefig('QE_I12_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
bar_width = 0.3;

x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extensão','Bolsa monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13 = [I13_AL[0],I13_AL[1],I13_AL[2],I13_AL[3], I13_AL[4],
          I13_AL[5]];
y13 = list(map(lambda t:t*100, y13));

# Configurando a posição no eixo x
axis13 = np.arange(len(y13))
y131 = [x + bar_width for x in axis13]
y132 = [x + bar_width for x in y131]

# Fazendo o plot
plt.bar(y131, y13, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y13))], \
           x13,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Tipo de bolsa recebido');
plt.legend();
plt.savefig('QE_I13_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
bar_width = 0.3;

x14 = ['Não','Sim, Ciências sem Fronteiras', 'Sim, intercâmbio pelo Governo Federal',
       'Sim, intercâmbio pelo Governo Estadual', 'Sim, intercâmbio pela minha IES',
       'Sim, intercâmbio não institucional'];
y14 = [I14_AL[0],I14_AL[1],I14_AL[2],I14_AL[3], I14_AL[4],
          I14_AL[5]];
y14 = list(map(lambda t:t*100, y14));

# Configurando a posição no eixo x
axis14 = np.arange(len(y14))
y141 = [x + bar_width for x in axis14]
y142 = [x + bar_width for x in y141]

# Fazendo o plot
plt.bar(y141, y14, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y14))], \
           x14,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Programas de atividade no exterior');
plt.legend();
plt.savefig('QE_I14_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1]);
bar_width = 0.3;

x15 = ['Não','Sim, étnico-racial','Sim, renda', 'Sim, escola pública ou particular (com bolsa)',
       'Sim, combina dois mais', 'Sim, outra'];
y15 = [I15_AL[0],I15_AL[1],I15_AL[2],I15_AL[3], I15_AL[4],
          I15_AL[5]];
y15 = list(map(lambda t:t*100, y15));

# Configurando a posição no eixo x
axis15 = np.arange(len(y15))
y151 = [x + bar_width for x in axis15]
y152 = [x + bar_width for x in y151]

# Fazendo o plot
plt.bar(y151, y15, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y15))], \
           x15,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Ingresso por cota');
plt.legend();
plt.savefig('QE_I15_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
bar_width = 0.3;

x16 = ['AL'];
y16 = [I16_AL[0]];
y16 = list(map(lambda t:t*100, y16));

# Configurando a posição no eixo x
axis16 = np.arange(len(y16))
y161 = [x + bar_width for x in axis16]
y162 = [x + bar_width for x in y161]

# Fazendo o plot
plt.bar(y161, y16, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y16))], \
           x16,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('UF que concluiu o médio');
plt.legend();
plt.savefig('QE_I16_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
#bar_width = 0.1;

x17 = ['Todo em escola pública', 'Todo em escola privada','Todo no exterior',
       'Maior parte em escola pública','Maior parte em escola privada',
       'Parte no Brasil e parte no exterior'];
y17 = [I17_AL[0],I17_AL[1],I17_AL[2],I17_AL[3], I17_AL[4],
          I17_AL[5]];
y17 = list(map(lambda t:t*100, y17));

# Configurando a posição no eixo x
axis17 = np.arange(len(y17))
y171 = [x + bar_width for x in axis17]
y172 = [x + bar_width for x in y171]

# Fazendo o plot
plt.bar(y171, y17, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y17))], \
           x17,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Tipo de escola no médio');
plt.legend();
plt.savefig('QE_I17_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
#bar_width = 0.1;

x18 = ['Tradicional', 'Prof. técnico', 'Prof. magistério (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18 = [I18_AL[0],I18_AL[1],I18_AL[2],I18_AL[3], I18_AL[4]];
y18 = list(map(lambda t:t*100, y18));

# Configurando a posição no eixo x
axis18 = np.arange(len(y18))
y181 = [x + bar_width for x in axis18]
y182 = [x + bar_width for x in y181]

# Fazendo o plot
plt.bar(y181, y18, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y18))], \
           x18,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Modalidade do Ensino Médio');
plt.legend();
plt.savefig('QE_I18_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
#bar_width = 0.1;

x19 = ['Ninguém', 'Pais', 'Outros membros (excluindo os pais)', 'Professores', 
       'Líder ou representante religioso', 'Colegas/amigos', 'Outras pessoas'];
y19 = [I19_AL[0],I19_AL[1],I19_AL[2],I19_AL[3], 
          I19_AL[4], I19_AL[5], I19_AL[6]];
y19 = list(map(lambda t:t*100, y19));

# Configurando a posição no eixo x
axis19 = np.arange(len(y19))
y191 = [x + bar_width for x in axis19]
y192 = [x + bar_width for x in y191]

# Fazendo o plot
plt.bar(y191, y19, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y19))], \
           x19,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Maior incentivo para cursar a graduação');
plt.legend();
plt.savefig('QE_I19_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
#bar_width = 0.1;

x20 = ['Não tive dificuldade', 'Não recebi apoio', 'Pais', 'Avós', 'Irmãos, primos ou tios',
       'Líder ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do serviço de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20 = [I20_AL[0],I20_AL[1],I20_AL[2],I20_AL[3], I20_AL[4], I20_AL[5], 
          I20_AL[6], I20_AL[7], I20_AL[8], I20_AL[9], I20_AL[10]];
y20 = list(map(lambda t:t*100, y20));

# Configurando a posição no eixo x
axis20 = np.arange(len(y20))
y201 = [x + bar_width for x in axis20]
y202 = [x + bar_width for x in y201]

# Fazendo o plot
plt.bar(y201, y20, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y20))], \
           x20,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e concluí-lo');
plt.legend();
plt.savefig('QE_I20_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
#bar_width = 0.1;

x21 = ['Sim', 'Não'];
y21 = [I21_AL[0],I21_AL[1]];
y21 = list(map(lambda t:t*100, y21));

# Configurando a posição no eixo x
axis21 = np.arange(len(y21))
y211 = [x + bar_width for x in axis21]
y212 = [x + bar_width for x in y211]

# Fazendo o plot
plt.bar(y211, y21, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y21))], \
           x21,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Alguém da família concluiu curso superior');
plt.legend();
plt.savefig('QE_I21_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
#bar_width = 0.1;

x22 = ['Nenhum  ', 'Um ou dois', 'Três a cinco', 'Seis a oito', 'Mais de oito'];
y22 = [I22_AL[0],I22_AL[1],I22_AL[2],I22_AL[3], I22_AL[4]];
y22 = list(map(lambda t:t*100, y22));

# Configurando a posição no eixo x
axis22 = np.arange(len(y22))
y221 = [x + bar_width for x in axis22]
y222 = [x + bar_width for x in y221]

# Fazendo o plot
plt.bar(y221, y22, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y22))], \
           x22,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Livros lido no ano (excluindo da Biografia do curso)');
plt.legend();
plt.savefig('QE_I22_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
#bar_width = 0.1;

x23 = ['Nenhuma', 'De uma a três', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23 = [I23_AL[0],I23_AL[1],I23_AL[2],I23_AL[3],I23_AL[4]];
y23 = list(map(lambda t:t*100, y23));

# Configurando a posição no eixo x
axis23 = np.arange(len(y23))
y231 = [x + bar_width for x in axis23]
y232 = [x + bar_width for x in y231]

# Fazendo o plot
plt.bar(y231, y23, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y23))], \
           x23,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.legend();
plt.savefig('QE_I23_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
#bar_width = 0.1;

x24 = ['Sim, apenas presencial', 'Sim, apenas semipresencial', 
       'Sim, parte presencial e parte semipresencial', 'Sim, EAD', 'Não'];
y24 = [I24_AL[0],I24_AL[1],I24_AL[2],I24_AL[3], I24_AL[4]];
y24 = list(map(lambda t:t*100, y24));

# Configurando a posição no eixo x
axis24 = np.arange(len(y24))
y241 = [x + bar_width for x in axis24]
y242 = [x + bar_width for x in y241]

# Fazendo o plot
plt.bar(y241, y24, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y24))], \
           x24,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.legend();
plt.savefig('QE_I24_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
#bar_width = 0.1;

x25 = ['Inserção no mercado de trabalho', 'Influência familiar','Valorização profissional',
       'Prestígio social', 'Vocação', 'Oferecido na modalidade EAD',
       'Baixa concorrência', 'Outro motivo'];
y25 = [I25_AL[0],I25_AL[1],I25_AL[2],I25_AL[3], 
          I25_AL[4], I25_AL[5], I25_AL[6], I25_AL[7]];
y25 = list(map(lambda t:t*100, y25));

# Configurando a posição no eixo x
axis25 = np.arange(len(y25))
y251 = [x + bar_width for x in axis25]
y252 = [x + bar_width for x in y251]

# Fazendo o plot
plt.bar(y251, y25, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y25))], \
           x25,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Porque escolhi o curso');
plt.legend();
plt.savefig('QE_I25_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
#bar_width = 0.1;

x26 = ['Gratuidade', 'Preço da mensalidade', 'Prox. a residência', 'Prox. ao trabalho', 
       'Facilidade de acesso', 'Qualidade/reputação', 'Única opção de aprovação',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26 = [I26_AL[0],I26_AL[1],I26_AL[2],I26_AL[3], I26_AL[4], I26_AL[5], 
          I26_AL[6], I26_AL[7], I26_AL[8]];
y26 = list(map(lambda t:t*100, y26));

# Configurando a posição no eixo x
axis26 = np.arange(len(y26))
y261 = [x + bar_width for x in axis26]
y262 = [x + bar_width for x in y261]

# Fazendo o plot
plt.bar(y261, y26, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y26))], \
           x26,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('Porque escolhi essa IES');
plt.legend();
plt.savefig('QE_I26_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27a
fig27a = plt.figure();
ax27aa = fig27a.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27a = ['QE_I01_AL', 'QE_I02_AL', 'QE_I03_AL', 'QE_I04_AL', 'QE_I05_AL', 'QE_I06_AL',
         'QE_I07_AL', 'QE_I08_AL', 'QE_I09_AL', 'QE_I10_AL', 'QE_I11_AL', 'QE_I12_AL', 
         'QE_I13_AL'];
y27a = [np.sum(I01_AL),np.sum(I02_AL),np.sum(I03_AL),np.sum(I04_AL),
          np.sum(I05_AL),np.sum(I06_AL),np.sum(I07_AL),np.sum(I08_AL),
          np.sum(I09_AL),np.sum(I10_AL),np.sum(I11_AL),np.sum(I12_AL),
          np.sum(I13_AL)];
y27a = list(map(lambda t:t*100, y27a));

# Configurando a posição no eixo x
axis27a = np.arange(len(y27a))
y27a1 = [x + bar_width for x in axis27a]
y27a2 = [x + bar_width for x in y27a1]

# Fazendo o plot
plt.bar(y27a1, y27a, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27a))], \
           ax27a,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)');
plt.xlabel('Variável');
plt.title('QE_I01 a QE_I13');
plt.legend();
plt.savefig('QE_I27a_AL_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27b
fig27b = plt.figure();
ax27ab = fig27b.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27b = ['QE_I14_AL', 'QE_I15_AL', 'QE_I16_AL', 'QE_I17_AL', 'QE_I18_AL', 'QE_I19_AL',
         'QE_I20_AL', 'QE_I21_AL', 'QE_I22_AL', 'QE_I23_AL', 'QE_I24_AL', 'QE_I25_AL', 
         'QE_I26_AL'];
y27b = [np.sum(I14_AL),np.sum(I15_AL),np.sum(I16_AL),np.sum(I17_AL),
          np.sum(I18_AL),np.sum(I19_AL),np.sum(I20_AL), np.sum(I21_AL),
          np.sum(I22_AL),np.sum(I23_AL),np.sum(I24_AL),
          np.sum(I25_AL),np.sum(I26_AL)];
y27b = list(map(lambda t:t*100, y27b));

# Configurando a posição no eixo x
axis27b = np.arange(len(y27b))
y27b1 = [x + bar_width for x in axis27b]
y27b2 = [x + bar_width for x in y27b1]

# Fazendo o plot
plt.bar(y27b1, y27b, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27b))], \
           ax27b,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)'); 
plt.xlabel('Variável');
plt.title('QE_I14 a QE_I26');
plt.legend();
plt.savefig('QE_I27b_AL_LS.png', dpi=450, bbox_inches='tight');