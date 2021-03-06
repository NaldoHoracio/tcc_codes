"""
T��tulo: Lasso Regression aplicado a dados no Brasil (excluindo Alagoas)

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
print('O formato dos dados �: ', features_br.shape)

describe_br = features_br.describe()

print('Descri��o para as colunas: ', describe_br)
print(describe_br.columns)

#%% N�meros que s�o strings para float
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
print('O formato dos dados �: ', features_br.shape)

describe_br = features_br.describe()

print('Descri��o para as colunas: ', describe_br)
print(describe_br.columns)
#%% Convertendo os labels de predi��o para arrays numpy
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
    
    # Usando o Random Forest para predi��o dos dados
    predictions_br = lasso_br.predict(test_features_br)
    
    # Erro
    errors_br = abs(predictions_br - test_labels_br)
    
    # Acur�cia
    accuracy_br = 100 - mean_absolute_error(test_labels_br, predictions_br)
    
    # Import�nncia das vari�veis
    importance_fields_aux_br = lasso_br.coef_
    importance_fields_br += importance_fields_aux_br
    
    
    # Append em cada valor m�dio
    scores_br.append(accuracy_br)

#%% - Acur�cia BR
print('Accuracy: ', round(np.average(scores_br), 2), "%.")

importance_fields_br_t = importance_fields_br/number_splits

#%% Importancia das vari�veis
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

x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Vi�vo', 'Outro'];
y1 = [I01_BR[0],I01_BR[1],I01_BR[2],I01_BR[3],I01_BR[4]];
y1 = list(map(lambda t:t*100, y1))

# Configurando a posi��o no eixo x
axis1 = np.arange(len(y1))
y11 = [x + bar_width for x in axis1]

# Fazendo o plot
plt.bar(y11, y1, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y1))], \
           ['Solteiro', 'Casado (a)', 'Separado', 'Vi�vo', 'Outro'],\
           rotation=90, fontsize=8)
plt.ylabel('Import�ncia (%)');
plt.xlabel('Vari�vel');
plt.title('Estado civil');
plt.legend();
plt.savefig('QE_I01_BR_LS.png', dpi=450, bbox_inches='tight');

#%% VisuBRization of Variable Importances
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
bar_width = 0.3;

x2 = ['Branca','Preta','Amarela','Parda','Ind�gena','N�o quero declarar'];
y2 = [I02_BR[0],I02_BR[1],I02_BR[2],I02_BR[3],I02_BR[4],I02_BR[5]];
y2 = list(map(lambda t:t*100, y2))

# Configurando a posi��o no eixo x
axis2 = np.arange(len(y2))
y21 = [x + bar_width for x in axis2]

# Fazendo o plot
plt.bar(y21, y2, color='red', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y2))], \
           x2,\
           rotation=90, fontsize=8)
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Cor/ra�a');
plt.legend();
plt.savefig('QE_I02_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
bar_width = 0.3;

x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3 = [I03_BR[0],I03_BR[1],I03_BR[2]];
y3 = list(map(lambda t:t*100, y3))

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Nacionalidade');
plt.legend();
plt.savefig('QE_I03_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
bar_width = 0.3;

x4 = ['Nenhum','1� ao 5� ano','6� ao 9� ano','Ensino m�dio','Gradua��o','P�s-gradua��o'];
y4 = [I04_BR[0],I04_BR[1],I04_BR[2],I04_BR[3],I04_BR[4],I04_BR[5]];
y4 = list(map(lambda t:t*100, y4))
# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Escolariza��o da pai');
plt.legend();
plt.savefig('QE_I04_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
bar_width = 0.3;

x5 = ['Nenhum','1� ao 5� ano','6� ao 9� ano','Ensino m�dio','Gradua��o','P�s-gradua��o'];
y5 = [I05_BR[0],I05_BR[1],I05_BR[2],I05_BR[3],I05_BR[4],I05_BR[5]];
y5 = list(map(lambda t:t*100, y5))

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Escolariza��o da m�e');
plt.legend();
plt.savefig('QE_I05_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
bar_width = 0.3;

x6 = ['Casa/apartamento (sozinho)','Casa/apartamento (pais/parentes)',
      'Casa/apartamento (c�njugue/filhos)','Casa/apartamento (outras pessoas)',
      'Alojamento univ. na pr�pria IES','Outro'];
y6 = [I06_BR[0],I06_BR[1],I06_BR[2],I06_BR[3],I06_BR[4],I06_BR[5]];
y6 = list(map(lambda t:t*100, y6));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Onde e com quem moro');
plt.legend();
plt.savefig('QE_I06_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
bar_width = 0.3;

x7 = ['Nenhuma','Uma','Duas','Tr�s','Quatro','Cinco','Seis','Sete ou mais'];
y7 = [I07_BR[0],I07_BR[1],I07_BR[2],I07_BR[3],
         I07_BR[4],I07_BR[5],I07_BR[6],I07_BR[7]];
y7 = list(map(lambda t:t*100, y7));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Quantos moram com o estudante');
plt.legend();
plt.savefig('QE_I07_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
bar_width = 0.3;

x8 = ['At� 1,5 s�l. m�n','De 1 a 3 s�l. m�n.','De 3 a 4,5 s�l. m�n.',
      'De 4,5 a 6 s�l. m�n','De 6 a 10 s�l. m�n.','De 30 a 10 s�l. m�n',
      'Acima de 30 s�l. m�n.'];
y8 = [I08_BR[0],I08_BR[1],I08_BR[2],I08_BR[3],
         I08_BR[4],I08_BR[5],I08_BR[6]];
y8 = list(map(lambda t:t*100, y8));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Renda total');
plt.legend();
plt.savefig('QE_I08_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I09
fig9 = plt.figure();
ax9 = fig1.add_axes([0,0,1,1]);
bar_width = 0.3;

x9 = ['Sem renda (financiamento governamental)','Sem renda (financ. por fam�lia/outros)',
      'Tenho renda, mas recebo ajuda (fam�lia/outras pessoas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a fam�lia',
      'Sou o principal a ajudar a fam�lia'];
y9 = [I09_BR[0],I09_BR[1],I09_BR[2],I09_BR[3],
         I09_BR[4],I09_BR[5]];
y9 = list(map(lambda t:t*100, y9));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Situa��o financeira');
plt.legend();
plt.savefig('QE_I09_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
bar_width = 0.3;

x10 = ['N�o estou trabalhando','Trabalho eventualmente','Trablho (at� 20h/sem)',
       'Trabalho (de 21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10 = [I10_BR[0],I10_BR[1],I10_BR[2],I10_BR[3],
         I10_BR[4]];
y10 = list(map(lambda t:t*100, y10));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Situa��o de trabalho');
plt.legend();
plt.savefig('QE_I10_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I11
fig11 = plt.figure();
ax11 = fig11.add_axes([0,0,1,1]);
bar_width = 0.3;

x11 = ['Nenhum (curso gratuito)','Nenhum (mas n�o gratuito)','ProUni integral',
       'ProUni parcial, apenas','FIES, apenas','ProUni parcial e FIES',
       'Bolsa do governo (estadual/distrital/municipal)',
       'Bolsa pela IES','Bolsa por outra entidade','Financiamento pela IES',
       'Financiamento banc�rio'];
y11 = [I11_BR[0],I11_BR[1],I11_BR[2],I11_BR[3], I11_BR[4],
          I11_BR[5],I11_BR[6],I11_BR[7],I11_BR[8], I11_BR[9], I11_BR[10]];
y11 = list(map(lambda t:t*100, y11));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.legend();
plt.savefig('QE_I11_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
bar_width = 0.3;

x12 = ['Nenhum','Moradia','Alimenta��o','Moradia e alimenta��o', 'Perman�ncia','Outros'];
y12 = [I12_BR[0],I12_BR[1],I12_BR[2],I12_BR[3], I12_BR[4],
          I12_BR[5]];
y12 = list(map(lambda t:t*100, y12));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Aux�lio perman�ncia');
plt.legend();
plt.savefig('QE_I12_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
bar_width = 0.3;

x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extens�o','Bolsa monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13 = [I13_BR[0],I13_BR[1],I13_BR[2],I13_BR[3], I13_BR[4],
          I13_BR[5]];
y13 = list(map(lambda t:t*100, y13));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Tipo de bolsa recebido');
plt.legend();
plt.savefig('QE_I13_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
bar_width = 0.3;

x14 = ['N�o','Sim, Ci�ncias sem Fronteiras', 'Sim, interc�mbio pelo Governo Federal',
       'Sim, interc�mbio pelo Governo Estadual', 'Sim, interc�mbio pela minha IES',
       'Sim, interc�mbio n�o institucional'];
y14 = [I14_BR[0],I14_BR[1],I14_BR[2],I14_BR[3], I14_BR[4],
          I14_BR[5]];
y14 = list(map(lambda t:t*100, y14));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Programas de atividade no exterior');
plt.legend();
plt.savefig('QE_I14_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1]);
bar_width = 0.3;

x15 = ['N�o','Sim, �tnico-racial','Sim, renda', 'Sim, escola p�blica ou particular (com bolsa)',
       'Sim, combina dois mais', 'Sim, outra'];
y15 = [I15_BR[0],I15_BR[1],I15_BR[2],I15_BR[3], I15_BR[4],
          I15_BR[5]];
y15 = list(map(lambda t:t*100, y15));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Ingresso por cota');
plt.legend();
plt.savefig('QE_I15_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
bar_width = 0.3;

x16 = ['RO','AC','AM','RR','PA','AP','TO','MA','PI','CE','RN','PB','PE',
       'SE','BA','MG','ES','RJ','SP','PR','SC','RS','MS','MT','GO','DF','Outro'];
y16 = [I16_BR[0],I16_BR[1],I16_BR[2],I16_BR[3],I16_BR[4],I16_BR[5],
          I16_BR[6],I16_BR[7],I16_BR[8],I16_BR[9],I16_BR[10],I16_BR[11],
          I16_BR[12],I16_BR[13],I16_BR[14],I16_BR[15],I16_BR[16],I16_BR[17],
          I16_BR[18],I16_BR[19],I16_BR[20],I16_BR[21],I16_BR[22],I16_BR[23],
          I16_BR[24],I16_BR[25],I16_BR[26]];
y16 = list(map(lambda t:t*100, y16));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('UF que concluiu o m�dio');
plt.legend();
plt.savefig('QE_I16_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
#bar_width = 0.1;

x17 = ['Todo em escola p�blica', 'Todo em escola privada','Todo no exterior',
       'Maior parte em escola p�blica','Maior parte em escola privada',
       'Parte no Brasil e parte no exterior'];
y17 = [I17_BR[0],I17_BR[1],I17_BR[2],I17_BR[3], I17_BR[4],
          I17_BR[5]];
y17 = list(map(lambda t:t*100, y17));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Tipo de escola no m�dio');
plt.legend();
plt.savefig('QE_I17_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
#bar_width = 0.1;

x18 = ['Tradicional', 'Prof. t�cnico', 'Prof. magist�rio (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18 = [I18_BR[0],I18_BR[1],I18_BR[2],I18_BR[3], I18_BR[4]];
y18 = list(map(lambda t:t*100, y18));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Modalidade do Ensino M�dio');
plt.legend();
plt.savefig('QE_I18_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
#bar_width = 0.1;

x19 = ['Ningu�m', 'Pais', 'Outros membros (excluindo os pais)', 'Professores', 
       'L�der ou representante religioso', 'Colegas/amigos', 'Outras pessoas'];
y19 = [I19_BR[0],I19_BR[1],I19_BR[2],I19_BR[3], 
          I19_BR[4], I19_BR[5], I19_BR[6]];
y19 = list(map(lambda t:t*100, y19));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Maior incentivo para cursar a gradua��o');
plt.legend();
plt.savefig('QE_I19_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
#bar_width = 0.1;

x20 = ['N�o tive dificuldade', 'N�o recebi apoio', 'Pais', 'Av�s', 'Irm�os, primos ou tios',
       'L�der ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do servi�o de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20 = [I20_BR[0],I20_BR[1],I20_BR[2],I20_BR[3], I20_BR[4], I20_BR[5], 
          I20_BR[6], I20_BR[7], I20_BR[8], I20_BR[9], I20_BR[10]];
y20 = list(map(lambda t:t*100, y20));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e conclu�-lo');
plt.legend();
plt.savefig('QE_I20_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
#bar_width = 0.1;

x21 = ['Sim', 'N�o'];
y21 = [I21_BR[0],I21_BR[1]];
y21 = list(map(lambda t:t*100, y21));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Algu�m da fam�lia concluiu curso superior');
plt.legend();
plt.savefig('QE_I21_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
#bar_width = 0.1;

x22 = ['Nenhum  ', 'Um ou dois', 'Tr�s a cinco', 'Seis a oito', 'Mais de oito'];
y22 = [I22_BR[0],I22_BR[1],I22_BR[2],I22_BR[3], I22_BR[4]];
y22 = list(map(lambda t:t*100, y22));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Livros lido no ano (excluindo da Biografia do curso)');
plt.legend();
plt.savefig('QE_I22_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
#bar_width = 0.1;

x23 = ['Nenhuma', 'De uma a tr�s', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23 = [I23_BR[0],I23_BR[1],I23_BR[2],I23_BR[3],I23_BR[4]];
y23 = list(map(lambda t:t*100, y23));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.legend();
plt.savefig('QE_I23_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
#bar_width = 0.1;

x24 = ['Sim, apenas presencial', 'Sim, apenas semipresencial', 
       'Sim, parte presencial e parte semipresencial', 'Sim, EAD', 'N�o'];
y24 = [I24_BR[0],I24_BR[1],I24_BR[2],I24_BR[3], I24_BR[4]];
y24 = list(map(lambda t:t*100, y24));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.legend();
plt.savefig('QE_I24_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
#bar_width = 0.1;

x25 = ['Inser��o no mercado de trabalho', 'Influ�ncia familiar','Valoriza��o profissional',
       'Prest�gio social', 'Voca��o', 'Oferecido na modalidade EAD',
       'Baixa concorr�ncia', 'Outro motivo'];
y25 = [I25_BR[0],I25_BR[1],I25_BR[2],I25_BR[3], 
          I25_BR[4], I25_BR[5], I25_BR[6], I25_BR[7]];
y25 = list(map(lambda t:t*100, y25));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Porque escolhi o curso');
plt.legend();
plt.savefig('QE_I25_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
#bar_width = 0.1;

x26 = ['Gratuidade', 'Pre�o da mensalidade', 'Prox. a resid�ncia', 'Prox. ao trabalho', 
       'Facilidade de acesso', 'Qualidade/reputa��o', '�nica op��o de aprova��o',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26 = [I26_BR[0],I26_BR[1],I26_BR[2],I26_BR[3], I26_BR[4], I26_BR[5], 
          I26_BR[6], I26_BR[7], I26_BR[8]];
y26 = list(map(lambda t:t*100, y26));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('Porque escolhi essa IES');
plt.legend();
plt.savefig('QE_I26_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27a
fig27a = plt.figure();
ax27aa = fig27a.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27a = ['QE_I01_BR', 'QE_I02_BR', 'QE_I03_BR', 'QE_I04_BR', 'QE_I05_BR', 'QE_I06_BR',
         'QE_I07_BR', 'QE_I08_BR', 'QE_I09_BR', 'QE_I10_BR', 'QE_I11_BR', 'QE_I12_BR', 
         'QE_I13_BR'];
y27a = [np.sum(I01_BR),np.sum(I02_BR),np.sum(I03_BR),np.sum(I04_BR),
          np.sum(I05_BR),np.sum(I06_BR),np.sum(I07_BR),np.sum(I08_BR),
          np.sum(I09_BR),np.sum(I10_BR),np.sum(I11_BR),np.sum(I12_BR),
          np.sum(I13_BR)];
y27a = list(map(lambda t:t*100, y27a));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)');
plt.xlabel('Vari�vel');
plt.title('QE_I01 a QE_I13');
plt.legend();
plt.savefig('QE_I27a_BR_LS.png', dpi=450, bbox_inches='tight');

#%% Visualization of Variable Importances
# QE_I27b
fig27b = plt.figure();
ax27ab = fig27b.add_axes([0,0,1,1]);
#bar_width = 0.3;

ax27b = ['QE_I14_BR', 'QE_I15_BR', 'QE_I16_BR', 'QE_I17_BR', 'QE_I18_BR', 'QE_I19_BR',
         'QE_I20_BR', 'QE_I21_BR', 'QE_I22_BR', 'QE_I23_BR', 'QE_I24_BR', 'QE_I25_BR', 
         'QE_I26_BR'];
y27b = [np.sum(I14_BR),np.sum(I15_BR),np.sum(I16_BR),np.sum(I17_BR),
          np.sum(I18_BR),np.sum(I19_BR),np.sum(I20_BR), np.sum(I21_BR),
          np.sum(I22_BR),np.sum(I23_BR),np.sum(I24_BR),
          np.sum(I25_BR),np.sum(I26_BR)];
y27b = list(map(lambda t:t*100, y27b));

# Configurando a posi��o no eixo x
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
plt.ylabel('Import�ncia (%)'); 
plt.xlabel('Vari�vel');
plt.title('QE_I14 a QE_I26');
plt.legend();
plt.savefig('QE_I27b_BR_LS.png', dpi=450, bbox_inches='tight');