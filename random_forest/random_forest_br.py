# -*- coding: utf-8 -*-
"""
Created on Thu 26  15:37:56 2020

@author: edvonaldo
"""

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

#%% One hot encoding - QE_I01 a QE_I26
#features_al = pd.get_dummies(data=features_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
#                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
#                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
#                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
#                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
#                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
#                                                        'QE_I25','QE_I26'])
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
from sklearn.ensemble import RandomForestRegressor

scores_br = []

importance_fields_br = 0.0
importance_fields_aux_br = []

rf_br = RandomForestRegressor(n_estimators = 500, random_state=0)

kf_cv_br = KFold(n_splits=11, random_state=None, shuffle=False) # n_splits: divisores de 7084 ^ memory

for train_index_br, test_index_br in kf_cv_br.split(features_br):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_br), '-', np.max(test_index_br))
    
    # Dividindo nas features e labels
    train_features_br = features_br[train_index_br]
    test_features_br = features_br[test_index_br]
    train_labels_br = labels_br[train_index_br]
    test_labels_br = labels_br[test_index_br]
    
    # Ajustando cada features e label com RF
    rf_br.fit(train_features_br, train_labels_br)
    
    # Usando o Random Forest para predição dos dados
    predictions_br = rf_br.predict(test_features_br)
    
    # Erro
    errors_br = abs(predictions_br - test_labels_br)
    
    # Acurácia
    accuracy_br = 100 - mean_absolute_error(test_labels_br, predictions_br)
    
    # Importância das variáveis
    importance_fields_aux_br = rf_br.feature_importances_
    importance_fields_br += importance_fields_aux_br
    
    # Append em cada valor médio
    scores_br.append(accuracy_br)

#%% Acurácia AL
print('Accuracy: ', round(np.average(scores_br), 2), "%.")

importance_fields_br_t = importance_fields_br/11

print('Total: ', round(np.sum(importance_fields_br_t),8))

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



#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1 = [I01_BR[0],I01_BR[1],I01_BR[2],I01_BR[3],I01_BR[4]];
ax1.bar(x1,y1);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Estado civil');
plt.savefig('QE_I01_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
x2 = ['Branca','Preta','Amarela','Parda','Indígena','Não quero declarar'];
y2 =[I02_BR[0],I02_BR[1],I03_BR[2],I02_BR[3],I02_BR[4],I02_BR[5]];
ax2.bar(x2,y2);
plt.xticks(x2,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Cor/raça');
plt.savefig('QE_I02_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3 = [I03_BR[0],I03_BR[1],I03_BR[2]];
ax3.bar(x3,y3);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Nacionalidade');
plt.savefig('QE_I03_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
x4 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y4 = [I04_BR[0],I04_BR[1],I04_BR[2],I04_BR[3],I04_BR[4],I04_BR[5]];
ax4.bar(x4,y4);
plt.xticks(x4,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Escolarização da pai');
plt.savefig('QE_I04_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
x5 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y5 = [I05_BR[0],I05_BR[1],I05_BR[2],I05_BR[3],I05_BR[4],I05_BR[5]];
ax5.bar(x5,y5);
plt.xticks(x5,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Escolarização da mãe');
plt.savefig('QE_I05_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
x6 = ['Casa/apartamento (sozinho)','Casa/apartamento (pais/parentes)',
      'Casa/apartamento (cônjugue/filhos)','Casa/apartamento (outras pessoas)',
      'Alojamento univ. na própria IES','Outro'];
y6 = [I06_BR[0],I06_BR[1],I06_BR[2],I06_BR[3],I06_BR[4],I06_BR[5]];
ax6.bar(x6,y6);
plt.xticks(x6,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Onde e com quem moro');
plt.savefig('QE_I06_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
x7 = ['Nenhuma','Uma','Duas','Três','Quatro','Cinco','Seis','Sete ou mais'];
y7 = [I07_BR[0],I07_BR[1],I07_BR[2],I07_BR[3],I07_BR[4],I07_BR[5],I07_BR[6],I07_BR[7]];
ax7.bar(x7,y7);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Quantos moram com o estudante');
plt.savefig('QE_I07_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
x8 = ['Até 1,5 sál. mín','De 1 a 3 sál. mín.','De 3 a 4,5 sál. mín.',
      'De 4,5 a 6 sál. mín','De 6 a 10 sál. mín.','De 30 a 10 sál. mín',
      'Acima de 30 sál. mín.'];
y8 = [I08_BR[0],I08_BR[1],I08_BR[2],I08_BR[3],I08_BR[4],I08_BR[5],I08_BR[6]];
ax8.bar(x8,y8);
plt.xticks(x8,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Renda total');
plt.savefig('QE_I08_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I09
fig9 = plt.figure();
ax9 = fig9.add_axes([0,0,1,1]);
x9 = ['Sem renda (financiamento governamental)','Sem renda (financ. por família/outros)',
      'Tenho renda, mas recebo ajuda (família/outras pessoas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a família',
      'Sou o principal a ajudar a família'];
y9 = [I09_BR[0],I09_BR[1],I09_BR[2],I09_BR[3],I09_BR[4],I09_BR[5]];
ax9.bar(x9,y9);
plt.xticks(x9,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Situação financeira');
plt.savefig('QE_I09_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
x10 = ['Não estou trabalhando','Trabalho eventualmente','Trablho (até 20h/sem)',
       'Trabalho (de 21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10 = [I10_BR[0],I10_BR[1],I10_BR[2],I10_BR[3],I10_BR[4]];
ax10.bar(x10,y10);
plt.xticks(x10,rotation=90,fontsize=8)
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Situação de trabalho');
plt.savefig('QE_I10_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I11
fig11 = plt.figure();
ax11 = fig11.add_axes([0,0,1,1]);
x11 = ['Nenhum (curso gratuito)','Nenhum (mas não gratuito)','ProUni integral',
       'ProUni parcial, apenas','FIES, apenas','ProUni parcial e FIES',
       'Bolsa do governo (estadual/distrital/municipal)',
       'Bolsa pela IES','Bolsa por outra entidade','Financiamento pela IES',
       'Financiamento bancário'];
y11 = [I11_BR[0],I11_BR[1],I11_BR[2],I11_BR[3],I11_BR[4],I11_BR[5],I11_BR[6],I11_BR[7],I11_BR[8],I11_BR[9],I11_BR[10]];
ax11.bar(x11,y11);
plt.xticks(x11,rotation=90,fontsize=8)
#plt.legend(x11,loc=2)
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.savefig('QE_I11_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
x12 = ['Nenhum','Moradia','Alimentação','Moradia e alimentação', 'Permanência','Outros'];
y12 = [I12_BR[0],I12_BR[1],I12_BR[2],I12_BR[3],I12_BR[4],I12_BR[5]];
ax12.bar(x12,y12);
plt.xticks(x12,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Auxílio permanência');
plt.savefig('QE_I12_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extensão','Bolsa monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13 = [I13_BR[0],I13_BR[1],I13_BR[2],I13_BR[3],I13_BR[4],I13_BR[5]];
ax13.bar(x13,y13);
plt.xticks(x13,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Tipo de bolsa recebido');
plt.savefig('QE_I13_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
x14 = ['Não','Sim, Ciências sem Fronteiras', 'Sim, intercâmbio pelo Governo Federal',
       'Sim, intercâmbio pelo Governo Estadual', 'Sim, intercâmbio pela minha IES',
       'Sim, intercâmbio não institucional'];
y14 = [I14_BR[0],I14_BR[1],I14_BR[2],I14_BR[3],I14_BR[4],I14_BR[5]];
ax14.bar(x14,y14);
plt.xticks(x14,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Programas de atividade no exterior');
plt.savefig('QE_I14_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1])
x15 = ['Não','Sim, étnico-racial','Sim, renda', 'Sim, escola pública ou particular (com bolsa)',
       'Sim, combina dois mais', 'Sim, outra'];
y15 = [I15_BR[0],I15_BR[1],I15_BR[2],I15_BR[3],I15_BR[4],I15_BR[5]];
ax15.bar(x15,y15);
plt.xticks(x15,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Ingresso por cota');
plt.savefig('QE_I15_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
x16 = ['RO','AC','AM','RR','PA','AP','TO','MA','PI','CE','RN','PB','PE',
       'SE','BA','MG','ES','RJ','SP','PR','SC','RS','MS','MT','GO','DF','Outro'];
y16 = [I16_BR[0],I16_BR[1],I16_BR[2],I16_BR[3],I16_BR[4],I16_BR[5],I16_BR[6],I16_BR[7],
       I16_BR[8],I16_BR[9],I16_BR[10],I16_BR[11],I16_BR[12],I16_BR[13],I16_BR[14],I16_BR[15],
       I16_BR[16],I16_BR[17],I16_BR[18],I16_BR[19],I16_BR[20],I16_BR[21],I16_BR[22],I16_BR[23],
       I16_BR[24],I16_BR[25],I16_BR[26]];
ax16.bar(x16,y16);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('UF que concluiu o médio');
plt.savefig('QE_I16_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
x17 = ['Todo em escola pública', 'Todo em escola privada','Todo no exterior',
       'Maior parte em escola pública','Maior parte em escola privada',
       'Parte no Brasil e parte no exterior'];
y17 = [I17_BR[0],I17_BR[1],I17_BR[2],I17_BR[3],I17_BR[4],I17_BR[5]];
ax17.bar(x17,y17);
plt.xticks(x17,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Tipo de escola no médio');
plt.savefig('QE_I17_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
x18 = ['Tradicional', 'Prof. técnico', 'Prof. magistério (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18 = [I18_BR[0],I18_BR[1],I18_BR[2],I18_BR[3],I18_BR[4]];
ax18.bar(x18,y18);
plt.xticks(x18,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Modalidade do Ensino Médio');
plt.savefig('QE_I18_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
x19 = ['Ninguém', 'Pais', 'Outros membros (excluindo os pais)', 'Professores', 
       'Líder ou representante religioso', 'Colegas/amigos', 'Outras pessoas']
y19 = [I19_BR[0],I19_BR[1],I19_BR[2],I19_BR[3],I19_BR[4],I19_BR[5],I19_BR[6]];
ax19.bar(x19,y19);
plt.xticks(x19,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Maior incentivo para cursar a graduação');
plt.savefig('QE_I20_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
x20 = ['Não tive dificuldade', 'Não recebi apoio', 'Pais', 'Avós', 'Irmãos, primos ou tios',
       'Líder ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do serviço de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20 = [I20_BR[0],I20_BR[1],I20_BR[2],I20_BR[3],I20_BR[4],I20_BR[5],I20_BR[6],I20_BR[7],I20_BR[8],I20_BR[9],I20_BR[10]];
ax20.bar(x20,y20);
plt.xticks(x20,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e concluí-lo');
plt.savefig('QE_I20_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
x21 = ['Sim', 'Não'];
y21 = [I21_BR[0],I21_BR[1]];
ax21.bar(x21,y21);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Alguém da família concluiu curso superior');
plt.savefig('QE_I21_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
x22 = ['Nenhum  ', 'Um ou dois', 'Três a cinco', 'Seis a oito', 'Mais de oito'];
y22 = [I22_BR[0],I22_BR[1],I22_BR[2],I22_BR[3],I22_BR[4]];
ax22.bar(x22,y22);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Livros lido no ano (excluindo da Biografia do curso');
plt.savefig('QE_I22_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
x23 = ['Nenhuma', 'De uma a três', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23 = [I23_BR[0],I23_BR[1],I23_BR[2],I23_BR[3],I23_BR[4]];
ax23.bar(x23,y23);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.savefig('QE_I23_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
x24 = ['Sim, apenas presencial', 'Sim, apenas semipresencial', 
       'Sim, parte presencial e parte semipresencial', 'Sim, EAD', 'Não'];
y24 = [I24_BR[0],I24_BR[1],I24_BR[2],I24_BR[3],I24_BR[4]];
ax24.bar(x24,y24);
plt.xticks(x24,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.savefig('QE_I24_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
x25 = ['Inserção no mercado de trabalho', 'Influência familiar','Valorização profissional',
       'Prestígio social', 'Vocação', 'Oferecido na modalidade EAD',
       'Baixa concorrência', 'Outro motivo'];
y25 = [I25_BR[0],I25_BR[1],I25_BR[2],I25_BR[3],I25_BR[4],I25_BR[5],I25_BR[6],I25_BR[7]];
ax25.bar(x25,y25);
plt.xticks(x25,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Porque escolhi o curso');
plt.savefig('QE_I25_BR.png', dpi=450, bbox_inches='tight');

#%%
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
x26 = ['Gratuidade', 'Preço da mensalidade', 'Prox. a residência', 'Prox. ao trabalho', 
       'Facilidade de acesso', 'Qualidade/reputação', 'Única opção de aprovação',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26 = [I26_BR[0],I26_BR[1],I26_BR[2],I26_BR[3],I26_BR[4],I26_BR[5],I26_BR[6],I26_BR[7],I26_BR[8]];
ax26.bar(x26,y26);
plt.xticks(x26,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Porque escolhi essa IES');
plt.savefig('QE_I26_BR.png', dpi=450, bbox_inches='tight');

