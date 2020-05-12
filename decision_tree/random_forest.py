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

features_al.iloc[:,0:16] = features_al.iloc[:,0:16].replace(to_replace = 0, value = 1)
#%% Observando os dados
print('O formato dos dados é: ', features_al.shape)

describe_al = features_al.describe()

print('Descrição para as colunas: ', describe_al)
print(describe_al.columns)

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
from sklearn.ensemble import RandomForestRegressor

scores_al = []

importance_fields_al = 0.0
importance_fields_aux_al = []

rf_al = RandomForestRegressor(n_estimators = 500, random_state=0)

kf_cv_al = KFold(n_splits=11, random_state=None, shuffle=False) # n_splits: divisores de 7084 ^ memory

for train_index_al, test_index_al in kf_cv_al.split(features_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = features_al[train_index_al]
    test_features_al = features_al[test_index_al]
    train_labels_al = labels_al[train_index_al]
    test_labels_al = labels_al[test_index_al]
    
    # Ajustando cada features e label com RF
    rf_al.fit(train_features_al, train_labels_al)
    
    # Usando o Random Forest para predição dos dados
    predictions_al = rf_al.predict(test_features_al)
    
    # Erro
    errors_al = abs(predictions_al - test_labels_al)
    
    # Acurácia
    accuracy_al = 100 - mean_absolute_error(test_labels_al, predictions_al)
    
    # Importância das variáveis
    importance_fields_aux_al = rf_al.feature_importances_
    importance_fields_al += importance_fields_aux_al
    
    # Append em cada valor médio
    scores_al.append(accuracy_al)

#%% Acurácia AL
print('Accuracy: ', round(np.average(scores_al), 2), "%.")

importance_fields_al_t = importance_fields_al/11

print('Total: ', round(np.sum(importance_fields_al_t),8))

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



#%% Visualization of Variable Importances
# QE_I01
fig1 = plt.figure();
ax1 = fig1.add_axes([0,0,1,1]);
x1 = ['Solteiro', 'Casado (a)', 'Separado', 'Viúvo', 'Outro'];
y1 = [I01_AL[0],I01_AL[1],I01_AL[2],I01_AL[3],I01_AL[4]];
ax1.bar(x1,y1);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Estado civil');
plt.savefig('QE_I01_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I02
fig2 = plt.figure();
ax2 = fig2.add_axes([0,0,1,1]);
x2 = ['Branca','Preta','Amarela','Parda','Indígena','Não quero declarar'];
y2 =[I02_AL[0],I02_AL[1],I03_AL[2],I02_AL[3],I02_AL[4],I02_AL[5]];
ax2.bar(x2,y2);
plt.xticks(x2,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Cor/raça');
plt.savefig('QE_I02_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I03
fig3 = plt.figure();
ax3 = fig3.add_axes([0,0,1,1]);
x3 = ['Brasileira','Brasileira naturalizada','Estrangeira'];
y3 = [I03_AL[0],I03_AL[1],I03_AL[2]];
ax3.bar(x3,y3);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Nacionalidade');
plt.savefig('QE_I03_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I04
fig4 = plt.figure();
ax4 = fig4.add_axes([0,0,1,1]);
x4 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y4 = [I04_AL[0],I04_AL[1],I04_AL[2],I04_AL[3],I04_AL[4],I04_AL[5]];
ax4.bar(x4,y4);
plt.xticks(x4,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Escolarização da pai');
plt.savefig('QE_I04_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I05
fig5 = plt.figure();
ax5 = fig5.add_axes([0,0,1,1]);
x5 = ['Nenhum','1º ao 5º ano','6º ao 9º ano','Ensino médio','Graduação','Pós-graduação'];
y5 = [I05_AL[0],I05_AL[1],I05_AL[2],I05_AL[3],I05_AL[4],I05_AL[5]];
ax5.bar(x5,y5);
plt.xticks(x5,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Escolarização da mãe');
plt.savefig('QE_I05_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I06
fig6 = plt.figure();
ax6 = fig6.add_axes([0,0,1,1]);
x6 = ['Casa/apartamento (sozinho)','Casa/apartamento (pais/parentes)',
      'Casa/apartamento (cônjugue/filhos)','Casa/apartamento (outras pessoas)',
      'Alojamento univ. na própria IES','Outro'];
y6 = [I06_AL[0],I06_AL[1],I06_AL[2],I06_AL[3],I06_AL[4],I06_AL[5]];
ax6.bar(x6,y6);
plt.xticks(x6,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Onde e com quem moro');
plt.savefig('QE_I06_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I07
fig7 = plt.figure();
ax7 = fig7.add_axes([0,0,1,1]);
x7 = ['Nenhuma','Uma','Duas','Três','Quatro','Cinco','Seis','Sete ou mais'];
y7 = [I07_AL[0],I07_AL[1],I07_AL[2],I07_AL[3],I07_AL[4],I07_AL[5],I07_AL[6],I07_AL[7]];
ax7.bar(x7,y7);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Quantos moram com o estudante');
plt.savefig('QE_I07_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I08
fig8 = plt.figure();
ax8 = fig8.add_axes([0,0,1,1]);
x8 = ['Até 1,5 sál. mín','De 1 a 3 sál. mín.','De 3 a 4,5 sál. mín.',
      'De 4,5 a 6 sál. mín','De 6 a 10 sál. mín.','De 30 a 10 sál. mín',
      'Acima de 30 sál. mín.'];
y8 = [I08_AL[0],I08_AL[1],I08_AL[2],I08_AL[3],I08_AL[4],I08_AL[5],I08_AL[6]];
ax8.bar(x8,y8);
plt.xticks(x8,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Renda total');
plt.savefig('QE_I08_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I09
fig9 = plt.figure();
ax9 = fig9.add_axes([0,0,1,1]);
x9 = ['Sem renda (financiamento governamental)','Sem renda (financ. por família/outros)',
      'Tenho renda, mas recebo ajuda (família/outras pessoas)',
      'Tenho renda (autossuficiente)','Tenho renda e ajudo a família',
      'Sou o principal a ajudar a família'];
y9 = [I09_AL[0],I09_AL[1],I09_AL[2],I09_AL[3],I09_AL[4],I09_AL[5]];
ax9.bar(x9,y9);
plt.xticks(x9,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Situação financeira');
plt.savefig('QE_I09_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I10
fig10 = plt.figure();
ax10 = fig10.add_axes([0,0,1,1]);
x10 = ['Não estou trabalhando','Trabalho eventualmente','Trablho (até 20h/sem)',
       'Trabalho (de 21h/sem a 39h/sem)','Trabalho 40h/sem ou mais'];
y10 = [I10_AL[0],I10_AL[1],I10_AL[2],I10_AL[3],I10_AL[4]];
ax10.bar(x10,y10);
plt.xticks(x10,rotation=90,fontsize=8)
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Situação de trabalho');
plt.savefig('QE_I10_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I11
fig11 = plt.figure();
ax11 = fig11.add_axes([0,0,1,1]);
x11 = ['Nenhum (curso gratuito)','Nenhum (mas não gratuito)','ProUni integral',
       'ProUni parcial, apenas','FIES, apenas','ProUni parcial e FIES',
       'Bolsa do governo (estadual/distrital/municipal)',
       'Bolsa pela IES','Bolsa por outra entidade','Financiamento pela IES',
       'Financiamento bancário'];
y11 = [I11_AL[0],I11_AL[1],I11_AL[2],I11_AL[3],I11_AL[4],I11_AL[5],I11_AL[6],I11_AL[7],I11_AL[8],I11_AL[9],I11_AL[10]];
ax11.bar(x11,y11);
plt.xticks(x11,rotation=90,fontsize=8)
#plt.legend(x11,loc=2)
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Bolsa ou financiamento para custeio de mensalidade');
plt.savefig('QE_I11_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I12
fig12 = plt.figure();
ax12 = fig12.add_axes([0,0,1,1]);
x12 = ['Nenhum','Moradia','Alimentação','Moradia e alimentação', 'Permanência','Outros'];
y12 = [I12_AL[0],I12_AL[1],I12_AL[2],I12_AL[3],I12_AL[4],I12_AL[5]];
ax12.bar(x12,y12);
plt.xticks(x12,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Auxílio permanência');
plt.savefig('QE_I12_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I13
fig13 = plt.figure();
ax13 = fig13.add_axes([0,0,1,1]);
x13 = ['Nenhum', 'Bolsa IC', 'Bolsa extensão','Bolsa monitoria/tutoria',
       'Bolsa PET','Outro tipo'];
y13 = [I13_AL[0],I13_AL[1],I13_AL[2],I13_AL[3],I13_AL[4],I13_AL[5]];
ax13.bar(x13,y13);
plt.xticks(x13,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Tipo de bolsa recebido');
plt.savefig('QE_I13_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I14
fig14 = plt.figure();
ax14 = fig14.add_axes([0,0,1,1]);
x14 = ['Não','Sim, Ciências sem Fronteiras', 'Sim, intercâmbio pelo Governo Federal',
       'Sim, intercâmbio pelo Governo Estadual', 'Sim, intercâmbio pela minha IES',
       'Sim, intercâmbio não institucional'];
y14 = [I14_AL[0],I14_AL[1],I14_AL[2],I14_AL[3],I14_AL[4],I14_AL[5]];
ax14.bar(x14,y14);
plt.xticks(x14,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Programas de atividade no exterior');
plt.savefig('QE_I14_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I15
fig15 = plt.figure();
ax15 = fig15.add_axes([0,0,1,1])
x15 = ['Não','Sim, étnico-racial','Sim, renda', 'Sim, escola pública ou particular (com bolsa)',
       'Sim, combina dois mais', 'Sim, outra'];
y15 = [I15_AL[0],I15_AL[1],I15_AL[2],I15_AL[3],I15_AL[4],I15_AL[5]];
ax15.bar(x15,y15);
plt.xticks(x15,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Ingresso por cota');
plt.savefig('QE_I15_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I16
fig16 = plt.figure();
ax16 = fig16.add_axes([0,0,1,1]);
x16 = ['AL'];
y16 = [I16_AL[0]];
ax16.bar(x16,y16);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('UF que concluiu o médio');
plt.savefig('QE_I16_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I17
fig17 = plt.figure();
ax17 = fig17.add_axes([0,0,1,1]);
x17 = ['Todo em escola pública', 'Todo em escola privada','Todo no exterior',
       'Maior parte em escola pública','Maior parte em escola privada',
       'Parte no Brasil e parte no exterior'];
y17 = [I17_AL[0],I17_AL[1],I17_AL[2],I17_AL[3],I17_AL[4],I17_AL[5]];
ax17.bar(x17,y17);
plt.xticks(x17,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Tipo de escola no médio');
plt.savefig('QE_I17_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I18
fig18 = plt.figure();
ax18 = fig18.add_axes([0,0,1,1]);
x18 = ['Tradicional', 'Prof. técnico', 'Prof. magistério (curso normal)', 
       'EJA e/ou Supletivo', 'Outra'];
y18 = [I18_AL[0],I18_AL[1],I18_AL[2],I18_AL[3],I18_AL[4]];
ax18.bar(x18,y18);
plt.xticks(x18,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Modalidade do Ensino Médio');
plt.savefig('QE_I18_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I19
fig19 = plt.figure();
ax19 = fig19.add_axes([0,0,1,1]);
x19 = ['Ninguém', 'Pais', 'Outros membros (excluindo os pais)', 'Professores', 
       'Líder ou representante religioso', 'Colegas/amigos', 'Outras pessoas']
y19 = [I19_AL[0],I19_AL[1],I19_AL[2],I19_AL[3],I19_AL[4],I19_AL[5],I19_AL[6]];
ax19.bar(x19,y19);
plt.xticks(x19,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Maior incentivo para cursar a graduação');
plt.savefig('QE_I20_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I20
fig20 = plt.figure();
ax20 = fig20.add_axes([0,0,1,1]);
x20 = ['Não tive dificuldade', 'Não recebi apoio', 'Pais', 'Avós', 'Irmãos, primos ou tios',
       'Líder ou representante religioso', 'Colegas de curso ou amigos',
       'Professores do curso', 'Profissionais do serviço de apoio da IES',
       'Colegas de trabalho', 'Outro grupo'];
y20 = [I20_AL[0],I20_AL[1],I20_AL[2],I20_AL[3],I20_AL[4],I20_AL[5],I20_AL[6],I20_AL[7],I20_AL[8],I20_AL[9],I20_AL[10]];
ax20.bar(x20,y20);
plt.xticks(x20,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Grupo determinante para enfrentar as dificuldades do curso e concluí-lo');
plt.savefig('QE_I20_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I21
fig21 = plt.figure();
ax21 = fig21.add_axes([0,0,1,1]);
x21 = ['Sim', 'Não'];
y21 = [I21_AL[0],I21_AL[1]];
ax21.bar(x21,y21);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Alguém da família concluiu curso superior');
plt.savefig('QE_I21_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I22
fig22 = plt.figure();
ax22 = fig22.add_axes([0,0,1,1]);
x22 = ['Nenhum  ', 'Um ou dois', 'Três a cinco', 'Seis a oito', 'Mais de oito'];
y22 = [I22_AL[0],I22_AL[1],I22_AL[2],I22_AL[3],I22_AL[4]];
ax22.bar(x22,y22);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Livros lido no ano (excluindo da Biografia do curso');
plt.savefig('QE_I22_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I23
fig23 = plt.figure();
ax23 = fig23.add_axes([0,0,1,1]);
x23 = ['Nenhuma', 'De uma a três', 'De quatro a sete', 'De oito a doze', 'Mais de doze'];
y23 = [I23_AL[0],I23_AL[1],I23_AL[2],I23_AL[3],I23_AL[4]];
ax23.bar(x23,y23);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Horas de estudo por semana (excluindo aulas)');
plt.savefig('QE_I23_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I24
fig24 = plt.figure();
ax24 = fig24.add_axes([0,0,1,1]);
x24 = ['Sim, apenas presencial', 'Sim, apenas semipresencial', 
       'Sim, parte presencial e parte semipresencial', 'Sim, EAD', 'Não'];
y24 = [I24_AL[0],I24_AL[1],I24_AL[2],I24_AL[3],I24_AL[4]];
ax24.bar(x24,y24);
plt.xticks(x24,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Oportunidade de aprendizado de idioma estrangeiro');
plt.savefig('QE_I24_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I25
fig25 = plt.figure();
ax25 = fig25.add_axes([0,0,1,1]);
x25 = ['Inserção no mercado de trabalho', 'Influência familiar','Valorização profissional',
       'Prestígio social', 'Vocação', 'Oferecido na modalidade EAD',
       'Baixa concorrência', 'Outro motivo'];
y25 = [I25_AL[0],I25_AL[1],I25_AL[2],I25_AL[3],I25_AL[4],I25_AL[5],I25_AL[6],I25_AL[7]];
ax25.bar(x25,y25);
plt.xticks(x25,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Porque escolhi o curso');
plt.savefig('QE_I25_AL.png', dpi=450, bbox_inches='tight');

#%%
# QE_I26
fig26 = plt.figure();
ax26 = fig26.add_axes([0,0,1,1]);
x26 = ['Gratuidade', 'Preço da mensalidade', 'Prox. a residência', 'Prox. ao trabalho', 
       'Facilidade de acesso', 'Qualidade/reputação', 'Única opção de aprovação',
       'Possibilidade de bolsa de estudo', 'Outro motivo'];
y26 = [I26_AL[0],I26_AL[1],I26_AL[2],I26_AL[3],I26_AL[4],I26_AL[5],I26_AL[6],I26_AL[7],I26_AL[8]];
ax26.bar(x26,y26);
plt.xticks(x26,rotation=90,fontsize=8);
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Porque escolhi essa IES');
plt.savefig('QE_I26_AL.png', dpi=450, bbox_inches='tight');
#%%
# list of x locations for plotting
x_values = list(range(len(importance_fields_al_t)));

# Make a bar chart
plt.bar(x_values, importance_fields_al_t, orientation = 'vertical');

# Tick labels for x axis
plt.xticks(x_values, features_al_list_oh, rotation='vertical');

# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable'); 
plt.title('Variable Importances');
plt.savefig('VI_AL.png', dpi=450, bbox_inches='tight')