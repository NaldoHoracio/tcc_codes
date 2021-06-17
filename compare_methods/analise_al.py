# -*- coding: utf-8 -*-
"""
Criação:
    
Modificação:

@author: Edvonaldo (edvonaldohoracio@gmail.com)
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

#%% Percentual de valores NaN

nan_2014 = float(data_al2014['NT_GER'].isnull().sum());
nan_2015 = float(data_al2015['NT_GER'].isnull().sum());
nan_2016 = float(data_al2016['NT_GER'].isnull().sum());
nan_2017 = float(data_al2017['NT_GER'].isnull().sum());
nan_2018 = float(data_al2018['NT_GER'].isnull().sum());

column_2014 = float(data_al2014.shape[0]);
column_2015 = float(data_al2015.shape[0]);
column_2016 = float(data_al2016.shape[0]);
column_2017 = float(data_al2017.shape[0]);
column_2018 = float(data_al2018.shape[0]);

per_2014 = nan_2014/column_2014;
per_2015 = nan_2015/column_2015;
per_2016 = nan_2016/column_2016;
per_2017 = nan_2017/column_2017;
per_2018 = nan_2018/column_2018;


print("Qtde. % NaN values in NT_GER 2014:", 100*round(per_2014, 4));
print("Qtde. % NaN values in NT_GER 2015:", 100*round(per_2015, 4));
print("Qtde. % NaN values in NT_GER 2016:", 100*round(per_2016, 4));
print("Qtde. % NaN values in NT_GER 2017:", 100*round(per_2017, 4));
print("Qtde. % NaN values in NT_GER 2018:", 100*round(per_2018, 4));

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
    
    dataset_al = data_al
    
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
    
    return features_al, labels_al, features_al_list_oh, dataset_al

#%% Aplicando o pré-processamento

features_al, labels_al, features_al_list_oh, dataset_al = processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018)

#%% Percentual de valores NaN

nan_al = float(nan_2014+nan_2015+nan_2016+nan_2017+nan_2018);

column_al = float(labels_al.shape[0]);

per_al = nan_al/column_al;

print("Qtde. % NaN values in NT_GER AL:", 100*round(per_al, 4));
#%% Dados estatísticos gerais
# < 20 --> Amostra
# >= 20 --> população
import statistics as stats

min_al = min(labels_al)
max_al = max(labels_al)
mean_al = stats.mean(labels_al)
median_al = stats.median(labels_al)
variance_al = stats.variance(labels_al)
std_dev_al = stats.stdev(labels_al)

print("Min:", round(min_al, 4))
print("Max:", round(max_al, 4))
print("Media:", round(mean_al, 4))
print("Mediana:", round(median_al, 4))
print("Variancia:", round(variance_al, 4))
print("Desvio padrao: ", round(std_dev_al, 4))

#%% Escrevendo em arquivo

fields_stats_al = ['Version',
                   'Media',
                   'Mediana',
                   'Variancia',
                   'Desvio padrao',
                   'Max val',
                   'Min val',
                   '% Nan val']

rows_stats_al = {'Version':0,
                 'Media':mean_al,
                 'Mediana':median_al,
                 'Variancia':variance_al,
                 'Desvio padrao':std_dev_al,
                 'Max val':max_al,
                 'Min val':min_al,
                 '% Nan val': 100*per_al}

file_stats_al = "../tcc_codes/analise_stats/AL/Stats_AL.csv"

version_file(file_stats_al, fields_stats_al, rows_stats_al)

#%% Plotando gráfico de distribuição das notas em Alagoas
import seaborn as sns

sns.distplot(labels_al, kde=True)
plt.title("Distribuição de notas do Enade de 2014 a 2018: Alagoas");
plt.xlabel('Notas do Enade');
plt.ylabel('Distribuição');
plt.legend();
# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/DIST_NOTA_AL.png', dpi=450, bbox_inches='tight', pad_inches=0.015);

#%% Subplots - Maior impacto
# QE_02
#df_qe_02 = 

# QE_08

# QE_11

# QE_13

# QE_17

# QE_18

# QE_23

#%% Subplots - Menor impacto
# QE_01

# QE_03

# QE_12

# QE_15

# QE_16

# QE_18

# QE_19

# QE_21