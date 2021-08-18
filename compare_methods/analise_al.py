# -*- coding: utf-8 -*-
"""
Criação: Abril/2021
    
Modificação: 

@author: Edvonaldo (edvonaldohoracio@gmail.com)

"""
import os
import csv
import math
import random
import pylab
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from scipy.stats import norm
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

#%% Gaussiana com matplotlib da distribuição anterior
print("Matplotlib version: ",matplotlib.__version__)
print("Pandas version", pd.__version__)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

mu_al, std_al = norm.fit(labels_al) # Média e desvio padrão dos dados

# Histograma
plt.hist(labels_al, bins=150, density=True, alpha=0.0)
  
# Limites
min_ylim, max_ylim = plt.ylim(0,0.06)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando (Gaussiana)
p_al = norm.pdf(x_al, mu_al, std_al)

# Plot Gaussiana
plt.plot(x_al, p_al, 'k', linewidth=1.5)# Ref: 
plt.fill_between(x_al, p_al, color='royalblue')# Ref: 
plt.axvline(labels_al.mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.text(labels_al.mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(labels_al.mean()))
plt.text(labels_al.mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(labels_al.std()))
plt.title("Distribuição de notas do Enade em Alagoas: 2014 a 2018")
plt.xlabel('Notas do Enade');
plt.ylabel('Distribuição');
plt.savefig('../tcc_codes/analise_stats/AL/imagens/DIST_NOTA_AL.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);

#%% Get nas notas de 
qe_i02 = dataset_al[["QE_I02", "NT_GER"]]
qe_i08 = dataset_al[["QE_I08", "NT_GER"]]
qe_i11 = dataset_al[["QE_I11", "NT_GER"]]
qe_i13 = dataset_al[["QE_I13", "NT_GER"]]
qe_i17 = dataset_al[["QE_I17", "NT_GER"]]
qe_i18 = dataset_al[["QE_I18", "NT_GER"]]
qe_i23 = dataset_al[["QE_I23", "NT_GER"]]

#%%
# Ref: 
size_title = 18
size_subtitle = 14
fig_i02, axes_i02 = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(20,10))
fig_i02.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\nDado socieconômico:Cor',
                 fontsize=size_title)


# Alternative A
qe_i02_aa = qe_i02.loc[(qe_i02['QE_I02'] == 'A')]

# Média e desvio padrão
mu_al_qei02_aa, std_al_qei02_aa = norm.fit(qe_i02_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_aa = norm.pdf(x_al, mu_al_qei02_aa, std_al_qei02_aa)

# Plot histogram
axes_i02[0,0].plot(x_al, p_al_qei02_aa, 'k', linewidth=1.5)

axes_i02[0,0].fill_between(x_al, p_al_qei02_aa, color='royalblue')
axes_i02[0,0].axvline(qe_i02_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[0,0].text(qe_i02_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_aa['NT_GER'].mean()))
axes_i02[0,0].text(qe_i02_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_aa['NT_GER'].std()))
axes_i02[0,0].set_title("A:Branca", fontsize=size_subtitle)

# Plot Gaussiana
qe_i02_bb = qe_i02.loc[(qe_i02['QE_I02'] == 'B')]

mu_al_qei02_bb, std_al_qei02_bb = norm.fit(qe_i02_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_bb = norm.pdf(x_al, mu_al_qei02_bb, std_al_qei02_bb)

# Plot histogram
axes_i02[0,1].plot(x_al, p_al_qei02_bb, 'k', linewidth=1.5)

axes_i02[0,1].fill_between(x_al, p_al_qei02_bb, color='royalblue')
axes_i02[0,1].axvline(qe_i02_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[0,1].text(qe_i02_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_bb['NT_GER'].mean()))
axes_i02[0,1].text(qe_i02_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_bb['NT_GER'].std()))
axes_i02[0,1].set_title("B:Preta", fontsize=size_subtitle)

# Alternative C
qe_i02_cc = qe_i02.loc[(qe_i02['QE_I02'] == 'C')]

mu_al_qei02_cc, std_al_qei02_cc = norm.fit(qe_i02_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_cc = norm.pdf(x_al, mu_al_qei02_cc, std_al_qei02_cc)

# Plot histogram
axes_i02[0,2].plot(x_al, p_al_qei02_cc, 'k', linewidth=1.5)

axes_i02[0,2].fill_between(x_al, p_al_qei02_cc, color='royalblue')
axes_i02[0,2].axvline(qe_i02_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[0,2].text(qe_i02_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_cc['NT_GER'].mean()))
axes_i02[0,2].text(qe_i02_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_cc['NT_GER'].std()))
axes_i02[0,2].set_title("C:Amarela", fontsize=size_subtitle)

# Alternative D
qe_i02_dd = qe_i02.loc[(qe_i02['QE_I02'] == 'D')]

mu_al_qei02_dd, std_al_qei02_dd = norm.fit(qe_i02_dd['NT_GER'])

#axes_i02[1][0].hist(qe_i02_dd['NT_GER'], bins=150, density=True, alpha=0.0)

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_dd = norm.pdf(x_al, mu_al_qei02_dd, std_al_qei02_dd)

# Plot histogram
axes_i02[1,0].plot(x_al, p_al_qei02_dd, 'k', linewidth=1.5)
axes_i02[1,0].fill_between(x_al, p_al_qei02_dd, color='royalblue')
axes_i02[1,0].axvline(qe_i02_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,0].text(qe_i02_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_dd['NT_GER'].mean()))
axes_i02[1,0].text(qe_i02_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_dd['NT_GER'].std()))
axes_i02[1,0].set_title("D:Parda", fontsize=size_subtitle)

# Alternative E
qe_i02_ee = qe_i02.loc[(qe_i02['QE_I02'] == 'E')]

mu_al_qei02_ee, std_al_qei02_ee = norm.fit(qe_i02_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_ee = norm.pdf(x_al, mu_al_qei02_ee, std_al_qei02_ee)

# Plot histogram
axes_i02[1,1].plot(x_al, p_al_qei02_ee, 'k', linewidth=1.5)

axes_i02[1,1].fill_between(x_al, p_al_qei02_ee, color='royalblue')
axes_i02[1,1].axvline(qe_i02_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,1].text(qe_i02_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ee['NT_GER'].mean()))
axes_i02[1,1].text(qe_i02_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ee['NT_GER'].std()))
axes_i02[1,1].set_title("E:Indígena", fontsize=size_subtitle)

# Alternative F
qe_i02_ff = qe_i02.loc[(qe_i02['QE_I02'] == 'F')]

mu_al_qei02_ff, std_al_qei02_ff = norm.fit(qe_i02_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_ff = norm.pdf(x_al, mu_al_qei02_ff, std_al_qei02_ff)

# Plot histogram
axes_i02[1,2].plot(x_al, p_al_qei02_ff, 'k', linewidth=1.5)

axes_i02[1,2].fill_between(x_al, p_al_qei02_ff, color='royalblue')
axes_i02[1,2].axvline(qe_i02_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,2].text(qe_i02_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ff['NT_GER'].mean()))
axes_i02[1,2].text(qe_i02_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ff['NT_GER'].std()))
axes_i02[1,2].set_title("F:Não quero declarar", fontsize=size_subtitle)

for ax in axes_i02.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i02.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I02_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% Subplots - Maior impacto
size_title = 18
size_subtitle = 14
fig_i08, axes_i08 = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=True, figsize=(20,10))
fig_i08.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\nDado socieconômico:Renda bruta',
                 fontsize=size_title)


# Alternative A
qe_i08_aa = qe_i08.loc[(qe_i08['QE_I08'] == 'A')]

# Média e desvio padrão
mu_al_qei08_aa, std_al_qei08_aa = norm.fit(qe_i08_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_aa = norm.pdf(x_al, mu_al_qei08_aa, std_al_qei08_aa)

# Plot histogram
axes_i08[0,0].plot(x_al, p_al_qei08_aa, 'k', linewidth=1.5)

axes_i08[0,0].fill_between(x_al, p_al_qei08_aa, color='royalblue')
axes_i08[0,0].axvline(qe_i08_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_aa['NT_GER'].mean()))
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_aa['NT_GER'].std()))
axes_i08[0,0].set_title("A:Até R$ 1.431,00", fontsize=size_subtitle)

# Plot Gaussiana
qe_i08_bb = qe_i08.loc[(qe_i08['QE_I08'] == 'B')]

mu_al_qei08_bb, std_al_qei08_bb = norm.fit(qe_i08_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_bb = norm.pdf(x_al, mu_al_qei08_bb, std_al_qei08_bb)

# Plot histogram
axes_i08[0,1].plot(x_al, p_al_qei08_bb, 'k', linewidth=1.5)

axes_i08[0,1].fill_between(x_al, p_al_qei08_bb, color='royalblue')
axes_i08[0,1].axvline(qe_i08_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,1].text(qe_i08_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_bb['NT_GER'].mean()))
axes_i08[0,1].text(qe_i08_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_bb['NT_GER'].std()))
axes_i08[0,1].set_title("B:De R\$ 1.431,01 a R\$ 2.862,00", fontsize=size_subtitle)

# Alternative C
qe_i08_cc = qe_i08.loc[(qe_i08['QE_I08'] == 'C')]

mu_al_qei08_cc, std_al_qei08_cc = norm.fit(qe_i08_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_cc = norm.pdf(x_al, mu_al_qei08_cc, std_al_qei08_cc)

# Plot histogram
axes_i08[0,2].plot(x_al, p_al_qei08_cc, 'k', linewidth=1.5)

axes_i08[0,2].fill_between(x_al, p_al_qei08_cc, color='royalblue')
axes_i08[0,2].axvline(qe_i08_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,2].text(qe_i08_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_cc['NT_GER'].mean()))
axes_i08[0,2].text(qe_i08_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_cc['NT_GER'].std()))
axes_i08[0,2].set_title("C:De R\$ 2.862,01 a R\$ 4.293,00", fontsize=size_subtitle)

# Alternative D
qe_i08_dd = qe_i08.loc[(qe_i08['QE_I08'] == 'D')]

mu_al_qei08_dd, std_al_qei08_dd = norm.fit(qe_i08_dd['NT_GER'])

#axes_i02[1][0].hist(qe_i02_dd['NT_GER'], bins=150, density=True, alpha=0.0)

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_dd = norm.pdf(x_al, mu_al_qei08_dd, std_al_qei08_dd)

# Plot histogram
axes_i08[1,0].plot(x_al, p_al_qei08_dd, 'k', linewidth=1.5)
axes_i08[1,0].fill_between(x_al, p_al_qei08_dd, color='royalblue')
axes_i08[1,0].axvline(qe_i08_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,0].text(qe_i08_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_dd['NT_GER'].mean()))
axes_i08[1,0].text(qe_i08_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_dd['NT_GER'].std()))
axes_i08[1,0].set_title("D:De R\$ 4.293,01 a R\$ 5.724,00", fontsize=size_subtitle)

# Alternative E
qe_i08_ee = qe_i08.loc[(qe_i08['QE_I08'] == 'E')]

mu_al_qei08_ee, std_al_qei08_ee = norm.fit(qe_i08_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_ee = norm.pdf(x_al, mu_al_qei08_ee, std_al_qei08_ee)

# Plot histogram
axes_i08[1,1].plot(x_al, p_al_qei08_ee, 'k', linewidth=1.5)

axes_i08[1,1].fill_between(x_al, p_al_qei08_ee, color='royalblue')
axes_i08[1,1].axvline(qe_i08_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,1].text(qe_i08_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_ee['NT_GER'].mean()))
axes_i08[1,1].text(qe_i08_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_ee['NT_GER'].std()))
axes_i08[1,1].set_title("E:De R\$ 5.274,01 a R\$ 9.540,00", fontsize=size_subtitle)

# Alternative F
qe_i08_ff = qe_i08.loc[(qe_i08['QE_I08'] == 'F')]

mu_al_qei08_ff, std_al_qei08_ff = norm.fit(qe_i08_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_ff = norm.pdf(x_al, mu_al_qei08_ff, std_al_qei08_ff)

# Plot histogram
axes_i08[1,2].plot(x_al, p_al_qei08_ff, 'k', linewidth=1.5)

axes_i08[1,2].fill_between(x_al, p_al_qei08_ff, color='royalblue')
axes_i08[1,2].axvline(qe_i08_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,2].text(qe_i08_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_ff['NT_GER'].mean()))
axes_i08[1,2].text(qe_i08_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_ff['NT_GER'].std()))
axes_i08[1,2].set_title("F:De R\$ 9.540,01 a R\$ 28.620,00", fontsize=size_subtitle)

# Alternative G
qe_i08_gg = qe_i08.loc[(qe_i08['QE_I08'] == 'G')]

mu_al_qei08_gg, std_al_qei08_gg = norm.fit(qe_i08_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_gg = norm.pdf(x_al, mu_al_qei08_gg, std_al_qei08_gg)

# Plot histogram
axes_i08[2,0].plot(x_al, p_al_qei08_gg, 'k', linewidth=1.5)

axes_i08[2,0].fill_between(x_al, p_al_qei08_gg, color='royalblue')
axes_i08[2,0].axvline(qe_i08_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[2,0].text(qe_i08_gg['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_gg['NT_GER'].mean()))
axes_i08[2,0].text(qe_i08_gg['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_gg['NT_GER'].std()))
axes_i08[2,0].set_title("G:Mais de R$ 28.620,00", fontsize=size_subtitle)

axes_i08[2,1].axis('off')
axes_i08[2,2].axis('off')

for ax in axes_i08.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i08.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I08_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
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

#%%
'''
# All references:
    1) https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/
    2) https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
    3) https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#examples-using-matplotlib-pyplot-plot
    4) https://moonbooks.org/Articles/How-to-fill-an-area-in-matplotlib-/
    5) https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
    6) https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/
    7) https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
    8) https://stackoverflow.com/questions/10035446/how-can-i-make-a-blank-subplot-in-matplotlib
'''