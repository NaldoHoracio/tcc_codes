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
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/DIST_NOTA_AL.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);

#%% Get nas notas de Maior impacto
qe_i02 = dataset_al[["QE_I02", "NT_GER"]]
qe_i08 = dataset_al[["QE_I08", "NT_GER"]]
qe_i11 = dataset_al[["QE_I11", "NT_GER"]]
qe_i13 = dataset_al[["QE_I13", "NT_GER"]]
qe_i17 = dataset_al[["QE_I17", "NT_GER"]]
qe_i18 = dataset_al[["QE_I18", "NT_GER"]]
qe_i23 = dataset_al[["QE_I23", "NT_GER"]]
##### Get nas notas de menor impacto #####
qe_i01 = dataset_al[["QE_I01", "NT_GER"]]
qe_i03 = dataset_al[["QE_I03", "NT_GER"]]
qe_i12 = dataset_al[["QE_I12", "NT_GER"]]
qe_i15 = dataset_al[["QE_I15", "NT_GER"]]
qe_i16 = dataset_al[["QE_I16", "NT_GER"]]
qe_i19 = dataset_al[["QE_I19", "NT_GER"]]
qe_i21 = dataset_al[["QE_I21", "NT_GER"]]

#%% QE_I02
size_title = 18
size_subtitle = 14
fig_i02, axes_i02 = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(12,10))
fig_i02.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'
                 'Dado socieconômico:Cor',
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
axes_i02[0,0].set_xlim([0,100])
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
axes_i02[0,1].set_xlim([0,100])
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
axes_i02[1,0].plot(x_al, p_al_qei02_cc, 'k', linewidth=1.5)
axes_i02[1,0].set_xlim([0,100])
axes_i02[1,0].fill_between(x_al, p_al_qei02_cc, color='royalblue')
axes_i02[1,0].axvline(qe_i02_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,0].text(qe_i02_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_cc['NT_GER'].mean()))
axes_i02[1,0].text(qe_i02_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_cc['NT_GER'].std()))
axes_i02[1,0].set_title("C:Amarela", fontsize=size_subtitle)

####### Alternative D #######
qe_i02_dd = qe_i02.loc[(qe_i02['QE_I02'] == 'D')]

mu_al_qei02_dd, std_al_qei02_dd = norm.fit(qe_i02_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei02_dd = norm.pdf(x_al, mu_al_qei02_dd, std_al_qei02_dd)

# Plot histogram
axes_i02[1,1].plot(x_al, p_al_qei02_dd, 'k', linewidth=1.5)
axes_i02[1,1].set_xlim([0,100])
axes_i02[1,1].fill_between(x_al, p_al_qei02_dd, color='royalblue')
axes_i02[1,1].axvline(qe_i02_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,1].text(qe_i02_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_dd['NT_GER'].mean()))
axes_i02[1,1].text(qe_i02_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_dd['NT_GER'].std()))
axes_i02[1,1].set_title("D:Parda", fontsize=size_subtitle)

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
axes_i02[2,0].plot(x_al, p_al_qei02_ee, 'k', linewidth=1.5)
axes_i02[2,0].set_xlim([0,100])
axes_i02[2,0].fill_between(x_al, p_al_qei02_ee, color='royalblue')
axes_i02[2,0].axvline(qe_i02_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[2,0].text(qe_i02_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ee['NT_GER'].mean()))
axes_i02[2,0].text(qe_i02_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ee['NT_GER'].std()))
axes_i02[2,0].set_title("E:Indígena", fontsize=size_subtitle)

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
axes_i02[2,1].plot(x_al, p_al_qei02_ff, 'k', linewidth=1.5)
axes_i02[2,1].set_xlim([0,100])
axes_i02[2,1].fill_between(x_al, p_al_qei02_ff, color='royalblue')
axes_i02[2,1].axvline(qe_i02_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[2,1].text(qe_i02_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ff['NT_GER'].mean()))
axes_i02[2,1].text(qe_i02_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ff['NT_GER'].std()))
axes_i02[2,1].set_title("F:Não quero declarar", fontsize=size_subtitle)


for ax in axes_i02.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')
    ax.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/QE_I02_AL_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I08
fig_i08, axes_i08 = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=True, figsize=(12,10))
fig_i08.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'
                 'Dado socieconômico:Renda bruta',
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
axes_i08[0,0].set_xlim([0,100])
axes_i08[0,0].fill_between(x_al, p_al_qei08_aa, color='royalblue')
axes_i08[0,0].axvline(qe_i08_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_aa['NT_GER'].mean()))
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_aa['NT_GER'].std()))
axes_i08[0,0].set_title("A:Até R$ 1.431,00", fontsize=size_subtitle)

# Alternative B
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
axes_i08[0,1].set_xlim([0,100])
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
axes_i08[0,2].set_xlim([0,100])
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

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei08_dd = norm.pdf(x_al, mu_al_qei08_dd, std_al_qei08_dd)

# Plot histogram
axes_i08[1,0].plot(x_al, p_al_qei08_dd, 'k', linewidth=1.5)
axes_i08[1,0].set_xlim([0,100])
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
axes_i08[1,1].set_xlim([0,100])
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
axes_i08[1,2].set_xlim([0,100])
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
axes_i08[2,0].set_xlim([0,100])
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
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/QE_I08_AL_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I11A
size_title = 18
size_subtitle = 14
fig_i11a, axes_i11a = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i11a.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socieconômico:Tipo de financiamento p/ custeio das mensalidades',
                 fontsize=size_title)


####### Alternative A #######
qe_i11a_aa = qe_i11.loc[(qe_i11['QE_I11'] == 'A')]

# Média e desvio padrão
mu_al_qei11a_aa, std_al_qei11a_aa = norm.fit(qe_i11a_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_aa = norm.pdf(x_al, mu_al_qei11a_aa, std_al_qei11a_aa)

# Plot histogram
axes_i11a[0,0].plot(x_al, p_al_qei11a_aa, 'k', linewidth=1.5)
axes_i11a[0,0].set_xlim([0,100])
axes_i11a[0,0].fill_between(x_al, p_al_qei11a_aa, color='royalblue')
axes_i11a[0,0].axvline(qe_i11a_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[0,0].text(qe_i11a_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_aa['NT_GER'].mean()))
axes_i11a[0,0].text(qe_i11a_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_aa['NT_GER'].std()))
axes_i11a[0,0].set_title("A:Nenhum;curso gratuito", fontsize=size_subtitle)

####### Alternative B #######
qe_i11a_bb = qe_i11.loc[(qe_i11['QE_I11'] == 'B')]

mu_al_qei11a_bb, std_al_qei11a_bb = norm.fit(qe_i11a_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_bb = norm.pdf(x_al, mu_al_qei11a_bb, std_al_qei11a_bb)

# Plot histogram
axes_i11a[0,1].plot(x_al, p_al_qei11a_bb, 'k', linewidth=1.5)
axes_i11a[0,1].set_xlim([0,100])
axes_i11a[0,1].fill_between(x_al, p_al_qei11a_bb, color='royalblue')
axes_i11a[0,1].axvline(qe_i11a_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[0,1].text(qe_i11a_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_bb['NT_GER'].mean()))
axes_i11a[0,1].text(qe_i11a_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_bb['NT_GER'].std()))
axes_i11a[0,1].set_title("B:Nenhum;curso pago", fontsize=size_subtitle)

####### Alternative C #######
qe_i11a_cc = qe_i11.loc[(qe_i11['QE_I11'] == 'C')]

mu_al_qei11a_cc, std_al_qei11a_cc = norm.fit(qe_i11a_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_cc = norm.pdf(x_al, mu_al_qei11a_cc, std_al_qei11a_cc)

# Plot histogram
axes_i11a[1,0].plot(x_al, p_al_qei11a_cc, 'k', linewidth=1.5)
axes_i11a[1,0].set_xlim([0,100])
axes_i11a[1,0].fill_between(x_al, p_al_qei11a_cc, color='royalblue')
axes_i11a[1,0].axvline(qe_i11a_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[1,0].text(qe_i11a_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_cc['NT_GER'].mean()))
axes_i11a[1,0].text(qe_i11a_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_cc['NT_GER'].std()))
axes_i11a[1,0].set_title("C:ProUni integral", fontsize=size_subtitle)

####### Alternative D #######
qe_i11a_dd = qe_i11.loc[(qe_i11['QE_I11'] == 'D')]

mu_al_qei11a_dd, std_al_qei11a_dd = norm.fit(qe_i11a_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_dd = norm.pdf(x_al, mu_al_qei11a_dd, std_al_qei11a_dd)

# Plot histogram
axes_i11a[1,1].plot(x_al, p_al_qei11a_dd, 'k', linewidth=1.5)
axes_i11a[1,1].set_xlim([0,100])
axes_i11a[1,1].fill_between(x_al, p_al_qei11a_dd, color='royalblue')
axes_i11a[1,1].axvline(qe_i11a_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[1,1].text(qe_i11a_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_dd['NT_GER'].mean()))
axes_i11a[1,1].text(qe_i11a_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_dd['NT_GER'].std()))
axes_i11a[1,1].set_title("D:ProUni parcial,apenas", fontsize=size_subtitle)

####### Alternative E #######
qe_i11a_ee = qe_i11.loc[(qe_i11['QE_I11'] == 'E')]

mu_al_qei11a_ee, std_al_qei11a_ee = norm.fit(qe_i11a_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_ee = norm.pdf(x_al, mu_al_qei11a_ee, std_al_qei11a_ee)

# Plot histogram
axes_i11a[2,0].plot(x_al, p_al_qei11a_ee, 'k', linewidth=1.5)
axes_i11a[2,0].set_xlim([0,100])
axes_i11a[2,0].fill_between(x_al, p_al_qei11a_ee, color='royalblue')
axes_i11a[2,0].axvline(qe_i11a_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[2,0].text(qe_i11a_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_ee['NT_GER'].mean()))
axes_i11a[2,0].text(qe_i11a_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_ee['NT_GER'].std()))
axes_i11a[2,0].set_title("E:FIES,apenas", fontsize=size_subtitle)

####### Alternative F #######
qe_i11a_ff = qe_i11.loc[(qe_i11['QE_I11'] == 'F')]

mu_al_qei11a_ff, std_al_qei11a_ff = norm.fit(qe_i11a_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11a_ff = norm.pdf(x_al, mu_al_qei11a_ff, std_al_qei11a_ff)

# Plot histogram
axes_i11a[2,1].plot(x_al, p_al_qei11a_ff, 'k', linewidth=1.5)
axes_i11a[2,1].set_xlim([0,100])
axes_i11a[2,1].fill_between(x_al, p_al_qei11a_ff, color='royalblue')
axes_i11a[2,1].axvline(qe_i11a_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[2,1].text(qe_i11a_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_ff['NT_GER'].mean()))
axes_i11a[2,1].text(qe_i11a_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_ff['NT_GER'].std()))
axes_i11a[2,1].set_title("F:ProUni parcial e FIES", fontsize=size_subtitle)

for ax in axes_i11a.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i11a.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I11A_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I11B
fig_i11b, axes_i11b = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i11b.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'
                  'Dado socieconômico:Tipo de financiamento para custeio das mensalidades',
                 fontsize=size_title)
####### Alternative G #######
qe_i11b_gg = qe_i11.loc[(qe_i11['QE_I11'] == 'G')]

# Média e desvio padrão
mu_al_qei11b_gg, std_al_qei11b_gg = norm.fit(qe_i11b_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11b_gg = norm.pdf(x_al, mu_al_qei11b_gg, std_al_qei11b_gg)

# Plot histogram
axes_i11b[0,0].plot(x_al, p_al_qei11b_gg, 'k', linewidth=1.5)
axes_i11b[0,0].set_xlim([0,100])
axes_i11b[0,0].fill_between(x_al, p_al_qei11b_gg, color='royalblue')
axes_i11b[0,0].axvline(qe_i11b_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[0,0].text(qe_i11b_gg['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_gg['NT_GER'].mean()))
axes_i11b[0,0].text(qe_i11b_gg['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_gg['NT_GER'].std()))
axes_i11b[0,0].set_title("G:Bolsa pelo estado,governo ou município", fontsize=size_subtitle)

####### Alternative H #######
qe_i11b_hh = qe_i11.loc[(qe_i11['QE_I11'] == 'H')]

mu_al_qei11b_hh, std_al_qei11b_hh = norm.fit(qe_i11b_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11b_hh = norm.pdf(x_al, mu_al_qei11b_hh, std_al_qei11b_hh)

# Plot histogram
axes_i11b[0,1].plot(x_al, p_al_qei11b_hh, 'k', linewidth=1.5)
axes_i11b[0,1].set_xlim([0,100])
axes_i11b[0,1].fill_between(x_al, p_al_qei11b_hh, color='royalblue')
axes_i11b[0,1].axvline(qe_i11b_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[0,1].text(qe_i11b_hh['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_hh['NT_GER'].mean()))
axes_i11b[0,1].text(qe_i11b_hh['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_hh['NT_GER'].std()))
axes_i11b[0,1].set_title("H:Bolsa pela IES", fontsize=size_subtitle)

####### Alternative I #######
qe_i11b_ii = qe_i11.loc[(qe_i11['QE_I11'] == 'I')]

mu_al_qei11b_ii, std_al_qei11b_ii = norm.fit(qe_i11b_ii['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11b_ii = norm.pdf(x_al, mu_al_qei11b_ii, std_al_qei11b_ii)

# Plot histogram
axes_i11b[1,0].plot(x_al, p_al_qei11b_ii, 'k', linewidth=1.5)
axes_i11b[1,0].set_xlim([0,100])
axes_i11b[1,0].fill_between(x_al, p_al_qei11b_ii, color='royalblue')
axes_i11b[1,0].axvline(qe_i11b_ii['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[1,0].text(qe_i11b_ii['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_ii['NT_GER'].mean()))
axes_i11b[1,0].text(qe_i11b_ii['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_ii['NT_GER'].std()))
axes_i11b[1,0].set_title("I:Bolsa por outra entidade", fontsize=size_subtitle)

####### Alternative J #######
qe_i11b_jj = qe_i11.loc[(qe_i11['QE_I11'] == 'J')]

mu_al_qei11b_jj, std_al_qei11b_jj = norm.fit(qe_i11b_jj['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11b_jj = norm.pdf(x_al, mu_al_qei11b_jj, std_al_qei11b_jj)

# Plot histogram
axes_i11b[1,1].plot(x_al, p_al_qei11b_jj, 'k', linewidth=1.5)
axes_i11b[1,1].set_xlim([0,100])
axes_i11b[1,1].fill_between(x_al, p_al_qei11b_jj, color='royalblue')
axes_i11b[1,1].axvline(qe_i11b_jj['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[1,1].text(qe_i11b_jj['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_jj['NT_GER'].mean()))
axes_i11b[1,1].text(qe_i11b_jj['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_jj['NT_GER'].std()))
axes_i11b[1,1].set_title("J:Financiamento pela IES", fontsize=size_subtitle)

####### Alternative K #######
qe_i11b_kk = qe_i11.loc[(qe_i11['QE_I11'] == 'K')]

mu_al_qei11b_kk, std_al_qei11b_kk = norm.fit(qe_i11b_kk['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei11b_kk = norm.pdf(x_al, mu_al_qei11b_kk, std_al_qei11b_kk)

# Plot histogram
axes_i11b[2,0].plot(x_al, p_al_qei11b_kk, 'k', linewidth=1.5)
axes_i11b[2,0].set_xlim([0,100])
axes_i11b[2,0].fill_between(x_al, p_al_qei11b_kk, color='royalblue')
axes_i11b[2,0].axvline(qe_i11b_kk['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[2,0].text(qe_i11b_kk['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_kk['NT_GER'].mean()))
axes_i11b[2,0].text(qe_i11b_kk['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_kk['NT_GER'].std()))
axes_i11b[2,0].set_title("K:Financiamento bancário", fontsize=size_subtitle)

axes_i11b[2,1].axis('off')

for ax in axes_i11b.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i11b.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I11B_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()


#%% QE_13
fig_i13, axes_i13 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i13.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de bolsa acadêmica durante a graduação',
                 fontsize=size_title)

####### Alternative A #######
qe_i13_aa = qe_i13.loc[(qe_i13['QE_I13'] == 'A')]

# Média e desvio padrão
mu_al_qei13_aa, std_al_qei13_aa = norm.fit(qe_i13_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_aa = norm.pdf(x_al, mu_al_qei13_aa, std_al_qei13_aa)

# Plot histogram
axes_i13[0,0].plot(x_al, p_al_qei13_aa, 'k', linewidth=1.5)
axes_i13[0,0].set_xlim([0,100])
axes_i13[0,0].fill_between(x_al, p_al_qei13_aa, color='royalblue')
axes_i13[0,0].axvline(qe_i13_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[0,0].text(qe_i13_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_aa['NT_GER'].mean()))
axes_i13[0,0].text(qe_i13_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_aa['NT_GER'].std()))
axes_i13[0,0].set_title("A:Nenhum", fontsize=size_subtitle)

####### Alternative B #######
qe_i13_bb = qe_i13.loc[(qe_i13['QE_I13'] == 'B')]

mu_al_qei13_bb, std_al_qei13_bb = norm.fit(qe_i13_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_bb = norm.pdf(x_al, mu_al_qei13_bb, std_al_qei13_bb)

# Plot histogram
axes_i13[0,1].plot(x_al, p_al_qei13_bb, 'k', linewidth=1.5)
axes_i13[0,1].set_xlim([0,100])
axes_i13[0,1].fill_between(x_al, p_al_qei13_bb, color='royalblue')
axes_i13[0,1].axvline(qe_i13_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[0,1].text(qe_i13_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_bb['NT_GER'].mean()))
axes_i13[0,1].text(qe_i13_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_bb['NT_GER'].std()))
axes_i13[0,1].set_title("B:PIBIC", fontsize=size_subtitle)

####### Alternative C #######
qe_i13_cc = qe_i13.loc[(qe_i13['QE_I13'] == 'C')]

mu_al_qei13_cc, std_al_qei13_cc = norm.fit(qe_i13_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_cc = norm.pdf(x_al, mu_al_qei13_cc, std_al_qei13_cc)

# Plot histogram
axes_i13[1,0].plot(x_al, p_al_qei13_cc, 'k', linewidth=1.5)
axes_i13[1,0].set_xlim([0,100])
axes_i13[1,0].fill_between(x_al, p_al_qei13_cc, color='royalblue')
axes_i13[1,0].axvline(qe_i13_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[1,0].text(qe_i13_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_cc['NT_GER'].mean()))
axes_i13[1,0].text(qe_i13_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_cc['NT_GER'].std()))
axes_i13[1,0].set_title("C:Extensão", fontsize=size_subtitle)

####### Alternative D #######
qe_i13_dd = qe_i13.loc[(qe_i13['QE_I13'] == 'D')]

mu_al_qei13_dd, std_al_qei13_dd = norm.fit(qe_i13_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_dd = norm.pdf(x_al, mu_al_qei13_dd, std_al_qei13_dd)

# Plot histogram
axes_i13[1,1].plot(x_al, p_al_qei13_dd, 'k', linewidth=1.5)
axes_i13[1,1].set_xlim([0,100])
axes_i13[1,1].fill_between(x_al, p_al_qei13_dd, color='royalblue')
axes_i13[1,1].axvline(qe_i13_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[1,1].text(qe_i13_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_dd['NT_GER'].mean()))
axes_i13[1,1].text(qe_i13_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_dd['NT_GER'].std()))
axes_i13[1,1].set_title("D:Monitoria/tutoria", fontsize=size_subtitle)

####### Alternative E #######
qe_i13_ee = qe_i13.loc[(qe_i13['QE_I13'] == 'E')]

mu_al_qei13_ee, std_al_qei13_ee = norm.fit(qe_i13_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_ee = norm.pdf(x_al, mu_al_qei13_ee, std_al_qei13_ee)

# Plot histogram
axes_i13[2,0].plot(x_al, p_al_qei13_ee, 'k', linewidth=1.5)
axes_i13[2,0].set_xlim([0,100])
axes_i13[2,0].fill_between(x_al, p_al_qei13_ee, color='royalblue')
axes_i13[2,0].axvline(qe_i13_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[2,0].text(qe_i13_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_ee['NT_GER'].mean()))
axes_i13[2,0].text(qe_i13_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_ee['NT_GER'].std()))
axes_i13[2,0].set_title("E:PET", fontsize=size_subtitle)

####### Alternative F #######
qe_i13_ff = qe_i13.loc[(qe_i13['QE_I13'] == 'F')]

mu_al_qei13_ff, std_al_qei13_ff = norm.fit(qe_i13_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei13_ff = norm.pdf(x_al, mu_al_qei13_ff, std_al_qei13_ff)

# Plot histogram
axes_i13[2,1].plot(x_al, p_al_qei13_ff, 'k', linewidth=1.5)
axes_i13[2,1].set_xlim([0,100])
axes_i13[2,1].fill_between(x_al, p_al_qei13_ff, color='royalblue')
axes_i13[2,1].axvline(qe_i13_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[2,1].text(qe_i13_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_ff['NT_GER'].mean()))
axes_i13[2,1].text(qe_i13_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_ff['NT_GER'].std()))
axes_i13[2,1].set_title("F:Outro", fontsize=size_subtitle)

for ax in axes_i13.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i13.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/QE_I13_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I17
fig_i17, axes_i17 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i17.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de escola que cursou o ensino médio',
                 fontsize=size_title)


####### Alternative A #######
qe_i17_aa = qe_i17.loc[(qe_i17['QE_I17'] == 'A')]

# Média e desvio padrão
mu_al_qei17_aa, std_al_qei17_aa = norm.fit(qe_i17_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_aa = norm.pdf(x_al, mu_al_qei17_aa, std_al_qei17_aa)

# Plot histogram
axes_i17[0,0].plot(x_al, p_al_qei17_aa, 'k', linewidth=1.5)

axes_i17[0,0].fill_between(x_al, p_al_qei17_aa, color='royalblue')
axes_i17[0,0].axvline(qe_i17_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[0,0].text(qe_i17_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_aa['NT_GER'].mean()))
axes_i17[0,0].text(qe_i17_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_aa['NT_GER'].std()))
axes_i17[0,0].set_title("A:Todo em pública", fontsize=size_subtitle)

####### Alternative B #######
qe_i17_bb = qe_i17.loc[(qe_i17['QE_I17'] == 'B')]

mu_al_qei17_bb, std_al_qei17_bb = norm.fit(qe_i17_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_bb = norm.pdf(x_al, mu_al_qei17_bb, std_al_qei17_bb)

# Plot histogram
axes_i17[0,1].plot(x_al, p_al_qei17_bb, 'k', linewidth=1.5)

axes_i17[0,1].fill_between(x_al, p_al_qei17_bb, color='royalblue')
axes_i17[0,1].axvline(qe_i17_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[0,1].text(qe_i17_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_bb['NT_GER'].mean()))
axes_i17[0,1].text(qe_i17_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_bb['NT_GER'].std()))
axes_i17[0,1].set_title("B:Todo em particular", fontsize=size_subtitle)

####### Alternative C #######
qe_i17_cc = qe_i17.loc[(qe_i17['QE_I17'] == 'C')]

mu_al_qei17_cc, std_al_qei17_cc = norm.fit(qe_i17_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_cc = norm.pdf(x_al, mu_al_qei17_cc, std_al_qei17_cc)

# Plot histogram
axes_i17[1,0].plot(x_al, p_al_qei17_cc, 'k', linewidth=1.5)

axes_i17[1,0].fill_between(x_al, p_al_qei17_cc, color='royalblue')
axes_i17[1,0].axvline(qe_i17_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[1,0].text(qe_i17_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_cc['NT_GER'].mean()))
axes_i17[1,0].text(qe_i17_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_cc['NT_GER'].std()))
axes_i17[1,0].set_title("C:Todo no exterior", fontsize=size_subtitle)

####### Alternative D #######
qe_i17_dd = qe_i17.loc[(qe_i17['QE_I17'] == 'D')]

mu_al_qei17_dd, std_al_qei17_dd = norm.fit(qe_i17_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_dd = norm.pdf(x_al, mu_al_qei17_dd, std_al_qei17_dd)

# Plot histogram
axes_i17[1,1].plot(x_al, p_al_qei17_dd, 'k', linewidth=1.5)
axes_i17[1,1].fill_between(x_al, p_al_qei17_dd, color='royalblue')
axes_i17[1,1].axvline(qe_i17_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[1,1].text(qe_i17_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_dd['NT_GER'].mean()))
axes_i17[1,1].text(qe_i17_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_dd['NT_GER'].std()))
axes_i17[1,1].set_title("D:Maior parte em pública", fontsize=size_subtitle)

####### Alternative E #######
qe_i17_ee = qe_i17.loc[(qe_i17['QE_I17'] == 'E')]

mu_al_qei17_ee, std_al_qei17_ee = norm.fit(qe_i17_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_ee = norm.pdf(x_al, mu_al_qei17_ee, std_al_qei17_ee)

# Plot histogram
axes_i17[2,0].plot(x_al, p_al_qei17_ee, 'k', linewidth=1.5)

axes_i17[2,0].fill_between(x_al, p_al_qei17_ee, color='royalblue')
axes_i17[2,0].axvline(qe_i17_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[2,0].text(qe_i17_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_ee['NT_GER'].mean()))
axes_i17[2,0].text(qe_i17_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_ee['NT_GER'].std()))
axes_i17[2,0].set_title("E:Maior parte em particular", fontsize=size_subtitle)

####### Alternative F #######
qe_i17_ff = qe_i17.loc[(qe_i17['QE_I17'] == 'F')]

mu_al_qei17_ff, std_al_qei17_ff = norm.fit(qe_i17_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei17_ff = norm.pdf(x_al, mu_al_qei17_ff, std_al_qei17_ff)

# Plot histogram
axes_i17[2,1].plot(x_al, p_al_qei17_ff, 'k', linewidth=1.5)

axes_i17[2,1].fill_between(x_al, p_al_qei17_ff, color='royalblue')
axes_i17[2,1].axvline(qe_i17_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[2,1].text(qe_i17_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_ff['NT_GER'].mean()))
axes_i17[2,1].text(qe_i17_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_ff['NT_GER'].std()))
axes_i17[2,1].set_title("F:Brasil e exterior", fontsize=size_subtitle)

for ax in axes_i17.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i17.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/QE_I17_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()


#%% QE_I18
fig_i18, axes_i18 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i18.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Modalidade de ensino médio',
                 fontsize=size_title)


####### Alternative A #######
qe_i18_aa = qe_i18.loc[(qe_i18['QE_I18'] == 'A')]

# Média e desvio padrão
mu_al_qei18_aa, std_al_qei18_aa = norm.fit(qe_i18_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei18_aa = norm.pdf(x_al, mu_al_qei18_aa, std_al_qei18_aa)

# Plot histogram
axes_i18[0,0].plot(x_al, p_al_qei18_aa, 'k', linewidth=1.5)

axes_i18[0,0].fill_between(x_al, p_al_qei18_aa, color='royalblue')
axes_i18[0,0].axvline(qe_i18_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i18[0,0].text(qe_i18_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i18_aa['NT_GER'].mean()))
axes_i18[0,0].text(qe_i18_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i18_aa['NT_GER'].std()))
axes_i18[0,0].set_title("A:Tradicional", fontsize=size_subtitle)

####### Alternative B #######
qe_i18_bb = qe_i18.loc[(qe_i18['QE_I18'] == 'B')]

mu_al_qei18_bb, std_al_qei18_bb = norm.fit(qe_i18_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei18_bb = norm.pdf(x_al, mu_al_qei18_bb, std_al_qei18_bb)

# Plot histogram
axes_i18[0,1].plot(x_al, p_al_qei18_bb, 'k', linewidth=1.5)

axes_i18[0,1].fill_between(x_al, p_al_qei18_bb, color='royalblue')
axes_i18[0,1].axvline(qe_i18_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i18[0,1].text(qe_i18_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i18_bb['NT_GER'].mean()))
axes_i18[0,1].text(qe_i18_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i18_bb['NT_GER'].std()))
axes_i18[0,1].set_title("B:Profissionalizante técnico", fontsize=size_subtitle)

####### Alternative C #######
qe_i18_cc = qe_i18.loc[(qe_i18['QE_I18'] == 'C')]

mu_al_qei18_cc, std_al_qei18_cc = norm.fit(qe_i18_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei18_cc = norm.pdf(x_al, mu_al_qei18_cc, std_al_qei18_cc)

# Plot histogram
axes_i18[1,0].plot(x_al, p_al_qei18_cc, 'k', linewidth=1.5)

axes_i18[1,0].fill_between(x_al, p_al_qei18_cc, color='royalblue')
axes_i18[1,0].axvline(qe_i18_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i18[1,0].text(qe_i18_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i18_cc['NT_GER'].mean()))
axes_i18[1,0].text(qe_i18_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i18_cc['NT_GER'].std()))
axes_i18[1,0].set_title("C:Profissionalizante magistério", fontsize=size_subtitle)

####### Alternative D #######
qe_i18_dd = qe_i18.loc[(qe_i18['QE_I18'] == 'D')]

mu_al_qei18_dd, std_al_qei18_dd = norm.fit(qe_i18_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei18_dd = norm.pdf(x_al, mu_al_qei18_dd, std_al_qei18_dd)

# Plot histogram
axes_i18[1,1].plot(x_al, p_al_qei18_dd, 'k', linewidth=1.5)
axes_i18[1,1].fill_between(x_al, p_al_qei18_dd, color='royalblue')
axes_i18[1,1].axvline(qe_i18_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i18[1,1].text(qe_i18_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i18_dd['NT_GER'].mean()))
axes_i18[1,1].text(qe_i18_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i18_dd['NT_GER'].std()))
axes_i18[1,1].set_title("D:EJA e/ou Supletivo", fontsize=size_subtitle)

####### Alternative E #######
qe_i18_ee = qe_i18.loc[(qe_i18['QE_I18'] == 'E')]

mu_al_qei18_ee, std_al_qei18_ee = norm.fit(qe_i18_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei18_ee = norm.pdf(x_al, mu_al_qei18_ee, std_al_qei18_ee)

# Plot histogram
axes_i18[2,0].plot(x_al, p_al_qei18_ee, 'k', linewidth=1.5)

axes_i18[2,0].fill_between(x_al, p_al_qei18_ee, color='royalblue')
axes_i18[2,0].axvline(qe_i18_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i18[2,0].text(qe_i18_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i18_ee['NT_GER'].mean()))
axes_i18[2,0].text(qe_i18_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i18_ee['NT_GER'].std()))
axes_i18[2,0].set_title("E:Outro", fontsize=size_subtitle)

axes_i18[2,1].axis('off')

for ax in axes_i18.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i18.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/maior_impacto/QE_I18_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I23
fig_i23, axes_i23 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i23.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Horas de estudo por semana (excluindo aulas)',
                 fontsize=size_title)


####### Alternative A #######
qe_i23_aa = qe_i23.loc[(qe_i23['QE_I23'] == 'A')]

# Média e desvio padrão
mu_al_qei23_aa, std_al_qei23_aa = norm.fit(qe_i23_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei23_aa = norm.pdf(x_al, mu_al_qei23_aa, std_al_qei23_aa)

# Plot histogram
axes_i23[0,0].plot(x_al, p_al_qei23_aa, 'k', linewidth=1.5)

axes_i23[0,0].fill_between(x_al, p_al_qei23_aa, color='royalblue')
axes_i23[0,0].axvline(qe_i23_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[0,0].text(qe_i23_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_aa['NT_GER'].mean()))
axes_i23[0,0].text(qe_i23_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_aa['NT_GER'].std()))
axes_i23[0,0].set_title("A:Nenhuma", fontsize=size_subtitle)

####### Alternative B #######
qe_i23_bb = qe_i23.loc[(qe_i23['QE_I23'] == 'B')]

mu_al_qei23_bb, std_al_qei23_bb = norm.fit(qe_i23_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei23_bb = norm.pdf(x_al, mu_al_qei23_bb, std_al_qei23_bb)

# Plot histogram
axes_i23[0,1].plot(x_al, p_al_qei23_bb, 'k', linewidth=1.5)

axes_i23[0,1].fill_between(x_al, p_al_qei23_bb, color='royalblue')
axes_i23[0,1].axvline(qe_i23_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[0,1].text(qe_i23_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_bb['NT_GER'].mean()))
axes_i23[0,1].text(qe_i23_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_bb['NT_GER'].std()))
axes_i23[0,1].set_title("B:Uma a três", fontsize=size_subtitle)

####### Alternative C #######
qe_i23_cc = qe_i23.loc[(qe_i23['QE_I23'] == 'C')]

mu_al_qei23_cc, std_al_qei23_cc = norm.fit(qe_i23_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei23_cc = norm.pdf(x_al, mu_al_qei23_cc, std_al_qei23_cc)

# Plot histogram
axes_i23[1,0].plot(x_al, p_al_qei23_cc, 'k', linewidth=1.5)

axes_i23[1,0].fill_between(x_al, p_al_qei23_cc, color='royalblue')
axes_i23[1,0].axvline(qe_i23_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[1,0].text(qe_i23_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_cc['NT_GER'].mean()))
axes_i23[1,0].text(qe_i23_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_cc['NT_GER'].std()))
axes_i23[1,0].set_title("C:Quatro a sete", fontsize=size_subtitle)

####### Alternative D #######
qe_i23_dd = qe_i23.loc[(qe_i23['QE_I23'] == 'D')]

mu_al_qei23_dd, std_al_qei23_dd = norm.fit(qe_i23_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei23_dd = norm.pdf(x_al, mu_al_qei23_dd, std_al_qei23_dd)

# Plot histogram
axes_i23[1,1].plot(x_al, p_al_qei23_dd, 'k', linewidth=1.5)
axes_i23[1,1].fill_between(x_al, p_al_qei23_dd, color='royalblue')
axes_i23[1,1].axvline(qe_i23_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[1,1].text(qe_i23_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_dd['NT_GER'].mean()))
axes_i23[1,1].text(qe_i23_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_dd['NT_GER'].std()))
axes_i23[1,1].set_title("D:Oito a doze", fontsize=size_subtitle)

####### Alternative E #######
qe_i23_ee = qe_i23.loc[(qe_i23['QE_I23'] == 'E')]

mu_al_qei23_ee, std_al_qei23_ee = norm.fit(qe_i23_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei23_ee = norm.pdf(x_al, mu_al_qei23_ee, std_al_qei23_ee)

# Plot histogram
axes_i23[2,0].plot(x_al, p_al_qei23_ee, 'k', linewidth=1.5)

axes_i23[2,0].fill_between(x_al, p_al_qei23_ee, color='royalblue')
axes_i23[2,0].axvline(qe_i23_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[2,0].text(qe_i23_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_ee['NT_GER'].mean()))
axes_i23[2,0].text(qe_i23_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_ee['NT_GER'].std()))
axes_i23[2,0].set_title("E:Mais de doze", fontsize=size_subtitle)

axes_i23[2,1].axis('off')

for ax in axes_i23.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i23.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL//imagens/maior_impacto/QE_I23_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% 
############################## Subplots - Menor impacto ##############################
############################## QE_I01 ##############################
fig_i01, axes_i01 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i01.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Estado civil',
                 fontsize=size_title)


####### Alternative A #######
qe_i01_aa = qe_i01.loc[(qe_i01['QE_I01'] == 'A')]

# Média e desvio padrão
mu_al_qei01_aa, std_al_qei01_aa = norm.fit(qe_i01_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei01_aa = norm.pdf(x_al, mu_al_qei01_aa, std_al_qei01_aa)

# Plot histogram
axes_i01[0,0].plot(x_al, p_al_qei01_aa, 'k', linewidth=1.5)

axes_i01[0,0].fill_between(x_al, p_al_qei01_aa, color='royalblue')
axes_i01[0,0].axvline(qe_i01_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[0,0].text(qe_i01_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_aa['NT_GER'].mean()))
axes_i01[0,0].text(qe_i01_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()))
axes_i01[0,0].set_title("A:Solteiro(a)", fontsize=size_subtitle)

####### Alternative B #######
qe_i01_bb = qe_i01.loc[(qe_i01['QE_I01'] == 'B')]

mu_al_qei01_bb, std_al_qei01_bb = norm.fit(qe_i01_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei01_bb = norm.pdf(x_al, mu_al_qei01_bb, std_al_qei01_bb)

# Plot histogram
axes_i01[0,1].plot(x_al, p_al_qei01_bb, 'k', linewidth=1.5)

axes_i01[0,1].fill_between(x_al, p_al_qei01_bb, color='royalblue')
axes_i01[0,1].axvline(qe_i01_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[0,1].text(qe_i01_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_bb['NT_GER'].mean()))
axes_i01[0,1].text(qe_i01_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_bb['NT_GER'].std()))
axes_i01[0,1].set_title("B:Casado(a)", fontsize=size_subtitle)

####### Alternative C #######
qe_i01_cc = qe_i01.loc[(qe_i01['QE_I01'] == 'C')]

mu_al_qei01_cc, std_al_qei01_cc = norm.fit(qe_i01_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei01_cc = norm.pdf(x_al, mu_al_qei01_cc, std_al_qei01_cc)

# Plot histogram
axes_i01[1,0].plot(x_al, p_al_qei01_cc, 'k', linewidth=1.5)

axes_i01[1,0].fill_between(x_al, p_al_qei01_cc, color='royalblue')
axes_i01[1,0].axvline(qe_i01_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[1,0].text(qe_i01_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_cc['NT_GER'].mean()))
axes_i01[1,0].text(qe_i01_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_cc['NT_GER'].std()))
axes_i01[1,0].set_title("C:Separado(a)", fontsize=size_subtitle)

####### Alternative D #######
qe_i01_dd = qe_i01.loc[(qe_i01['QE_I01'] == 'D')]

mu_al_qei01_dd, std_al_qei01_dd = norm.fit(qe_i01_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei01_dd = norm.pdf(x_al, mu_al_qei01_dd, std_al_qei01_dd)

# Plot histogram
axes_i01[1,1].plot(x_al, p_al_qei01_dd, 'k', linewidth=1.5)
axes_i01[1,1].fill_between(x_al, p_al_qei01_dd, color='royalblue')
axes_i01[1,1].axvline(qe_i01_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[1,1].text(qe_i01_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_dd['NT_GER'].mean()))
axes_i01[1,1].text(qe_i01_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_dd['NT_GER'].std()))
axes_i01[1,1].set_title("D:Viúvo(a)", fontsize=size_subtitle)

####### Alternative E #######
qe_i01_ee = qe_i01.loc[(qe_i01['QE_I01'] == 'E')]

mu_al_qei01_ee, std_al_qei01_ee = norm.fit(qe_i01_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei01_ee = norm.pdf(x_al, mu_al_qei01_ee, std_al_qei01_ee)

# Plot histogram
axes_i01[2,0].plot(x_al, p_al_qei01_ee, 'k', linewidth=1.5)

axes_i01[2,0].fill_between(x_al, p_al_qei01_ee, color='royalblue')
axes_i01[2,0].axvline(qe_i01_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[2,0].text(qe_i01_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_ee['NT_GER'].mean()))
axes_i01[2,0].text(qe_i01_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_ee['NT_GER'].std()))
axes_i01[2,0].set_title("E:Outro", fontsize=size_subtitle)

axes_i01[2,1].axis('off')

for ax in axes_i01.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i01.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/QE_I01_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()


#%%
############################## QE_I03 ##############################
fig_i03, axes_i03 = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i03.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Nacionalidade',
                 fontsize=size_title)


####### Alternative A #######
qe_i03_aa = qe_i03.loc[(qe_i03['QE_I03'] == 'A')]

# Média e desvio padrão
mu_al_qei03_aa, std_al_qei03_aa = norm.fit(qe_i03_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei03_aa = norm.pdf(x_al, mu_al_qei03_aa, std_al_qei03_aa)

# Plot histogram
axes_i03[0,0].plot(x_al, p_al_qei03_aa, 'k', linewidth=1.5)

axes_i03[0,0].fill_between(x_al, p_al_qei03_aa, color='royalblue')
axes_i03[0,0].axvline(qe_i03_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[0,0].text(qe_i03_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_aa['NT_GER'].mean()))
axes_i03[0,0].text(qe_i03_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_aa['NT_GER'].std()))
axes_i03[0,0].set_title("A:Brasileira", fontsize=size_subtitle)

####### Alternative B #######
qe_i03_bb = qe_i03.loc[(qe_i03['QE_I03'] == 'B')]

mu_al_qei03_bb, std_al_qei03_bb = norm.fit(qe_i03_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei03_bb = norm.pdf(x_al, mu_al_qei03_bb, std_al_qei03_bb)

# Plot histogram
axes_i03[0,1].plot(x_al, p_al_qei03_bb, 'k', linewidth=1.5)

axes_i03[0,1].fill_between(x_al, p_al_qei03_bb, color='royalblue')
axes_i03[0,1].axvline(qe_i03_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[0,1].text(qe_i03_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_bb['NT_GER'].mean()))
axes_i03[0,1].text(qe_i03_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_bb['NT_GER'].std()))
axes_i03[0,1].set_title("B:Brasileira naturalizada", fontsize=size_subtitle)

####### Alternative C #######
qe_i03_cc = qe_i03.loc[(qe_i03['QE_I03'] == 'C')]

mu_al_qei03_cc, std_al_qei03_cc = norm.fit(qe_i03_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei03_cc = norm.pdf(x_al, mu_al_qei03_cc, std_al_qei03_cc)

# Plot histogram
axes_i03[1,0].plot(x_al, p_al_qei03_cc, 'k', linewidth=1.5)

axes_i03[1,0].fill_between(x_al, p_al_qei03_cc, color='royalblue')
axes_i03[1,0].axvline(qe_i03_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[1,0].text(qe_i03_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_cc['NT_GER'].mean()))
axes_i03[1,0].text(qe_i03_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_cc['NT_GER'].std()))
axes_i03[1,0].set_title("C:Estrageira", fontsize=size_subtitle)

axes_i03[1,1].axis('off')

for ax in axes_i03.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i03.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/maior_impacto/QE_I03_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I12 ##############################
fig_i12, axes_i12 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i12.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de bolsa de permanência durante da graduação',
                 fontsize=size_title)


####### Alternative A #######
qe_i12_aa = qe_i12.loc[(qe_i12['QE_I12'] == 'A')]

# Média e desvio padrão
mu_al_qei12_aa, std_al_qei12_aa = norm.fit(qe_i12_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_aa = norm.pdf(x_al, mu_al_qei12_aa, std_al_qei12_aa)

# Plot histogram
axes_i12[0,0].plot(x_al, p_al_qei12_aa, 'k', linewidth=1.5)

axes_i12[0,0].fill_between(x_al, p_al_qei12_aa, color='royalblue')
axes_i12[0,0].axvline(qe_i12_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[0,0].text(qe_i12_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_aa['NT_GER'].mean()))
axes_i12[0,0].text(qe_i12_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()))
axes_i01[0,0].set_title("A:Nenhum", fontsize=size_subtitle)

####### Alternative B #######
qe_i12_bb = qe_i12.loc[(qe_i12['QE_I12'] == 'B')]

mu_al_qei12_bb, std_al_qei12_bb = norm.fit(qe_i12_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_bb = norm.pdf(x_al, mu_al_qei12_bb, std_al_qei12_bb)

# Plot histogram
axes_i12[0,1].plot(x_al, p_al_qei12_bb, 'k', linewidth=1.5)

axes_i12[0,1].fill_between(x_al, p_al_qei12_bb, color='royalblue')
axes_i12[0,1].axvline(qe_i12_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[0,1].text(qe_i12_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_bb['NT_GER'].mean()))
axes_i12[0,1].text(qe_i12_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_bb['NT_GER'].std()))
axes_i12[0,1].set_title("B:Aux.moradia", fontsize=size_subtitle)

####### Alternative C #######
qe_i12_cc = qe_i12.loc[(qe_i12['QE_I12'] == 'C')]

mu_al_qei12_cc, std_al_qei12_cc = norm.fit(qe_i12_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_cc = norm.pdf(x_al, mu_al_qei12_cc, std_al_qei12_cc)

# Plot histogram
axes_i12[1,0].plot(x_al, p_al_qei12_cc, 'k', linewidth=1.5)

axes_i12[1,0].fill_between(x_al, p_al_qei12_cc, color='royalblue')
axes_i12[1,0].axvline(qe_i12_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[1,0].text(qe_i12_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_cc['NT_GER'].mean()))
axes_i12[1,0].text(qe_i12_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_cc['NT_GER'].std()))
axes_i12[1,0].set_title("C:Aux.alimentação", fontsize=size_subtitle)

####### Alternative D #######
qe_i12_dd = qe_i12.loc[(qe_i12['QE_I12'] == 'D')]

mu_al_qei12_dd, std_al_qei12_dd = norm.fit(qe_i12_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_dd = norm.pdf(x_al, mu_al_qei12_dd, std_al_qei12_dd)

# Plot histogram
axes_i12[1,0].plot(x_al, p_al_qei12_dd, 'k', linewidth=1.5)
axes_i12[1,0].fill_between(x_al, p_al_qei12_dd, color='royalblue')
axes_i12[1,0].axvline(qe_i12_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[1,0].text(qe_i12_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_dd['NT_GER'].mean()))
axes_i12[1,0].text(qe_i12_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_dd['NT_GER'].std()))
axes_i12[1,0].set_title("D:Aux.moradia e alimentação", fontsize=size_subtitle)

####### Alternative E #######
qe_i12_ee = qe_i12.loc[(qe_i12['QE_I12'] == 'E')]

mu_al_qei12_ee, std_al_qei12_ee = norm.fit(qe_i12_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_ee = norm.pdf(x_al, mu_al_qei12_ee, std_al_qei12_ee)

# Plot histogram
axes_i12[2,0].plot(x_al, p_al_qei12_ee, 'k', linewidth=1.5)

axes_i12[2,0].fill_between(x_al, p_al_qei12_ee, color='royalblue')
axes_i12[2,0].axvline(qe_i12_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[2,0].text(qe_i12_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_ee['NT_GER'].mean()))
axes_i12[2,0].text(qe_i12_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_ee['NT_GER'].std()))
axes_i12[2,0].set_title("E:Aux.permanência", fontsize=size_subtitle)

####### Alternative F #######
qe_i12_ff = qe_i12.loc[(qe_i12['QE_I12'] == 'F')]

mu_al_qei12_ff, std_al_qei12_ff = norm.fit(qe_i12_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei12_ff = norm.pdf(x_al, mu_al_qei12_ff, std_al_qei12_ff)

# Plot histogram
axes_i12[2,1].plot(x_al, p_al_qei12_ff, 'k', linewidth=1.5)

axes_i12[2,1].fill_between(x_al, p_al_qei12_ff, color='royalblue')
axes_i12[2,1].axvline(qe_i12_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[2,1].text(qe_i12_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_ff['NT_GER'].mean()))
axes_i12[2,1].text(qe_i12_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_ff['NT_GER'].std()))
axes_i12[2,1].set_title("F:Outro", fontsize=size_subtitle)

for ax in axes_i12.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i12.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/QE_I12_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I15 ##############################
fig_i15, axes_i15 = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i15.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Ingresso por ação afrimativa e critério',
                 fontsize=size_title)


####### Alternative A #######
qe_i15_aa = qe_i15.loc[(qe_i15['QE_I15'] == 'A')]

# Média e desvio padrão
mu_al_qei15_aa, std_al_qei15_aa = norm.fit(qe_i15_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_aa = norm.pdf(x_al, mu_al_qei15_aa, std_al_qei15_aa)

# Plot histogram
axes_i15[0,0].plot(x_al, p_al_qei15_aa, 'k', linewidth=1.5)

axes_i15[0,0].fill_between(x_al, p_al_qei15_aa, color='royalblue')
axes_i15[0,0].axvline(qe_i15_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[0,0].text(qe_i15_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_aa['NT_GER'].mean()))
axes_i15[0,0].text(qe_i15_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()))
axes_i01[0,0].set_title("A:Não", fontsize=size_subtitle)

####### Alternative B #######
qe_i15_bb = qe_i15.loc[(qe_i15['QE_I15'] == 'B')]

mu_al_qei15_bb, std_al_qei15_bb = norm.fit(qe_i15_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_bb = norm.pdf(x_al, mu_al_qei15_bb, std_al_qei15_bb)

# Plot histogram
axes_i15[0,1].plot(x_al, p_al_qei15_bb, 'k', linewidth=1.5)

axes_i15[0,1].fill_between(x_al, p_al_qei15_bb, color='royalblue')
axes_i15[0,1].axvline(qe_i15_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[0,1].text(qe_i15_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_bb['NT_GER'].mean()))
axes_i15[0,1].text(qe_i15_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_bb['NT_GER'].std()))
axes_i15[0,1].set_title("B:Sim;étnico-racial", fontsize=size_subtitle)

####### Alternative C #######
qe_i15_cc = qe_i15.loc[(qe_i15['QE_I15'] == 'C')]

mu_al_qei15_cc, std_al_qei15_cc = norm.fit(qe_i15_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_cc = norm.pdf(x_al, mu_al_qei15_cc, std_al_qei15_cc)

# Plot histogram
axes_i15[1,0].plot(x_al, p_al_qei15_cc, 'k', linewidth=1.5)

axes_i15[1,0].fill_between(x_al, p_al_qei15_cc, color='royalblue')
axes_i15[1,0].axvline(qe_i15_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[1,0].text(qe_i15_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_cc['NT_GER'].mean()))
axes_i15[1,0].text(qe_i15_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_cc['NT_GER'].std()))
axes_i15[1,0].set_title("C:Sim;renda", fontsize=size_subtitle)

####### Alternative D #######
qe_i15_dd = qe_i15.loc[(qe_i15['QE_I15'] == 'D')]

mu_al_qei15_dd, std_al_qei15_dd = norm.fit(qe_i15_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_dd = norm.pdf(x_al, mu_al_qei15_dd, std_al_qei15_dd)

# Plot histogram
axes_i15[1,0].plot(x_al, p_al_qei15_dd, 'k', linewidth=1.5)
axes_i15[1,0].fill_between(x_al, p_al_qei15_dd, color='royalblue')
axes_i15[1,0].axvline(qe_i15_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[1,0].text(qe_i15_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_dd['NT_GER'].mean()))
axes_i15[1,0].text(qe_i15_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_dd['NT_GER'].std()))
axes_i15[1,0].set_title("D:Sim;esc.pública/bolsa esc. privada", fontsize=size_subtitle)

####### Alternative E #######
qe_i15_ee = qe_i15.loc[(qe_i15['QE_I15'] == 'E')]

mu_al_qei15_ee, std_al_qei15_ee = norm.fit(qe_i15_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_ee = norm.pdf(x_al, mu_al_qei15_ee, std_al_qei15_ee)

# Plot histogram
axes_i15[2,0].plot(x_al, p_al_qei15_ee, 'k', linewidth=1.5)

axes_i15[2,0].fill_between(x_al, p_al_qei15_ee, color='royalblue')
axes_i15[2,0].axvline(qe_i15_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[2,0].text(qe_i15_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_ee['NT_GER'].mean()))
axes_i15[2,0].text(qe_i15_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_ee['NT_GER'].std()))
axes_i15[2,0].set_title("E:Sim;2 ou mais critérios anteriores", fontsize=size_subtitle)

####### Alternative F #######
qe_i15_ff = qe_i15.loc[(qe_i15['QE_I15'] == 'F')]

mu_al_qei15_ff, std_al_qei15_ff = norm.fit(qe_i15_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei15_ff = norm.pdf(x_al, mu_al_qei15_ff, std_al_qei15_ff)

# Plot histogram
axes_i15[1,2].plot(x_al, p_al_qei15_ff, 'k', linewidth=1.5)

axes_i15[1,2].fill_between(x_al, p_al_qei15_ff, color='royalblue')
axes_i15[1,2].axvline(qe_i15_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[1,2].text(qe_i15_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_ff['NT_GER'].mean()))
axes_i15[1,2].text(qe_i15_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_ff['NT_GER'].std()))
axes_i15[1,2].set_title("F:Sim;outro critério", fontsize=size_subtitle)

for ax in axes_i15.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i15.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/QE_I15_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I16
############################## QE_I16 ##############################
#%% QE_I19
############################## QE_I19 ##############################
fig_i19, axes_i19 = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=True, figsize=(20,10))
fig_i19.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Pessoa que mais incentivou a cursar a graduação',
                 fontsize=size_title)


####### Alternative A #######
qe_i19_aa = qe_i19.loc[(qe_i19['QE_I19'] == 'A')]

# Média e desvio padrão
mu_al_qei19_aa, std_al_qei19_aa = norm.fit(qe_i19_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_aa = norm.pdf(x_al, mu_al_qei19_aa, std_al_qei19_aa)

# Plot histogram
axes_i19[0,0].plot(x_al, p_al_qei19_aa, 'k', linewidth=1.5)

axes_i19[0,0].fill_between(x_al, p_al_qei19_aa, color='royalblue')
axes_i19[0,0].axvline(qe_i19_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,0].text(qe_i19_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_aa['NT_GER'].mean()))
axes_i19[0,0].text(qe_i19_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()))
axes_i01[0,0].set_title("A:Todo em pública", fontsize=size_subtitle)

####### Alternative B #######
qe_i19_bb = qe_i19.loc[(qe_i19['QE_I19'] == 'B')]

mu_al_qei19_bb, std_al_qei19_bb = norm.fit(qe_i19_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_bb = norm.pdf(x_al, mu_al_qei19_bb, std_al_qei19_bb)

# Plot histogram
axes_i19[0,1].plot(x_al, p_al_qei19_bb, 'k', linewidth=1.5)

axes_i19[0,1].fill_between(x_al, p_al_qei19_bb, color='royalblue')
axes_i19[0,1].axvline(qe_i19_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,1].text(qe_i19_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_bb['NT_GER'].mean()))
axes_i19[0,1].text(qe_i19_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_bb['NT_GER'].std()))
axes_i19[0,1].set_title("B:Pais", fontsize=size_subtitle)

####### Alternative C #######
qe_i19_cc = qe_i19.loc[(qe_i19['QE_I19'] == 'C')]

mu_al_qei19_cc, std_al_qei19_cc = norm.fit(qe_i19_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_cc = norm.pdf(x_al, mu_al_qei19_cc, std_al_qei19_cc)

# Plot histogram
axes_i19[0,2].plot(x_al, p_al_qei19_cc, 'k', linewidth=1.5)

axes_i19[0,2].fill_between(x_al, p_al_qei19_cc, color='royalblue')
axes_i19[0,2].axvline(qe_i19_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,2].text(qe_i19_cc['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_cc['NT_GER'].mean()))
axes_i19[0,2].text(qe_i19_cc['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_cc['NT_GER'].std()))
axes_i19[0,2].set_title("C:Outros membros da família", fontsize=size_subtitle)

####### Alternative D #######
qe_i19_dd = qe_i19.loc[(qe_i19['QE_I19'] == 'D')]

mu_al_qei19_dd, std_al_qei19_dd = norm.fit(qe_i19_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.19)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_dd = norm.pdf(x_al, mu_al_qei19_dd, std_al_qei19_dd)

# Plot histogram
axes_i19[1,0].plot(x_al, p_al_qei19_dd, 'k', linewidth=1.5)
axes_i19[1,0].fill_between(x_al, p_al_qei19_dd, color='royalblue')
axes_i19[1,0].axvline(qe_i19_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,0].text(qe_i19_dd['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_dd['NT_GER'].mean()))
axes_i19[1,0].text(qe_i19_dd['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_dd['NT_GER'].std()))
axes_i19[1,0].set_title("D:Professores", fontsize=size_subtitle)

####### Alternative E #######
qe_i19_ee = qe_i19.loc[(qe_i19['QE_I19'] == 'E')]

mu_al_qei19_ee, std_al_qei19_ee = norm.fit(qe_i19_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_ee = norm.pdf(x_al, mu_al_qei19_ee, std_al_qei19_ee)

# Plot histogram
axes_i19[1,1].plot(x_al, p_al_qei19_ee, 'k', linewidth=1.5)

axes_i19[1,1].fill_between(x_al, p_al_qei19_ee, color='royalblue')
axes_i19[1,1].axvline(qe_i19_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,1].text(qe_i19_ee['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_ee['NT_GER'].mean()))
axes_i19[1,1].text(qe_i19_ee['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_ee['NT_GER'].std()))
axes_i19[1,1].set_title("E:Líder religioso", fontsize=size_subtitle)

####### Alternative F #######
qe_i19_ff = qe_i19.loc[(qe_i19['QE_I19'] == 'F')]

mu_al_qei19_ff, std_al_qei19_ff = norm.fit(qe_i19_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_ff = norm.pdf(x_al, mu_al_qei19_ff, std_al_qei19_ff)

# Plot histogram
axes_i19[1,2].plot(x_al, p_al_qei19_ff, 'k', linewidth=1.5)

axes_i19[1,2].fill_between(x_al, p_al_qei19_ff, color='royalblue')
axes_i19[1,2].axvline(qe_i19_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,2].text(qe_i19_ff['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_ff['NT_GER'].mean()))
axes_i19[1,2].text(qe_i19_ff['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_ff['NT_GER'].std()))
axes_i19[1,2].set_title("F:Colegas/amigos", fontsize=size_subtitle)

####### Alternative G #######
qe_i19_gg = qe_i19.loc[(qe_i19['QE_I19'] == 'F')]

mu_al_qei19_gg, std_al_qei19_gg = norm.fit(qe_i19_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei19_gg = norm.pdf(x_al, mu_al_qei19_gg, std_al_qei19_gg)

# Plot histogram
axes_i19[2,0].plot(x_al, p_al_qei19_gg, 'k', linewidth=1.5)

axes_i19[2,0].fill_between(x_al, p_al_qei19_gg, color='royalblue')
axes_i19[2,0].axvline(qe_i19_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[2,0].text(qe_i19_gg['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_gg['NT_GER'].mean()))
axes_i19[2,0].text(qe_i19_gg['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_gg['NT_GER'].std()))
axes_i19[2,0].set_title("G:Outras pessoas", fontsize=size_subtitle)

axes_i19[2,1].axis('off')
axes_i19[2,2].axis('off')

for ax in axes_i19.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i19.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/QE_I19_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I21
############################## QE_I21 ##############################
fig_i21, axes_i21 = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(20,10))
fig_i21.suptitle('Distribuição de notas do Enade em Alagoas de 2014 a 2018\n'    
                  'Dado socioeconômico:Alguém da família concluiu um curso superior',
                 fontsize=size_title)


####### Alternative A #######
qe_i21_aa = qe_i21.loc[(qe_i21['QE_I21'] == 'A')]

# Média e desvio padrão
mu_al_qei21_aa, std_al_qei21_aa = norm.fit(qe_i21_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei21_aa = norm.pdf(x_al, mu_al_qei21_aa, std_al_qei21_aa)

# Plot histogram
axes_i21[0,0].plot(x_al, p_al_qei21_aa, 'k', linewidth=1.5)
axes_i21[0,0].fill_between(x_al, p_al_qei21_aa, color='royalblue')
axes_i21[0,0].axvline(qe_i21_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i21[0,0].text(qe_i21_aa['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i21_aa['NT_GER'].mean()))
axes_i21[0,0].text(qe_i21_aa['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()))
axes_i21[0,0].set_title("A:Sim", fontsize=size_subtitle)

####### Alternative B #######
qe_i21_bb = qe_i21.loc[(qe_i21['QE_I21'] == 'B')]

mu_al_qei21_bb, std_al_qei21_bb = norm.fit(qe_i21_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando
p_al_qei21_bb = norm.pdf(x_al, mu_al_qei21_bb, std_al_qei21_bb)

# Plot histogram
axes_i21[0,1].plot(x_al, p_al_qei21_bb, 'k', linewidth=1.5)

axes_i21[0,1].fill_between(x_al, p_al_qei21_bb, color='royalblue')
axes_i21[0,1].axvline(qe_i21_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i21[0,1].text(qe_i21_bb['NT_GER'].mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i21_bb['NT_GER'].mean()))
axes_i21[0,1].text(qe_i21_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i21_bb['NT_GER'].std()))
axes_i21[0,1].set_title("B:Não", fontsize=size_subtitle)

axes_i21[1,0].axes('off')
axes_i21[2,0].axes('off')

for ax in axes_i21.flat:
    ax.set(xlabel='Nota', ylabel='Distribuição')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes_i21.flat:
    ax.label_outer()

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/menor_impacto/QE_I21_AL_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
########## Referências ##########
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