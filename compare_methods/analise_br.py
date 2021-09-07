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

def plot_axis_name(ax, fontsize, hide_labels=False):
    #ax.plot([1, 2])

    #ax.locator_params(nbins=3)
    if hide_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xlabel('Nota', fontsize=10)
        #ax.set_ylabel('Distribuição', fontsize=fontsize)
        #ax.set_title('Title', fontsize=fontsize)

#%% PREPARANDO OS DADOS

data_br2014 = pd.read_csv(r'tcc_data/BR_2014.csv')
data_br2015 = pd.read_csv(r'tcc_data/BR_2015.csv')
data_br2016 = pd.read_csv(r'tcc_data/BR_2016.csv')
data_br2017 = pd.read_csv(r'tcc_data/BR_2017.csv')
data_br2018 = pd.read_csv(r'tcc_data/BR_2018.csv')

labels_br = [] # Labels
features_br = [] # Features
features_br_list = [] # Guardando as variáveis das features
features_br_list_oh = [] # Variáveis das features com one-hot

#%% Percentual de valores NaN

nan_2014 = float(data_br2014['NT_GER'].isnull().sum());
nan_2015 = float(data_br2015['NT_GER'].isnull().sum());
nan_2016 = float(data_br2016['NT_GER'].isnull().sum());
nan_2017 = float(data_br2017['NT_GER'].isnull().sum());
nan_2018 = float(data_br2018['NT_GER'].isnull().sum());

column_2014 = float(data_br2014.shape[0]);
column_2015 = float(data_br2015.shape[0]);
column_2016 = float(data_br2016.shape[0]);
column_2017 = float(data_br2017.shape[0]);
column_2018 = float(data_br2018.shape[0]);

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

def processing_set_br(data_br2014, data_br2015, data_br2016, data_br2017, data_br2018):
    #% 2.1 - Limpeza
    del data_br2014['Unnamed: 0']
    del data_br2015['Unnamed: 0']
    del data_br2016['Unnamed: 0']
    del data_br2017['Unnamed: 0']
    del data_br2018['Unnamed: 0']

    # Escolhendo apenas as colunas de interesse
    data_br2014 = data_br2014.loc[:,'NT_GER':'QE_I26']
    data_br2015 = data_br2015.loc[:,'NT_GER':'QE_I26']
    data_br2016 = data_br2016.loc[:,'NT_GER':'QE_I26']
    data_br2017 = data_br2017.loc[:,'NT_GER':'QE_I26']
    data_br2018 = data_br2018.loc[:,'NT_GER':'QE_I26']

    data_br2014 = data_br2014.drop(data_br2014.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_br2015 = data_br2015.drop(data_br2015.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_br2016 = data_br2016.drop(data_br2016.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_br2017 = data_br2017.drop(data_br2017.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_br2018 = data_br2018.drop(data_br2018.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

    data_br2014 = data_br2014.drop(data_br2014.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_br2015 = data_br2015.drop(data_br2015.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_br2016 = data_br2016.drop(data_br2016.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_br2017 = data_br2017.drop(data_br2017.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_br2018 = data_br2018.drop(data_br2018.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    
    # MERGE NOS DADOS: data br
    frames = [data_br2014, data_br2015, data_br2016, data_br2017, data_br2018];
    data_br = pd.concat(frames);

    # Enriquecimento
    data_br['NT_GER'] = data_br['NT_GER'].str.replace(',','.')
    data_br['NT_GER'] = data_br['NT_GER'].astype(float)

    data_br_media = round(data_br['NT_GER'].mean(),2)
    
    data_br['NT_GER'] = data_br['NT_GER'].fillna(data_br_media)
    
    dataset_br = data_br
    
    describe_br = data_br.describe()
    
    # 3 - Transformação
    labels_br = np.array(data_br['NT_GER'])

    # Removendo as features de notas
    data_br = data_br.drop(['NT_GER'], axis = 1)
    
    features_br_list = list(data_br.columns)


    # One hot encoding - QE_I01 a QE_I26
    features_br = pd.get_dummies(data=data_br, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
    # Sbrvando os nomes das colunas (features) com os dados para uso posterior
    # depois de codificar
    features_br_list_oh = list(features_br.columns)
    #
    # Convertendo para numpy
    features_br = np.array(features_br)
    
    return features_br, labels_br, features_br_list_oh, dataset_br

#%% Aplicando o pré-processamento

features_br, labels_br, features_br_list_oh, dataset_br = processing_set_br(data_br2014, data_br2015, data_br2016, data_br2017, data_br2018)

#%% Percentual de valores NaN

nan_br = float(nan_2014+nan_2015+nan_2016+nan_2017+nan_2018);

column_br = float(labels_br.shape[0]);

per_br = nan_br/column_br;

print("Qtde. % NaN values in NT_GER BR:", 100*round(per_br, 4));
#%% Dados estatísticos gerais
# < 20 --> Amostra
# >= 20 --> população
import statistics as stats

min_br = min(labels_br)
max_br = max(labels_br)
mean_br = stats.mean(labels_br)
median_br = stats.median(labels_br)
variance_br = stats.variance(labels_br)
std_dev_br = stats.stdev(labels_br)

print("Min:", round(min_br, 4))
print("Max:", round(max_br, 4))
print("Media:", round(mean_br, 4))
print("Mediana:", round(median_br, 4))
print("Variancia:", round(variance_br, 4))
print("Desvio padrao: ", round(std_dev_br, 4))

#%% Escrevendo em arquivo

fields_stats_br = ['Version',
                   'Media',
                   'Mediana',
                   'Variancia',
                   'Desvio padrao',
                   'Max val',
                   'Min val',
                   '% Nan val']

rows_stats_br = {'Version':0,
                 'Media':mean_br,
                 'Mediana':median_br,
                 'Variancia':variance_br,
                 'Desvio padrao':std_dev_br,
                 'Max val':max_br,
                 'Min val':min_br,
                 '% Nan val': 100*per_br}

file_stats_br = "../tcc_codes/analise_stats/BR/Stats_BR.csv"

version_file(file_stats_br, fields_stats_br, rows_stats_br)

#%% Gaussiana com matplotlib da distribuição anterior
print("Matplotlib version: ",matplotlib.__version__)
print("Pandas version", pd.__version__)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

mu_br, std_br = norm.fit(labels_br) # Média e desvio padrão dos dados

# Histograma
plt.hist(labels_br, bins=150, density=True, alpha=0.0)
  
# Limites
min_ylim, max_ylim = plt.ylim(0,0.06)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando (Gaussiana)
p_br = norm.pdf(x_br, mu_br, std_br)

# Plot Gaussiana
plt.plot(x_br, p_br, 'k', linewidth=1.5)# Ref: 
plt.fill_between(x_br, p_br, color='mediumseagreen')# Ref: 
plt.axvline(labels_br.mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.text(labels_br.mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(labels_br.mean()),fontsize=10, style='italic', weight='bold')
plt.text(labels_br.mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(labels_br.std()),fontsize=10, style='italic', weight='bold')
plt.title("Distribuição de notas do Enade no Brasil: 2014 a 2018")
plt.xlabel('Notas do Enade');
plt.ylabel('Distribuição');
plt.savefig('../tcc_codes/analise_stats/BR/imagens/DIST_NOTA_BR.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);

#%% Get nas notas de Maior impacto
qe_i02 = dataset_br[["QE_I02", "NT_GER"]]
qe_i08 = dataset_br[["QE_I08", "NT_GER"]]
qe_i11 = dataset_br[["QE_I11", "NT_GER"]]
qe_i13 = dataset_br[["QE_I13", "NT_GER"]]
qe_i17 = dataset_br[["QE_I17", "NT_GER"]]
qe_i23 = dataset_br[["QE_I23", "NT_GER"]]
qe_i25 = dataset_br[["QE_I25", "NT_GER"]]
##### Get nas notas de menor impacto #####
qe_i01 = dataset_br[["QE_I01", "NT_GER"]]
qe_i03 = dataset_br[["QE_I03", "NT_GER"]]
qe_i06 = dataset_br[["QE_I06", "NT_GER"]]
qe_i12 = dataset_br[["QE_I12", "NT_GER"]]
qe_i15 = dataset_br[["QE_I15", "NT_GER"]]
qe_i16 = dataset_br[["QE_I16", "NT_GER"]]
qe_i19 = dataset_br[["QE_I19", "NT_GER"]]
qe_i20 = dataset_br[["QE_I20", "NT_GER"]]
qe_i21 = dataset_br[["QE_I21", "NT_GER"]]

#%% QE_I02
size_title = 18
size_subtitle = 12
fig_i02, axes_i02 = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i02.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                 'Dado socieconômico:Cor',
                 fontsize=size_title)


# Alternative A
qe_i02_aa = qe_i02.loc[(qe_i02['QE_I02'] == 'A')]

# Média e desvio padrão
mu_br_qei02_aa, std_br_qei02_aa = norm.fit(qe_i02_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_aa = norm.pdf(x_br, mu_br_qei02_aa, std_br_qei02_aa)

# Plot histogram
axes_i02[0,0].plot(x_br, p_br_qei02_aa, 'k', linewidth=1.5)
axes_i02[0,0].set_xlim([0,100])
axes_i02[0,0].fill_between(x_br, p_br_qei02_aa, color='mediumseagreen')
axes_i02[0,0].axvline(qe_i02_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[0,0].text(qe_i02_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_aa['NT_GER'].mean()), fontstyle='italic', weight='bold')
axes_i02[0,0].text(qe_i02_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[0,0].set_title("A:Branca", fontsize=size_subtitle, weight='bold')

# Plot Gaussiana
qe_i02_bb = qe_i02.loc[(qe_i02['QE_I02'] == 'B')]

mu_br_qei02_bb, std_br_qei02_bb = norm.fit(qe_i02_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_bb = norm.pdf(x_br, mu_br_qei02_bb, std_br_qei02_bb)

# Plot histogram
axes_i02[0,1].plot(x_br, p_br_qei02_bb, 'k', linewidth=1.5)
axes_i02[0,1].set_xlim([0,100])
axes_i02[0,1].fill_between(x_br, p_br_qei02_bb, color='mediumseagreen')
axes_i02[0,1].axvline(qe_i02_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[0,1].text(qe_i02_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i02[0,1].text(qe_i02_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[0,1].set_title("B:Preta", fontsize=size_subtitle, weight='bold')

# Alternative C
qe_i02_cc = qe_i02.loc[(qe_i02['QE_I02'] == 'C')]

mu_br_qei02_cc, std_br_qei02_cc = norm.fit(qe_i02_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_cc = norm.pdf(x_br, mu_br_qei02_cc, std_br_qei02_cc)

# Plot histogram
axes_i02[1,0].plot(x_br, p_br_qei02_cc, 'k', linewidth=1.5)
axes_i02[1,0].set_xlim([0,100])
axes_i02[1,0].fill_between(x_br, p_br_qei02_cc, color='mediumseagreen')
axes_i02[1,0].axvline(qe_i02_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,0].text(qe_i02_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i02[1,0].text(qe_i02_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[1,0].set_title("C:Amarela", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i02_dd = qe_i02.loc[(qe_i02['QE_I02'] == 'D')]

mu_br_qei02_dd, std_br_qei02_dd = norm.fit(qe_i02_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_dd = norm.pdf(x_br, mu_br_qei02_dd, std_br_qei02_dd)

# Plot histogram
axes_i02[1,1].plot(x_br, p_br_qei02_dd, 'k', linewidth=1.5)
axes_i02[1,1].set_xlim([0,100])
axes_i02[1,1].fill_between(x_br, p_br_qei02_dd, color='mediumseagreen')
axes_i02[1,1].axvline(qe_i02_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[1,1].text(qe_i02_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i02[1,1].text(qe_i02_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[1,1].set_title("D:Parda", fontsize=size_subtitle, weight='bold')

# Alternative E
qe_i02_ee = qe_i02.loc[(qe_i02['QE_I02'] == 'E')]

mu_br_qei02_ee, std_br_qei02_ee = norm.fit(qe_i02_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_ee = norm.pdf(x_br, mu_br_qei02_ee, std_br_qei02_ee)

# Plot histogram
axes_i02[2,0].plot(x_br, p_br_qei02_ee, 'k', linewidth=1.5)
axes_i02[2,0].set_xlim([0,100])
axes_i02[2,0].fill_between(x_br, p_br_qei02_ee, color='mediumseagreen')
axes_i02[2,0].axvline(qe_i02_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[2,0].text(qe_i02_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i02[2,0].text(qe_i02_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[2,0].set_title("E:Indígena", fontsize=size_subtitle, weight='bold')

# Alternative F
qe_i02_ff = qe_i02.loc[(qe_i02['QE_I02'] == 'F')]

mu_br_qei02_ff, std_br_qei02_ff = norm.fit(qe_i02_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei02_ff = norm.pdf(x_br, mu_br_qei02_ff, std_br_qei02_ff)

# Plot histogram
axes_i02[2,1].plot(x_br, p_br_qei02_ff, 'k', linewidth=1.5)
axes_i02[2,1].set_xlim([0,100])
axes_i02[2,1].fill_between(x_br, p_br_qei02_ff, color='mediumseagreen')
axes_i02[2,1].axvline(qe_i02_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i02[2,1].text(qe_i02_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i02_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i02[2,1].text(qe_i02_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i02_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i02[2,1].set_title("F:Não declarou", fontsize=size_subtitle, weight='bold')

axes_i02[0,0].set_ylabel('Distribuição')
axes_i02[1,0].set_ylabel('Distribuição')
axes_i02[2,0].set_ylabel('Distribuição')

for ax in axes_i02.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I02_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I08
fig_i08, axes_i08 = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i08.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                 'Dado socieconômico:Renda bruta',
                 fontsize=size_title)
# Alternative A
qe_i08_aa = qe_i08.loc[(qe_i08['QE_I08'] == 'A')]

# Média e desvio padrão
mu_br_qei08_aa, std_br_qei08_aa = norm.fit(qe_i08_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_aa = norm.pdf(x_br, mu_br_qei08_aa, std_br_qei08_aa)

# Plot histogram
axes_i08[0,0].plot(x_br, p_br_qei08_aa, 'k', linewidth=1.5)
axes_i08[0,0].set_xlim([0,100])
axes_i08[0,0].fill_between(x_br, p_br_qei08_aa, color='mediumseagreen')
axes_i08[0,0].axvline(qe_i08_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[0,0].text(qe_i08_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[0,0].set_title("A:Até R$ 1.431,00", fontsize=size_subtitle, weight='bold')

# Alternative B
qe_i08_bb = qe_i08.loc[(qe_i08['QE_I08'] == 'B')]

mu_br_qei08_bb, std_br_qei08_bb = norm.fit(qe_i08_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_bb = norm.pdf(x_br, mu_br_qei08_bb, std_br_qei08_bb)

# Plot histogram
axes_i08[0,1].plot(x_br, p_br_qei08_bb, 'k', linewidth=1.5)
axes_i08[0,1].set_xlim([0,100])
axes_i08[0,1].fill_between(x_br, p_br_qei08_bb, color='mediumseagreen')
axes_i08[0,1].axvline(qe_i08_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,1].text(qe_i08_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[0,1].text(qe_i08_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[0,1].set_title("B:De R\$ 1.431,01 a R\$ 2.862,00", fontsize=size_subtitle, weight='bold')

# Alternative C
qe_i08_cc = qe_i08.loc[(qe_i08['QE_I08'] == 'C')]

mu_br_qei08_cc, std_br_qei08_cc = norm.fit(qe_i08_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_cc = norm.pdf(x_br, mu_br_qei08_cc, std_br_qei08_cc)

# Plot histogram
axes_i08[0,2].plot(x_br, p_br_qei08_cc, 'k', linewidth=1.5)
axes_i08[0,2].set_xlim([0,100])
axes_i08[0,2].fill_between(x_br, p_br_qei08_cc, color='mediumseagreen')
axes_i08[0,2].axvline(qe_i08_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[0,2].text(qe_i08_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[0,2].text(qe_i08_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[0,2].set_title("C:De R\$ 2.862,01 a R\$ 4.293,00", fontsize=size_subtitle, weight='bold')

# Alternative D
qe_i08_dd = qe_i08.loc[(qe_i08['QE_I08'] == 'D')]

mu_br_qei08_dd, std_br_qei08_dd = norm.fit(qe_i08_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_dd = norm.pdf(x_br, mu_br_qei08_dd, std_br_qei08_dd)

# Plot histogram
axes_i08[1,0].plot(x_br, p_br_qei08_dd, 'k', linewidth=1.5)
axes_i08[1,0].set_xlim([0,100])
axes_i08[1,0].fill_between(x_br, p_br_qei08_dd, color='mediumseagreen')
axes_i08[1,0].axvline(qe_i08_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,0].text(qe_i08_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[1,0].text(qe_i08_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[1,0].set_title("D:De R\$ 4.293,01 a R\$ 5.724,00", fontsize=size_subtitle, weight='bold')

# Alternative E
qe_i08_ee = qe_i08.loc[(qe_i08['QE_I08'] == 'E')]

mu_br_qei08_ee, std_br_qei08_ee = norm.fit(qe_i08_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_ee = norm.pdf(x_br, mu_br_qei08_ee, std_br_qei08_ee)

# Plot histogram
axes_i08[1,1].plot(x_br, p_br_qei08_ee, 'k', linewidth=1.5)
axes_i08[1,1].set_xlim([0,100])
axes_i08[1,1].fill_between(x_br, p_br_qei08_ee, color='mediumseagreen')
axes_i08[1,1].axvline(qe_i08_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,1].text(qe_i08_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[1,1].text(qe_i08_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[1,1].set_title("E:De R\$ 5.274,01 a R\$ 9.540,00", fontsize=size_subtitle, weight='bold')

# Alternative F
qe_i08_ff = qe_i08.loc[(qe_i08['QE_I08'] == 'F')]

mu_br_qei08_ff, std_br_qei08_ff = norm.fit(qe_i08_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_ff = norm.pdf(x_br, mu_br_qei08_ff, std_br_qei08_ff)

# Plot histogram
axes_i08[1,2].plot(x_br, p_br_qei08_ff, 'k', linewidth=1.5)
axes_i08[1,2].set_xlim([0,100])
axes_i08[1,2].fill_between(x_br, p_br_qei08_ff, color='mediumseagreen')
axes_i08[1,2].axvline(qe_i08_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[1,2].text(qe_i08_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[1,2].text(qe_i08_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[1,2].set_title("F:De R\$ 9.540,01 a R\$ 28.620,00", fontsize=size_subtitle, weight='bold')

# Alternative G
qe_i08_gg = qe_i08.loc[(qe_i08['QE_I08'] == 'G')]

mu_br_qei08_gg, std_br_qei08_gg = norm.fit(qe_i08_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei08_gg = norm.pdf(x_br, mu_br_qei08_gg, std_br_qei08_gg)

# Plot histogram
axes_i08[2,0].plot(x_br, p_br_qei08_gg, 'k', linewidth=1.5)
axes_i08[2,0].set_xlim([0,100])
axes_i08[2,0].fill_between(x_br, p_br_qei08_gg, color='mediumseagreen')
axes_i08[2,0].axvline(qe_i08_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i08[2,0].text(qe_i08_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i08_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i08[2,0].text(qe_i08_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i08_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i08[2,0].set_title("G:Mais de R$ 28.620,00", fontsize=size_subtitle, weight='bold')

axes_i08[2,1].axis('off')
axes_i08[2,2].axis('off')

axes_i08[0,0].set_ylabel('Distribuição')
axes_i08[1,0].set_ylabel('Distribuição')
axes_i08[2,0].set_ylabel('Distribuição')

for ax in axes_i08.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/maior_impacto/QE_I08_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I11A
size_title = 18
size_subtitle = 14
fig_i11a, axes_i11a = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                   sharex=False, sharey=True, figsize=(15,15))
fig_i11a.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socieconômico:Tipo de financiamento p/ custeio das mensalidades',
                 fontsize=size_title)


####### Alternative A #######
qe_i11a_aa = qe_i11.loc[(qe_i11['QE_I11'] == 'A')]

# Média e desvio padrão
mu_br_qei11a_aa, std_br_qei11a_aa = norm.fit(qe_i11a_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_aa = norm.pdf(x_br, mu_br_qei11a_aa, std_br_qei11a_aa)

# Plot histogram
axes_i11a[0,0].plot(x_br, p_br_qei11a_aa, 'k', linewidth=1.5)
axes_i11a[0,0].set_xlim([0,100])
axes_i11a[0,0].fill_between(x_br, p_br_qei11a_aa, color='mediumseagreen')
axes_i11a[0,0].axvline(qe_i11a_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[0,0].text(qe_i11a_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[0,0].text(qe_i11a_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[0,0].set_title("A:Nenhum;curso gratuito", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i11a_bb = qe_i11.loc[(qe_i11['QE_I11'] == 'B')]

mu_br_qei11a_bb, std_br_qei11a_bb = norm.fit(qe_i11a_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_bb = norm.pdf(x_br, mu_br_qei11a_bb, std_br_qei11a_bb)

# Plot histogram
axes_i11a[0,1].plot(x_br, p_br_qei11a_bb, 'k', linewidth=1.5)
axes_i11a[0,1].set_xlim([0,100])
axes_i11a[0,1].fill_between(x_br, p_br_qei11a_bb, color='mediumseagreen')
axes_i11a[0,1].axvline(qe_i11a_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[0,1].text(qe_i11a_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[0,1].text(qe_i11a_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[0,1].set_title("B:Nenhum;curso pago", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i11a_cc = qe_i11.loc[(qe_i11['QE_I11'] == 'C')]

mu_br_qei11a_cc, std_br_qei11a_cc = norm.fit(qe_i11a_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_cc = norm.pdf(x_br, mu_br_qei11a_cc, std_br_qei11a_cc)

# Plot histogram
axes_i11a[1,0].plot(x_br, p_br_qei11a_cc, 'k', linewidth=1.5)
axes_i11a[1,0].set_xlim([0,100])
axes_i11a[1,0].fill_between(x_br, p_br_qei11a_cc, color='mediumseagreen')
axes_i11a[1,0].axvline(qe_i11a_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[1,0].text(qe_i11a_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[1,0].text(qe_i11a_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[1,0].set_title("C:ProUni integral", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i11a_dd = qe_i11.loc[(qe_i11['QE_I11'] == 'D')]

mu_br_qei11a_dd, std_br_qei11a_dd = norm.fit(qe_i11a_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_dd = norm.pdf(x_br, mu_br_qei11a_dd, std_br_qei11a_dd)

# Plot histogram
axes_i11a[1,1].plot(x_br, p_br_qei11a_dd, 'k', linewidth=1.5)
axes_i11a[1,1].set_xlim([0,100])
axes_i11a[1,1].fill_between(x_br, p_br_qei11a_dd, color='mediumseagreen')
axes_i11a[1,1].axvline(qe_i11a_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[1,1].text(qe_i11a_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[1,1].text(qe_i11a_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[1,1].set_title("D:ProUni parcial,apenas", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i11a_ee = qe_i11.loc[(qe_i11['QE_I11'] == 'E')]

mu_br_qei11a_ee, std_br_qei11a_ee = norm.fit(qe_i11a_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_ee = norm.pdf(x_br, mu_br_qei11a_ee, std_br_qei11a_ee)

# Plot histogram
axes_i11a[2,0].plot(x_br, p_br_qei11a_ee, 'k', linewidth=1.5)
axes_i11a[2,0].set_xlim([0,100])
axes_i11a[2,0].fill_between(x_br, p_br_qei11a_ee, color='mediumseagreen')
axes_i11a[2,0].axvline(qe_i11a_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[2,0].text(qe_i11a_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[2,0].text(qe_i11a_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[2,0].set_title("E:FIES,apenas", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i11a_ff = qe_i11.loc[(qe_i11['QE_I11'] == 'F')]

mu_br_qei11a_ff, std_br_qei11a_ff = norm.fit(qe_i11a_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11a_ff = norm.pdf(x_br, mu_br_qei11a_ff, std_br_qei11a_ff)

# Plot histogram
axes_i11a[2,1].plot(x_br, p_br_qei11a_ff, 'k', linewidth=1.5)
axes_i11a[2,1].set_xlim([0,100])
axes_i11a[2,1].fill_between(x_br, p_br_qei11a_ff, color='mediumseagreen')
axes_i11a[2,1].axvline(qe_i11a_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11a[2,1].text(qe_i11a_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11a_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11a[2,1].text(qe_i11a_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11a_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11a[2,1].set_title("F:ProUni parcial e FIES", fontsize=size_subtitle, weight='bold')

axes_i11a[0,0].set_ylabel('Distribuição')
axes_i11a[1,0].set_ylabel('Distribuição')
axes_i11a[2,0].set_ylabel('Distribuição')

for ax in axes_i11a.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/maior_impacto/QE_I11A_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I11B
fig_i11b, axes_i11b = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                   sharex=False, sharey=True, figsize=(15,15))
fig_i11b.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                  'Dado socieconômico:Tipo de financiamento para custeio das mensalidades',
                 fontsize=size_title)
####### Alternative G #######
qe_i11b_gg = qe_i11.loc[(qe_i11['QE_I11'] == 'G')]

# Média e desvio padrão
mu_br_qei11b_gg, std_br_qei11b_gg = norm.fit(qe_i11b_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11b_gg = norm.pdf(x_br, mu_br_qei11b_gg, std_br_qei11b_gg)

# Plot histogram
axes_i11b[0,0].plot(x_br, p_br_qei11b_gg, 'k', linewidth=1.5)
axes_i11b[0,0].set_xlim([0,100])
axes_i11b[0,0].fill_between(x_br, p_br_qei11b_gg, color='mediumseagreen')
axes_i11b[0,0].axvline(qe_i11b_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[0,0].text(qe_i11b_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11b[0,0].text(qe_i11b_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11b[0,0].set_title("G:Bolsa pelo estado,governo ou município", fontsize=size_subtitle, weight='bold')

####### Alternative H #######
qe_i11b_hh = qe_i11.loc[(qe_i11['QE_I11'] == 'H')]

mu_br_qei11b_hh, std_br_qei11b_hh = norm.fit(qe_i11b_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11b_hh = norm.pdf(x_br, mu_br_qei11b_hh, std_br_qei11b_hh)

# Plot histogram
axes_i11b[0,1].plot(x_br, p_br_qei11b_hh, 'k', linewidth=1.5)
axes_i11b[0,1].set_xlim([0,100])
axes_i11b[0,1].fill_between(x_br, p_br_qei11b_hh, color='mediumseagreen')
axes_i11b[0,1].axvline(qe_i11b_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[0,1].text(qe_i11b_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11b[0,1].text(qe_i11b_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11b[0,1].set_title("H:Bolsa pela IES", fontsize=size_subtitle, weight='bold')

####### Alternative I #######
qe_i11b_ii = qe_i11.loc[(qe_i11['QE_I11'] == 'I')]

mu_br_qei11b_ii, std_br_qei11b_ii = norm.fit(qe_i11b_ii['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11b_ii = norm.pdf(x_br, mu_br_qei11b_ii, std_br_qei11b_ii)

# Plot histogram
axes_i11b[1,0].plot(x_br, p_br_qei11b_ii, 'k', linewidth=1.5)
axes_i11b[1,0].set_xlim([0,100])
axes_i11b[1,0].fill_between(x_br, p_br_qei11b_ii, color='mediumseagreen')
axes_i11b[1,0].axvline(qe_i11b_ii['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[1,0].text(qe_i11b_ii['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_ii['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11b[1,0].text(qe_i11b_ii['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_ii['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11b[1,0].set_title("I:Bolsa por outra entidade", fontsize=size_subtitle, weight='bold')

####### Alternative J #######
qe_i11b_jj = qe_i11.loc[(qe_i11['QE_I11'] == 'J')]

mu_br_qei11b_jj, std_br_qei11b_jj = norm.fit(qe_i11b_jj['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11b_jj = norm.pdf(x_br, mu_br_qei11b_jj, std_br_qei11b_jj)

# Plot histogram
axes_i11b[1,1].plot(x_br, p_br_qei11b_jj, 'k', linewidth=1.5)
axes_i11b[1,1].set_xlim([0,100])
axes_i11b[1,1].fill_between(x_br, p_br_qei11b_jj, color='mediumseagreen')
axes_i11b[1,1].axvline(qe_i11b_jj['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[1,1].text(qe_i11b_jj['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_jj['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11b[1,1].text(qe_i11b_jj['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_jj['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11b[1,1].set_title("J:Financiamento pela IES", fontsize=size_subtitle, weight='bold')

####### Alternative K #######
qe_i11b_kk = qe_i11.loc[(qe_i11['QE_I11'] == 'K')]

mu_br_qei11b_kk, std_br_qei11b_kk = norm.fit(qe_i11b_kk['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei11b_kk = norm.pdf(x_br, mu_br_qei11b_kk, std_br_qei11b_kk)

# Plot histogram
axes_i11b[2,0].plot(x_br, p_br_qei11b_kk, 'k', linewidth=1.5)
axes_i11b[2,0].set_xlim([0,100])
axes_i11b[2,0].fill_between(x_br, p_br_qei11b_kk, color='mediumseagreen')
axes_i11b[2,0].axvline(qe_i11b_kk['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i11b[2,0].text(qe_i11b_kk['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i11b_kk['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i11b[2,0].text(qe_i11b_kk['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i11b_kk['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i11b[2,0].set_title("K:Financiamento bancário", fontsize=size_subtitle, weight='bold')

axes_i11b[2,1].axis('off')

axes_i11b[0,0].set_ylabel('Distribuição')
axes_i11b[1,0].set_ylabel('Distribuição')
axes_i11b[2,0].set_ylabel('Distribuição')

for ax in axes_i11b.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/maior_impacto/QE_I11B_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()


#%% QE_13
fig_i13, axes_i13 = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i13.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de bolsa acadêmica durante a graduação',
                 fontsize=size_title)

####### Alternative A #######
qe_i13_aa = qe_i13.loc[(qe_i13['QE_I13'] == 'A')]

# Média e desvio padrão
mu_br_qei13_aa, std_br_qei13_aa = norm.fit(qe_i13_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_aa = norm.pdf(x_br, mu_br_qei13_aa, std_br_qei13_aa)

# Plot histogram
axes_i13[0,0].plot(x_br, p_br_qei13_aa, 'k', linewidth=1.5)
axes_i13[0,0].set_xlim([0,100])
axes_i13[0,0].fill_between(x_br, p_br_qei13_aa, color='mediumseagreen')
axes_i13[0,0].axvline(qe_i13_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[0,0].text(qe_i13_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[0,0].text(qe_i13_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[0,0].set_title("A:Nenhum", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i13_bb = qe_i13.loc[(qe_i13['QE_I13'] == 'B')]

mu_br_qei13_bb, std_br_qei13_bb = norm.fit(qe_i13_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_bb = norm.pdf(x_br, mu_br_qei13_bb, std_br_qei13_bb)

# Plot histogram
axes_i13[0,1].plot(x_br, p_br_qei13_bb, 'k', linewidth=1.5)
axes_i13[0,1].set_xlim([0,100])
axes_i13[0,1].fill_between(x_br, p_br_qei13_bb, color='mediumseagreen')
axes_i13[0,1].axvline(qe_i13_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[0,1].text(qe_i13_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[0,1].text(qe_i13_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[0,1].set_title("B:PIBIC", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i13_cc = qe_i13.loc[(qe_i13['QE_I13'] == 'C')]

mu_br_qei13_cc, std_br_qei13_cc = norm.fit(qe_i13_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_cc = norm.pdf(x_br, mu_br_qei13_cc, std_br_qei13_cc)

# Plot histogram
axes_i13[1,0].plot(x_br, p_br_qei13_cc, 'k', linewidth=1.5)
axes_i13[1,0].set_xlim([0,100])
axes_i13[1,0].fill_between(x_br, p_br_qei13_cc, color='mediumseagreen')
axes_i13[1,0].axvline(qe_i13_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[1,0].text(qe_i13_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[1,0].text(qe_i13_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[1,0].set_title("C:Extensão", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i13_dd = qe_i13.loc[(qe_i13['QE_I13'] == 'D')]

mu_br_qei13_dd, std_br_qei13_dd = norm.fit(qe_i13_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_dd = norm.pdf(x_br, mu_br_qei13_dd, std_br_qei13_dd)

# Plot histogram
axes_i13[1,1].plot(x_br, p_br_qei13_dd, 'k', linewidth=1.5)
axes_i13[1,1].set_xlim([0,100])
axes_i13[1,1].fill_between(x_br, p_br_qei13_dd, color='mediumseagreen')
axes_i13[1,1].axvline(qe_i13_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[1,1].text(qe_i13_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[1,1].text(qe_i13_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[1,1].set_title("D:Monitoria/tutoria", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i13_ee = qe_i13.loc[(qe_i13['QE_I13'] == 'E')]

mu_br_qei13_ee, std_br_qei13_ee = norm.fit(qe_i13_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_ee = norm.pdf(x_br, mu_br_qei13_ee, std_br_qei13_ee)

# Plot histogram
axes_i13[2,0].plot(x_br, p_br_qei13_ee, 'k', linewidth=1.5)
axes_i13[2,0].set_xlim([0,100])
axes_i13[2,0].fill_between(x_br, p_br_qei13_ee, color='mediumseagreen')
axes_i13[2,0].axvline(qe_i13_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[2,0].text(qe_i13_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[2,0].text(qe_i13_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[2,0].set_title("E:PET", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i13_ff = qe_i13.loc[(qe_i13['QE_I13'] == 'F')]

mu_br_qei13_ff, std_br_qei13_ff = norm.fit(qe_i13_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei13_ff = norm.pdf(x_br, mu_br_qei13_ff, std_br_qei13_ff)

# Plot histogram
axes_i13[2,1].plot(x_br, p_br_qei13_ff, 'k', linewidth=1.5)
axes_i13[2,1].set_xlim([0,100])
axes_i13[2,1].fill_between(x_br, p_br_qei13_ff, color='mediumseagreen')
axes_i13[2,1].axvline(qe_i13_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i13[2,1].text(qe_i13_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i13_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i13[2,1].text(qe_i13_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i13_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i13[2,1].set_title("F:Outro", fontsize=size_subtitle, weight='bold')

axes_i13[0,0].set_ylabel('Distribuição')
axes_i13[1,0].set_ylabel('Distribuição')
axes_i13[2,0].set_ylabel('Distribuição')

for ax in axes_i13.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/maior_impacto/QE_I13_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I17
fig_i17, axes_i17 = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i17.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de escola que cursou o ensino médio',
                 fontsize=size_title)


####### Alternative A #######
qe_i17_aa = qe_i17.loc[(qe_i17['QE_I17'] == 'A')]

# Média e desvio padrão
mu_br_qei17_aa, std_br_qei17_aa = norm.fit(qe_i17_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_aa = norm.pdf(x_br, mu_br_qei17_aa, std_br_qei17_aa)

# Plot histogram
axes_i17[0,0].plot(x_br, p_br_qei17_aa, 'k', linewidth=1.5)

axes_i17[0,0].fill_between(x_br, p_br_qei17_aa, color='mediumseagreen')
axes_i17[0,0].axvline(qe_i17_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[0,0].text(qe_i17_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[0,0].text(qe_i17_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[0,0].set_title("A:Todo em pública", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i17_bb = qe_i17.loc[(qe_i17['QE_I17'] == 'B')]

mu_br_qei17_bb, std_br_qei17_bb = norm.fit(qe_i17_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_bb = norm.pdf(x_br, mu_br_qei17_bb, std_br_qei17_bb)

# Plot histogram
axes_i17[0,1].plot(x_br, p_br_qei17_bb, 'k', linewidth=1.5)

axes_i17[0,1].fill_between(x_br, p_br_qei17_bb, color='mediumseagreen')
axes_i17[0,1].axvline(qe_i17_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[0,1].text(qe_i17_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[0,1].text(qe_i17_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[0,1].set_title("B:Todo em particular", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i17_cc = qe_i17.loc[(qe_i17['QE_I17'] == 'C')]

mu_br_qei17_cc, std_br_qei17_cc = norm.fit(qe_i17_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_cc = norm.pdf(x_br, mu_br_qei17_cc, std_br_qei17_cc)

# Plot histogram
axes_i17[1,0].plot(x_br, p_br_qei17_cc, 'k', linewidth=1.5)

axes_i17[1,0].fill_between(x_br, p_br_qei17_cc, color='mediumseagreen')
axes_i17[1,0].axvline(qe_i17_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[1,0].text(qe_i17_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[1,0].text(qe_i17_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[1,0].set_title("C:Todo no exterior", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i17_dd = qe_i17.loc[(qe_i17['QE_I17'] == 'D')]

mu_br_qei17_dd, std_br_qei17_dd = norm.fit(qe_i17_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_dd = norm.pdf(x_br, mu_br_qei17_dd, std_br_qei17_dd)

# Plot histogram
axes_i17[1,1].plot(x_br, p_br_qei17_dd, 'k', linewidth=1.5)
axes_i17[1,1].fill_between(x_br, p_br_qei17_dd, color='mediumseagreen')
axes_i17[1,1].axvline(qe_i17_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[1,1].text(qe_i17_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[1,1].text(qe_i17_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[1,1].set_title("D:Maior parte em pública", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i17_ee = qe_i17.loc[(qe_i17['QE_I17'] == 'E')]

mu_br_qei17_ee, std_br_qei17_ee = norm.fit(qe_i17_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_ee = norm.pdf(x_br, mu_br_qei17_ee, std_br_qei17_ee)

# Plot histogram
axes_i17[2,0].plot(x_br, p_br_qei17_ee, 'k', linewidth=1.5)

axes_i17[2,0].fill_between(x_br, p_br_qei17_ee, color='mediumseagreen')
axes_i17[2,0].axvline(qe_i17_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[2,0].text(qe_i17_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[2,0].text(qe_i17_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[2,0].set_title("E:Maior parte em particular", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i17_ff = qe_i17.loc[(qe_i17['QE_I17'] == 'F')]

mu_br_qei17_ff, std_br_qei17_ff = norm.fit(qe_i17_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.4)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei17_ff = norm.pdf(x_br, mu_br_qei17_ff, std_br_qei17_ff)

# Plot histogram
axes_i17[2,1].plot(x_br, p_br_qei17_ff, 'k', linewidth=1.5)

axes_i17[2,1].fill_between(x_br, p_br_qei17_ff, color='mediumseagreen')
axes_i17[2,1].axvline(qe_i17_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i17[2,1].text(qe_i17_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i17_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i17[2,1].text(qe_i17_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i17_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i17[2,1].set_title("F:Brasil e exterior", fontsize=size_subtitle, weight='bold')

axes_i17[0,0].set_ylabel('Distribuição')
axes_i17[1,0].set_ylabel('Distribuição')
axes_i17[2,0].set_ylabel('Distribuição')

for ax in axes_i17.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/maior_impacto/QE_I17_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I23
fig_i23, axes_i23 = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i23.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Horas de estudo por semana (excluindo aulas)',
                 fontsize=size_title)


####### Alternative A #######
qe_i23_aa = qe_i23.loc[(qe_i23['QE_I23'] == 'A')]

# Média e desvio padrão
mu_br_qei23_aa, std_br_qei23_aa = norm.fit(qe_i23_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei23_aa = norm.pdf(x_br, mu_br_qei23_aa, std_br_qei23_aa)

# Plot histogram
axes_i23[0,0].plot(x_br, p_br_qei23_aa, 'k', linewidth=1.5)

axes_i23[0,0].fill_between(x_br, p_br_qei23_aa, color='mediumseagreen')
axes_i23[0,0].axvline(qe_i23_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[0,0].text(qe_i23_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i23[0,0].text(qe_i23_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i23[0,0].set_title("A:Nenhuma", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i23_bb = qe_i23.loc[(qe_i23['QE_I23'] == 'B')]

mu_br_qei23_bb, std_br_qei23_bb = norm.fit(qe_i23_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei23_bb = norm.pdf(x_br, mu_br_qei23_bb, std_br_qei23_bb)

# Plot histogram
axes_i23[0,1].plot(x_br, p_br_qei23_bb, 'k', linewidth=1.5)

axes_i23[0,1].fill_between(x_br, p_br_qei23_bb, color='mediumseagreen')
axes_i23[0,1].axvline(qe_i23_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[0,1].text(qe_i23_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i23[0,1].text(qe_i23_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i23[0,1].set_title("B:Uma a três", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i23_cc = qe_i23.loc[(qe_i23['QE_I23'] == 'C')]

mu_br_qei23_cc, std_br_qei23_cc = norm.fit(qe_i23_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei23_cc = norm.pdf(x_br, mu_br_qei23_cc, std_br_qei23_cc)

# Plot histogram
axes_i23[1,0].plot(x_br, p_br_qei23_cc, 'k', linewidth=1.5)

axes_i23[1,0].fill_between(x_br, p_br_qei23_cc, color='mediumseagreen')
axes_i23[1,0].axvline(qe_i23_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[1,0].text(qe_i23_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i23[1,0].text(qe_i23_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i23[1,0].set_title("C:Quatro a sete", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i23_dd = qe_i23.loc[(qe_i23['QE_I23'] == 'D')]

mu_br_qei23_dd, std_br_qei23_dd = norm.fit(qe_i23_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei23_dd = norm.pdf(x_br, mu_br_qei23_dd, std_br_qei23_dd)

# Plot histogram
axes_i23[1,1].plot(x_br, p_br_qei23_dd, 'k', linewidth=1.5)
axes_i23[1,1].fill_between(x_br, p_br_qei23_dd, color='mediumseagreen')
axes_i23[1,1].axvline(qe_i23_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[1,1].text(qe_i23_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i23[1,1].text(qe_i23_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i23[1,1].set_title("D:Oito a doze", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i23_ee = qe_i23.loc[(qe_i23['QE_I23'] == 'E')]

mu_br_qei23_ee, std_br_qei23_ee = norm.fit(qe_i23_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei23_ee = norm.pdf(x_br, mu_br_qei23_ee, std_br_qei23_ee)

# Plot histogram
axes_i23[2,0].plot(x_br, p_br_qei23_ee, 'k', linewidth=1.5)

axes_i23[2,0].fill_between(x_br, p_br_qei23_ee, color='mediumseagreen')
axes_i23[2,0].axvline(qe_i23_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i23[2,0].text(qe_i23_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i23_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i23[2,0].text(qe_i23_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i23_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i23[2,0].set_title("E:Mais de doze", fontsize=size_subtitle, weight='bold')

axes_i23[2,1].axis('off')

axes_i23[0,0].set_ylabel('Distribuição')
axes_i23[1,0].set_ylabel('Distribuição')
axes_i23[2,0].set_ylabel('Distribuição')

for ax in axes_i23.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR//imagens/maior_impacto/QE_I23_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I25
fig_i25, axes_i25 = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i25.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Principal motivo de escolha do curso',
                 fontsize=size_title)


####### Alternative A #######
qe_i25_aa = qe_i25.loc[(qe_i25['QE_I25'] == 'A')]

# Média e desvio padrão
mu_br_qei25_aa, std_br_qei25_aa = norm.fit(qe_i25_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_aa = norm.pdf(x_br, mu_br_qei25_aa, std_br_qei25_aa)

# Plot histogram
axes_i25[0,0].plot(x_br, p_br_qei25_aa, 'k', linewidth=1.5)

axes_i25[0,0].fill_between(x_br, p_br_qei25_aa, color='mediumseagreen')
axes_i25[0,0].axvline(qe_i25_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[0,0].text(qe_i25_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[0,0].text(qe_i25_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[0,0].set_title("A:Inserção no mercado de trabalho", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i25_bb = qe_i25.loc[(qe_i25['QE_I25'] == 'B')]

mu_br_qei25_bb, std_br_qei25_bb = norm.fit(qe_i25_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_bb = norm.pdf(x_br, mu_br_qei25_bb, std_br_qei25_bb)

# Plot histogram
axes_i25[0,1].plot(x_br, p_br_qei25_bb, 'k', linewidth=1.5)

axes_i25[0,1].fill_between(x_br, p_br_qei25_bb, color='mediumseagreen')
axes_i25[0,1].axvline(qe_i25_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[0,1].text(qe_i25_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[0,1].text(qe_i25_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[0,1].set_title("B:Influência familiar", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i25_cc = qe_i25.loc[(qe_i25['QE_I25'] == 'C')]

mu_br_qei25_cc, std_br_qei25_cc = norm.fit(qe_i25_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_cc = norm.pdf(x_br, mu_br_qei25_cc, std_br_qei25_cc)

# Plot histogram
axes_i25[0,2].plot(x_br, p_br_qei25_cc, 'k', linewidth=1.5)

axes_i25[0,2].fill_between(x_br, p_br_qei25_cc, color='mediumseagreen')
axes_i25[0,2].axvline(qe_i25_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[0,2].text(qe_i25_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[0,2].text(qe_i25_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[0,2].set_title("C:Valorização profissional", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i25_dd = qe_i25.loc[(qe_i25['QE_I25'] == 'D')]

mu_br_qei25_dd, std_br_qei25_dd = norm.fit(qe_i25_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_dd = norm.pdf(x_br, mu_br_qei25_dd, std_br_qei25_dd)

# Plot histogram
axes_i25[1,0].plot(x_br, p_br_qei25_dd, 'k', linewidth=1.5)
axes_i25[1,0].fill_between(x_br, p_br_qei25_dd, color='mediumseagreen')
axes_i25[1,0].axvline(qe_i25_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[1,0].text(qe_i25_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[1,0].text(qe_i25_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[1,0].set_title("D:Prestígio social", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i25_ee = qe_i25.loc[(qe_i25['QE_I25'] == 'E')]

mu_br_qei25_ee, std_br_qei25_ee = norm.fit(qe_i25_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_ee = norm.pdf(x_br, mu_br_qei25_ee, std_br_qei25_ee)

# Plot histogram
axes_i25[1,1].plot(x_br, p_br_qei25_ee, 'k', linewidth=1.5)

axes_i25[1,1].fill_between(x_br, p_br_qei25_ee, color='mediumseagreen')
axes_i25[1,1].axvline(qe_i25_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[1,1].text(qe_i25_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[1,1].text(qe_i25_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[1,1].set_title("E:Vocação", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i25_ff = qe_i25.loc[(qe_i25['QE_I25'] == 'F')]

mu_br_qei25_ff, std_br_qei25_ff = norm.fit(qe_i25_ff['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_ff = norm.pdf(x_br, mu_br_qei25_ff, std_br_qei25_ff)

# Plot histogram
axes_i25[1,2].plot(x_br, p_br_qei25_ff, 'k', linewidth=1.5)

axes_i25[1,2].fill_between(x_br, p_br_qei25_ff, color='mediumseagreen')
axes_i25[1,2].axvline(qe_i25_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[1,2].text(qe_i25_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[1,2].text(qe_i25_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[1,2].set_title("F:Porque é EAD", fontsize=size_subtitle, weight='bold')

####### Alternative G #######
qe_i25_gg = qe_i25.loc[(qe_i25['QE_I25'] == 'G')]

mu_br_qei25_gg, std_br_qei25_gg = norm.fit(qe_i25_gg['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_gg = norm.pdf(x_br, mu_br_qei25_gg, std_br_qei25_gg)

# Plot histogram
axes_i25[2,0].plot(x_br, p_br_qei25_gg, 'k', linewidth=1.5)

axes_i25[2,0].fill_between(x_br, p_br_qei25_gg, color='mediumseagreen')
axes_i25[2,0].axvline(qe_i25_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[2,0].text(qe_i25_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[2,0].text(qe_i25_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[2,0].set_title("H:Outro", fontsize=size_subtitle, weight='bold')

####### Alternative H #######
qe_i25_hh = qe_i25.loc[(qe_i25['QE_I25'] == 'H')]

mu_br_qei25_hh, std_br_qei25_hh = norm.fit(qe_i25_hh['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei25_hh = norm.pdf(x_br, mu_br_qei25_hh, std_br_qei25_hh)

# Plot histogram
axes_i25[1,1].plot(x_br, p_br_qei25_hh, 'k', linewidth=1.5)

axes_i25[1,1].fill_between(x_br, p_br_qei25_hh, color='mediumseagreen')
axes_i25[1,1].axvline(qe_i25_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i25[1,1].text(qe_i25_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i25_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i25[1,1].text(qe_i25_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i25_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i25[1,1].set_title("E:Vocação", fontsize=size_subtitle, weight='bold')

axes_i25[2,2].axis('off')

axes_i25[0,0].set_ylabel('Distribuição')
axes_i25[1,0].set_ylabel('Distribuição')
axes_i25[2,0].set_ylabel('Distribuição')

for ax in axes_i25.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR//imagens/maior_impacto/QE_I25_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% 
############################## Subplots - Menor impacto ##############################
############################## QE_I01 ##############################
fig_i01, axes_i01 = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(15,15))
fig_i01.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Estado civil',
                 fontsize=size_title)


####### Alternative A #######
qe_i01_aa = qe_i01.loc[(qe_i01['QE_I01'] == 'A')]

# Média e desvio padrão
mu_br_qei01_aa, std_br_qei01_aa = norm.fit(qe_i01_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei01_aa = norm.pdf(x_br, mu_br_qei01_aa, std_br_qei01_aa)

# Plot histogram
axes_i01[0,0].plot(x_br, p_br_qei01_aa, 'k', linewidth=1.5)

axes_i01[0,0].fill_between(x_br, p_br_qei01_aa, color='mediumseagreen')
axes_i01[0,0].axvline(qe_i01_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[0,0].text(qe_i01_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i01[0,0].text(qe_i01_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i01[0,0].set_title("A:Solteiro(a)", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i01_bb = qe_i01.loc[(qe_i01['QE_I01'] == 'B')]

mu_br_qei01_bb, std_br_qei01_bb = norm.fit(qe_i01_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei01_bb = norm.pdf(x_br, mu_br_qei01_bb, std_br_qei01_bb)

# Plot histogram
axes_i01[0,1].plot(x_br, p_br_qei01_bb, 'k', linewidth=1.5)

axes_i01[0,1].fill_between(x_br, p_br_qei01_bb, color='mediumseagreen')
axes_i01[0,1].axvline(qe_i01_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[0,1].text(qe_i01_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i01[0,1].text(qe_i01_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i01[0,1].set_title("B:Casado(a)", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i01_cc = qe_i01.loc[(qe_i01['QE_I01'] == 'C')]

mu_br_qei01_cc, std_br_qei01_cc = norm.fit(qe_i01_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei01_cc = norm.pdf(x_br, mu_br_qei01_cc, std_br_qei01_cc)

# Plot histogram
axes_i01[1,0].plot(x_br, p_br_qei01_cc, 'k', linewidth=1.5)

axes_i01[1,0].fill_between(x_br, p_br_qei01_cc, color='mediumseagreen')
axes_i01[1,0].axvline(qe_i01_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[1,0].text(qe_i01_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i01[1,0].text(qe_i01_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i01[1,0].set_title("C:Separado(a)", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i01_dd = qe_i01.loc[(qe_i01['QE_I01'] == 'D')]

mu_br_qei01_dd, std_br_qei01_dd = norm.fit(qe_i01_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei01_dd = norm.pdf(x_br, mu_br_qei01_dd, std_br_qei01_dd)

# Plot histogram
axes_i01[1,1].plot(x_br, p_br_qei01_dd, 'k', linewidth=1.5)
axes_i01[1,1].fill_between(x_br, p_br_qei01_dd, color='mediumseagreen')
axes_i01[1,1].axvline(qe_i01_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[1,1].text(qe_i01_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i01[1,1].text(qe_i01_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i01[1,1].set_title("D:Viúvo(a)", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i01_ee = qe_i01.loc[(qe_i01['QE_I01'] == 'E')]

mu_br_qei01_ee, std_br_qei01_ee = norm.fit(qe_i01_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei01_ee = norm.pdf(x_br, mu_br_qei01_ee, std_br_qei01_ee)

# Plot histogram
axes_i01[2,0].plot(x_br, p_br_qei01_ee, 'k', linewidth=1.5)

axes_i01[2,0].fill_between(x_br, p_br_qei01_ee, color='mediumseagreen')
axes_i01[2,0].axvline(qe_i01_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i01[2,0].text(qe_i01_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i01_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i01[2,0].text(qe_i01_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i01_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i01[2,0].set_title("E:Outro", fontsize=size_subtitle, weight='bold')

axes_i01[2,1].axis('off')

axes_i01[0,0].set_ylabel('Distribuição')
axes_i01[1,0].set_ylabel('Distribuição')
axes_i01[2,0].set_ylabel('Distribuição')

for ax in axes_i01.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I01_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()


#%%
############################## QE_I03 ##############################
fig_i03, axes_i03 = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=False, sharey=True, figsize=(15,15))
fig_i03.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Nacionalidade',
                 fontsize=size_title)


####### Alternative A #######
qe_i03_aa = qe_i03.loc[(qe_i03['QE_I03'] == 'A')]

# Média e desvio padrão
mu_br_qei03_aa, std_br_qei03_aa = norm.fit(qe_i03_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei03_aa = norm.pdf(x_br, mu_br_qei03_aa, std_br_qei03_aa)

# Plot histogram
axes_i03[0,0].plot(x_br, p_br_qei03_aa, 'k', linewidth=1.5)

axes_i03[0,0].fill_between(x_br, p_br_qei03_aa, color='mediumseagreen')
axes_i03[0,0].axvline(qe_i03_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[0,0].text(qe_i03_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i03[0,0].text(qe_i03_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i03[0,0].set_title("A:Brasileira", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i03_bb = qe_i03.loc[(qe_i03['QE_I03'] == 'B')]

mu_br_qei03_bb, std_br_qei03_bb = norm.fit(qe_i03_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei03_bb = norm.pdf(x_br, mu_br_qei03_bb, std_br_qei03_bb)

# Plot histogram
axes_i03[0,1].plot(x_br, p_br_qei03_bb, 'k', linewidth=1.5)

axes_i03[0,1].fill_between(x_br, p_br_qei03_bb, color='mediumseagreen')
axes_i03[0,1].axvline(qe_i03_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[0,1].text(qe_i03_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i03[0,1].text(qe_i03_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i03[0,1].set_title("B:Brasileira naturalizada", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i03_cc = qe_i03.loc[(qe_i03['QE_I03'] == 'C')]

mu_br_qei03_cc, std_br_qei03_cc = norm.fit(qe_i03_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei03_cc = norm.pdf(x_br, mu_br_qei03_cc, std_br_qei03_cc)

# Plot histogram
axes_i03[1,0].plot(x_br, p_br_qei03_cc, 'k', linewidth=1.5)

axes_i03[1,0].fill_between(x_br, p_br_qei03_cc, color='mediumseagreen')
axes_i03[1,0].axvline(qe_i03_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i03[1,0].text(qe_i03_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i03_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i03[1,0].text(qe_i03_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i03_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i03[1,0].set_title("C:Estrageira", fontsize=size_subtitle, weight='bold')

axes_i03[1,1].axis('off')

axes_i03[0,0].set_ylabel('Distribuição')
axes_i03[1,0].set_ylabel('Distribuição')
#axes_i03[2,0].set_ylabel('Distribuição')

for ax in axes_i03.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I03_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I06 ##############################
size_title = 18
size_subtitle = 12
fig_i06, axes_i06 = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i06.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                 'Dado socieconômico:Onde e com quem o estudante mora',
                 fontsize=size_title)


# Alternative A
qe_i06_aa = qe_i06.loc[(qe_i06['QE_I06'] == 'A')]

# Média e desvio padrão
mu_br_qei06_aa, std_br_qei06_aa = norm.fit(qe_i06_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_aa = norm.pdf(x_br, mu_br_qei06_aa, std_br_qei06_aa)

# Plot histogram
axes_i06[0,0].plot(x_br, p_br_qei06_aa, 'k', linewidth=1.5)
axes_i06[0,0].set_xlim([0,100])
axes_i06[0,0].fill_between(x_br, p_br_qei06_aa, color='mediumseagreen')
axes_i06[0,0].axvline(qe_i06_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[0,0].text(qe_i06_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_aa['NT_GER'].mean()), fontstyle='italic', weight='bold')
axes_i06[0,0].text(qe_i06_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[0,0].set_title("A:Casa/apartamento,sozinho", fontsize=size_subtitle, weight='bold')

# Plot Gaussiana
qe_i06_bb = qe_i06.loc[(qe_i06['QE_I06'] == 'B')]

mu_br_qei06_bb, std_br_qei06_bb = norm.fit(qe_i06_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_bb = norm.pdf(x_br, mu_br_qei06_bb, std_br_qei06_bb)

# Plot histogram
axes_i06[0,1].plot(x_br, p_br_qei06_bb, 'k', linewidth=1.5)
axes_i06[0,1].set_xlim([0,100])
axes_i06[0,1].fill_between(x_br, p_br_qei06_bb, color='mediumseagreen')
axes_i06[0,1].axvline(qe_i06_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[0,1].text(qe_i06_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i06[0,1].text(qe_i06_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[0,1].set_title("B:Casa/apartamento,pais/parentes", fontsize=size_subtitle, weight='bold')

# Alternative C
qe_i06_cc = qe_i06.loc[(qe_i06['QE_I06'] == 'C')]

mu_br_qei06_cc, std_br_qei06_cc = norm.fit(qe_i06_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_cc = norm.pdf(x_br, mu_br_qei06_cc, std_br_qei06_cc)

# Plot histogram
axes_i06[1,0].plot(x_br, p_br_qei06_cc, 'k', linewidth=1.5)
axes_i06[1,0].set_xlim([0,100])
axes_i06[1,0].fill_between(x_br, p_br_qei06_cc, color='mediumseagreen')
axes_i06[1,0].axvline(qe_i06_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[1,0].text(qe_i06_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i06[1,0].text(qe_i06_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[1,0].set_title("C:Casa/apartamento,cônjugue/filhos", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i06_dd = qe_i06.loc[(qe_i06['QE_I06'] == 'D')]

mu_br_qei06_dd, std_br_qei06_dd = norm.fit(qe_i06_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_dd = norm.pdf(x_br, mu_br_qei06_dd, std_br_qei06_dd)

# Plot histogram
axes_i06[1,1].plot(x_br, p_br_qei06_dd, 'k', linewidth=1.5)
axes_i06[1,1].set_xlim([0,100])
axes_i06[1,1].fill_between(x_br, p_br_qei06_dd, color='mediumseagreen')
axes_i06[1,1].axvline(qe_i06_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[1,1].text(qe_i06_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i06[1,1].text(qe_i06_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[1,1].set_title("D:Casa/apartamento/república,com outras pessoas", fontsize=size_subtitle, weight='bold')

# Alternative E
qe_i06_ee = qe_i06.loc[(qe_i06['QE_I06'] == 'E')]

mu_br_qei06_ee, std_br_qei06_ee = norm.fit(qe_i06_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_ee = norm.pdf(x_br, mu_br_qei06_ee, std_br_qei06_ee)

# Plot histogram
axes_i06[2,0].plot(x_br, p_br_qei06_ee, 'k', linewidth=1.5)
axes_i06[2,0].set_xlim([0,100])
axes_i06[2,0].fill_between(x_br, p_br_qei06_ee, color='mediumseagreen')
axes_i06[2,0].axvline(qe_i06_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[2,0].text(qe_i06_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i06[2,0].text(qe_i06_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[2,0].set_title("E:Alojamento na IES", fontsize=size_subtitle, weight='bold')

# Alternative F
qe_i06_ff = qe_i06.loc[(qe_i06['QE_I06'] == 'F')]

mu_br_qei06_ff, std_br_qei06_ff = norm.fit(qe_i06_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei06_ff = norm.pdf(x_br, mu_br_qei06_ff, std_br_qei06_ff)

# Plot histogram
axes_i06[2,1].plot(x_br, p_br_qei06_ff, 'k', linewidth=1.5)
axes_i06[2,1].set_xlim([0,100])
axes_i06[2,1].fill_between(x_br, p_br_qei06_ff, color='mediumseagreen')
axes_i06[2,1].axvline(qe_i06_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i06[2,1].text(qe_i06_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i06_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i06[2,1].text(qe_i06_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i06_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i06[2,1].set_title("F:Outros (hotel,hospedaria,etc)", fontsize=size_subtitle, weight='bold')

axes_i06[0,0].set_ylabel('Distribuição')
axes_i06[1,0].set_ylabel('Distribuição')
axes_i06[2,0].set_ylabel('Distribuição')

for ax in axes_i06.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I06_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I12 ##############################
fig_i12, axes_i12 = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i12.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Tipo de bolsa de permanência durante da graduação',
                 fontsize=size_title)


####### Alternative A #######
qe_i12_aa = qe_i12.loc[(qe_i12['QE_I12'] == 'A')]

# Média e desvio padrão
mu_br_qei12_aa, std_br_qei12_aa = norm.fit(qe_i12_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_aa = norm.pdf(x_br, mu_br_qei12_aa, std_br_qei12_aa)

# Plot histogram
axes_i12[0,0].plot(x_br, p_br_qei12_aa, 'k', linewidth=1.5)
axes_i12[0,0].set_xlim([0,100])
axes_i12[0,0].fill_between(x_br, p_br_qei12_aa, color='mediumseagreen')
axes_i12[0,0].axvline(qe_i12_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[0,0].text(qe_i12_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[0,0].text(qe_i12_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[0,0].set_title("A:Nenhum", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i12_bb = qe_i12.loc[(qe_i12['QE_I12'] == 'B')]

mu_br_qei12_bb, std_br_qei12_bb = norm.fit(qe_i12_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_bb = norm.pdf(x_br, mu_br_qei12_bb, std_br_qei12_bb)

# Plot histogram
axes_i12[0,1].plot(x_br, p_br_qei12_bb, 'k', linewidth=1.5)
axes_i12[0,1].set_xlim([0,100])
axes_i12[0,1].fill_between(x_br, p_br_qei12_bb, color='mediumseagreen')
axes_i12[0,1].axvline(qe_i12_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[0,1].text(qe_i12_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[0,1].text(qe_i12_bb['NT_GER'].mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[0,1].set_title("B:Aux.moradia", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i12_cc = qe_i12.loc[(qe_i12['QE_I12'] == 'C')]

mu_br_qei12_cc, std_br_qei12_cc = norm.fit(qe_i12_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_cc = norm.pdf(x_br, mu_br_qei12_cc, std_br_qei12_cc)

# Plot histogram
axes_i12[1,0].plot(x_br, p_br_qei12_cc, 'k', linewidth=1.5)
axes_i12[1,0].set_xlim([0,100])
axes_i12[1,0].fill_between(x_br, p_br_qei12_cc, color='mediumseagreen')
axes_i12[1,0].axvline(qe_i12_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[1,0].text(qe_i12_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[1,0].text(qe_i12_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[1,0].set_title("C:Aux.alimentação", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i12_dd = qe_i12.loc[(qe_i12['QE_I12'] == 'D')]

mu_br_qei12_dd, std_br_qei12_dd = norm.fit(qe_i12_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_dd = norm.pdf(x_br, mu_br_qei12_dd, std_br_qei12_dd)

# Plot histogram
axes_i12[1,1].plot(x_br, p_br_qei12_dd, 'k', linewidth=1.5)
axes_i12[1,1].set_xlim([0,100])
axes_i12[1,1].fill_between(x_br, p_br_qei12_dd, color='mediumseagreen')
axes_i12[1,1].axvline(qe_i12_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[1,1].text(qe_i12_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[1,1].text(qe_i12_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[1,1].set_title("D:Aux.moradia e alimentação", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i12_ee = qe_i12.loc[(qe_i12['QE_I12'] == 'E')]

mu_br_qei12_ee, std_br_qei12_ee = norm.fit(qe_i12_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_ee = norm.pdf(x_br, mu_br_qei12_ee, std_br_qei12_ee)

# Plot histogram
axes_i12[2,0].plot(x_br, p_br_qei12_ee, 'k', linewidth=1.5)
axes_i12[2,0].set_xlim([0,100])
axes_i12[2,0].fill_between(x_br, p_br_qei12_ee, color='mediumseagreen')
axes_i12[2,0].axvline(qe_i12_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[2,0].text(qe_i12_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[2,0].text(qe_i12_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[2,0].set_title("E:Aux.permanência", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i12_ff = qe_i12.loc[(qe_i12['QE_I12'] == 'F')]

mu_br_qei12_ff, std_br_qei12_ff = norm.fit(qe_i12_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei12_ff = norm.pdf(x_br, mu_br_qei12_ff, std_br_qei12_ff)

# Plot histogram
axes_i12[2,1].plot(x_br, p_br_qei12_ff, 'k', linewidth=1.5)
axes_i12[2,1].set_xlim([0,100])
axes_i12[2,1].fill_between(x_br, p_br_qei12_ff, color='mediumseagreen')
axes_i12[2,1].axvline(qe_i12_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i12[2,1].text(qe_i12_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i12_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i12[2,1].text(qe_i12_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i12_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i12[2,1].set_title("F:Outro", fontsize=size_subtitle, weight='bold')

axes_i12[0,0].set_ylabel('Distribuição')
axes_i12[1,0].set_ylabel('Distribuição')
axes_i12[2,0].set_ylabel('Distribuição')

for ax in axes_i12.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I12_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I15 ##############################
fig_i15, axes_i15 = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i15.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Ingresso por ação afrimativa e critério',
                 fontsize=size_title)


####### Alternative A #######
qe_i15_aa = qe_i15.loc[(qe_i15['QE_I15'] == 'A')]

# Média e desvio padrão
mu_br_qei15_aa, std_br_qei15_aa = norm.fit(qe_i15_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_aa = norm.pdf(x_br, mu_br_qei15_aa, std_br_qei15_aa)

# Plot histogram
axes_i15[0,0].plot(x_br, p_br_qei15_aa, 'k', linewidth=1.5)

axes_i15[0,0].fill_between(x_br, p_br_qei15_aa, color='mediumseagreen')
axes_i15[0,0].axvline(qe_i15_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[0,0].text(qe_i15_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[0,0].text(qe_i15_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[0,0].set_title("A:Não", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i15_bb = qe_i15.loc[(qe_i15['QE_I15'] == 'B')]

mu_br_qei15_bb, std_br_qei15_bb = norm.fit(qe_i15_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_bb = norm.pdf(x_br, mu_br_qei15_bb, std_br_qei15_bb)

# Plot histogram
axes_i15[0,1].plot(x_br, p_br_qei15_bb, 'k', linewidth=1.5)

axes_i15[0,1].fill_between(x_br, p_br_qei15_bb, color='mediumseagreen')
axes_i15[0,1].axvline(qe_i15_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[0,1].text(qe_i15_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[0,1].text(qe_i15_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[0,1].set_title("B:Sim;étnico-racial", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i15_cc = qe_i15.loc[(qe_i15['QE_I15'] == 'C')]

mu_br_qei15_cc, std_br_qei15_cc = norm.fit(qe_i15_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_cc = norm.pdf(x_br, mu_br_qei15_cc, std_br_qei15_cc)

# Plot histogram
axes_i15[1,0].plot(x_br, p_br_qei15_cc, 'k', linewidth=1.5)

axes_i15[1,0].fill_between(x_br, p_br_qei15_cc, color='mediumseagreen')
axes_i15[1,0].axvline(qe_i15_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[1,0].text(qe_i15_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[1,0].text(qe_i15_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[1,0].set_title("C:Sim;renda", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i15_dd = qe_i15.loc[(qe_i15['QE_I15'] == 'D')]

mu_br_qei15_dd, std_br_qei15_dd = norm.fit(qe_i15_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_dd = norm.pdf(x_br, mu_br_qei15_dd, std_br_qei15_dd)

# Plot histogram
axes_i15[1,1].plot(x_br, p_br_qei15_dd, 'k', linewidth=1.5)
axes_i15[1,1].fill_between(x_br, p_br_qei15_dd, color='mediumseagreen')
axes_i15[1,1].axvline(qe_i15_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[1,1].text(qe_i15_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[1,1].text(qe_i15_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[1,1].set_title("D:Sim;esc.pública/bolsa esc. privada", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i15_ee = qe_i15.loc[(qe_i15['QE_I15'] == 'E')]

mu_br_qei15_ee, std_br_qei15_ee = norm.fit(qe_i15_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_ee = norm.pdf(x_br, mu_br_qei15_ee, std_br_qei15_ee)

# Plot histogram
axes_i15[2,0].plot(x_br, p_br_qei15_ee, 'k', linewidth=1.5)

axes_i15[2,0].fill_between(x_br, p_br_qei15_ee, color='mediumseagreen')
axes_i15[2,0].axvline(qe_i15_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[2,0].text(qe_i15_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[2,0].text(qe_i15_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[2,0].set_title("E:Sim;2 ou mais critérios anteriores", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i15_ff = qe_i15.loc[(qe_i15['QE_I15'] == 'F')]

mu_br_qei15_ff, std_br_qei15_ff = norm.fit(qe_i15_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei15_ff = norm.pdf(x_br, mu_br_qei15_ff, std_br_qei15_ff)

# Plot histogram
axes_i15[2,1].plot(x_br, p_br_qei15_ff, 'k', linewidth=1.5)

axes_i15[2,1].fill_between(x_br, p_br_qei15_ff, color='mediumseagreen')
axes_i15[2,1].axvline(qe_i15_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i15[2,1].text(qe_i15_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i15_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i15[2,1].text(qe_i15_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i15_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i15[2,1].set_title("F:Sim;outro critério", fontsize=size_subtitle, weight='bold')

axes_i15[0,0].set_ylabel('Distribuição')
axes_i15[1,0].set_ylabel('Distribuição')
axes_i15[2,0].set_ylabel('Distribuição')

for ax in axes_i15.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I15_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I16
############################## QE_I16A ##############################
size_title = 18
size_subtitle = 14
fig_i16a, axes_i16a = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                   sharex=False, sharey=True, figsize=(10,10))
fig_i16a.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socieconômico:Unidade da Federação que concluiu o ensino médio',
                 fontsize=size_title)


####### Alternative 11 #######
qe_i16a_aa = qe_i16.loc[(qe_i16['QE_I16'] == 11)]
# Média e desvio padrão
mu_br_qei16a_aa, std_br_qei16a_aa = norm.fit(qe_i16a_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_aa = norm.pdf(x_br, mu_br_qei16a_aa, std_br_qei16a_aa)

# Plot histogram
axes_i16a[0,0].plot(x_br, p_br_qei16a_aa, 'k', linewidth=1.5)
axes_i16a[0,0].set_xlim([0,100])
axes_i16a[0,0].fill_between(x_br, p_br_qei16a_aa, color='mediumseagreen')
axes_i16a[0,0].axvline(qe_i16a_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[0,0].text(qe_i16a_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,0].text(qe_i16a_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,0].set_title("Rondônia", fontsize=size_subtitle, weight='bold')

####### Alternative 12 #######
qe_i16a_bb = qe_i16.loc[(qe_i16['QE_I16'] == 12)]

mu_br_qei16a_bb, std_br_qei16a_bb = norm.fit(qe_i16a_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_bb = norm.pdf(x_br, mu_br_qei16a_bb, std_br_qei16a_bb)

# Plot histogram
axes_i16a[0,1].plot(x_br, p_br_qei16a_bb, 'k', linewidth=1.5)
axes_i16a[0,1].set_xlim([0,100])
axes_i16a[0,1].fill_between(x_br, p_br_qei16a_bb, color='mediumseagreen')
axes_i16a[0,1].axvline(qe_i16a_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[0,1].text(qe_i16a_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,1].text(qe_i16a_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,1].set_title("Acre", fontsize=size_subtitle, weight='bold')

####### Alternative 13 #######
qe_i16a_cc = qe_i16.loc[(qe_i16['QE_I16'] == 13)]

mu_br_qei16a_cc, std_br_qei16a_cc = norm.fit(qe_i16a_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_cc = norm.pdf(x_br, mu_br_qei16a_cc, std_br_qei16a_cc)

# Plot histogram
axes_i16a[0,2].plot(x_br, p_br_qei16a_cc, 'k', linewidth=1.5)
axes_i16a[0,2].set_xlim([0,100])
axes_i16a[0,2].fill_between(x_br, p_br_qei16a_cc, color='mediumseagreen')
axes_i16a[0,2].axvline(qe_i16a_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[0,2].text(qe_i16a_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,2].text(qe_i16a_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[0,2].set_title("Amazonas", fontsize=size_subtitle, weight='bold')

####### Alternative 14 #######
qe_i16a_dd = qe_i16.loc[(qe_i16['QE_I16'] == 14)]

mu_br_qei16a_dd, std_br_qei16a_dd = norm.fit(qe_i16a_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_dd = norm.pdf(x_br, mu_br_qei16a_dd, std_br_qei16a_dd)

# Plot histogram
axes_i16a[1,0].plot(x_br, p_br_qei16a_dd, 'k', linewidth=1.5)
axes_i16a[1,0].set_xlim([0,100])
axes_i16a[1,0].fill_between(x_br, p_br_qei16a_dd, color='mediumseagreen')
axes_i16a[1,0].axvline(qe_i16a_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[1,0].text(qe_i16a_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,0].text(qe_i16a_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,0].set_title("Roraima", fontsize=size_subtitle, weight='bold')

####### Alternative 15 #######
qe_i16a_ee = qe_i16.loc[(qe_i16['QE_I16'] == 15)]

mu_br_qei16a_ee, std_br_qei16a_ee = norm.fit(qe_i16a_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_ee = norm.pdf(x_br, mu_br_qei16a_ee, std_br_qei16a_ee)

# Plot histogram
axes_i16a[1,1].plot(x_br, p_br_qei16a_ee, 'k', linewidth=1.5)
axes_i16a[1,1].set_xlim([0,100])
axes_i16a[1,1].fill_between(x_br, p_br_qei16a_ee, color='mediumseagreen')
axes_i16a[1,1].axvline(qe_i16a_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[1,1].text(qe_i16a_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,1].text(qe_i16a_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,1].set_title("Pará", fontsize=size_subtitle, weight='bold')

####### Alternative 16 #######
qe_i16a_ff = qe_i16.loc[(qe_i16['QE_I16'] == 16)]

mu_br_qei16a_ff, std_br_qei16a_ff = norm.fit(qe_i16a_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_ff = norm.pdf(x_br, mu_br_qei16a_ff, std_br_qei16a_ff)

# Plot histogram
axes_i16a[1,2].plot(x_br, p_br_qei16a_ff, 'k', linewidth=1.5)
axes_i16a[1,2].set_xlim([0,100])
axes_i16a[1,2].fill_between(x_br, p_br_qei16a_ff, color='mediumseagreen')
axes_i16a[1,2].axvline(qe_i16a_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[1,2].text(qe_i16a_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,2].text(qe_i16a_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[1,2].set_title("Amapá", fontsize=size_subtitle, weight='bold')

####### Alternative 17 #######
qe_i16a_gg = qe_i16.loc[(qe_i16['QE_I16'] == 17)]

mu_br_qei16a_gg, std_br_qei16a_gg = norm.fit(qe_i16a_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_gg = norm.pdf(x_br, mu_br_qei16a_gg, std_br_qei16a_gg)

# Plot histogram
axes_i16a[2,0].plot(x_br, p_br_qei16a_gg, 'k', linewidth=1.5)
axes_i16a[2,0].set_xlim([0,100])
axes_i16a[2,0].fill_between(x_br, p_br_qei16a_gg, color='mediumseagreen')
axes_i16a[2,0].axvline(qe_i16a_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[2,0].text(qe_i16a_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,0].text(qe_i16a_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,0].set_title("Tocantins", fontsize=size_subtitle, weight='bold')

####### Alternative 21 #######
qe_i16a_hh = qe_i16.loc[(qe_i16['QE_I16'] == 21)]

mu_br_qei16a_hh, std_br_qei16a_hh = norm.fit(qe_i16a_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_hh = norm.pdf(x_br, mu_br_qei16a_hh, std_br_qei16a_hh)

# Plot histogram
axes_i16a[2,1].plot(x_br, p_br_qei16a_hh, 'k', linewidth=1.5)
axes_i16a[2,1].set_xlim([0,100])
axes_i16a[2,1].fill_between(x_br, p_br_qei16a_hh, color='mediumseagreen')
axes_i16a[2,1].axvline(qe_i16a_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[2,1].text(qe_i16a_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,1].text(qe_i16a_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,1].set_title("Maranhão", fontsize=size_subtitle, weight='bold')

####### Alternative 22 #######
qe_i16a_ii = qe_i16.loc[(qe_i16['QE_I16'] == 22)]

mu_br_qei16a_ii, std_br_qei16a_ii = norm.fit(qe_i16a_ii['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.2)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16a_ii = norm.pdf(x_br, mu_br_qei16a_ii, std_br_qei16a_ii)

# Plot histogram
axes_i16a[2,2].plot(x_br, p_br_qei16a_ii, 'k', linewidth=1.5)
axes_i16a[2,2].set_xlim([0,100])
axes_i16a[2,2].fill_between(x_br, p_br_qei16a_ii, color='mediumseagreen')
axes_i16a[2,2].axvline(qe_i16a_ii['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16a[2,2].text(qe_i16a_ii['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16a_ii['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,2].text(qe_i16a_ii['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16a_ii['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16a[2,2].set_title("Piauí", fontsize=size_subtitle, weight='bold')

axes_i16a[0,0].set_ylabel('Distribuição')
axes_i16a[1,0].set_ylabel('Distribuição')
axes_i16a[2,0].set_ylabel('Distribuição')

for ax in axes_i16a.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I16A_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I16B
############################## QE_I16B ##############################
size_title = 18
size_subtitle = 14
fig_i16b, axes_i16b = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                   sharex=False, sharey=True, figsize=(10,10))
fig_i16b.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socieconômico:Unidade da Federação que concluiu o ensino médio',
                 fontsize=size_title)


####### Alternative 23 #######
qe_i16b_aa = qe_i16.loc[(qe_i16['QE_I16'] == 23)]

# Média e desvio padrão
mu_br_qei16b_aa, std_br_qei16b_aa = norm.fit(qe_i16b_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_aa = norm.pdf(x_br, mu_br_qei16b_aa, std_br_qei16b_aa)

# Plot histogram
axes_i16b[0,0].plot(x_br, p_br_qei16b_aa, 'k', linewidth=1.5)
axes_i16b[0,0].set_xlim([0,100])
axes_i16b[0,0].fill_between(x_br, p_br_qei16b_aa, color='mediumseagreen')
axes_i16b[0,0].axvline(qe_i16b_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[0,0].text(qe_i16b_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,0].text(qe_i16b_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,0].set_title("Ceará", fontsize=size_subtitle, weight='bold')

####### Alternative 24 #######
qe_i16b_bb = qe_i16.loc[(qe_i16['QE_I16'] == 24)]

mu_br_qei16b_bb, std_br_qei16b_bb = norm.fit(qe_i16b_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_bb = norm.pdf(x_br, mu_br_qei16b_bb, std_br_qei16b_bb)

# Plot histogram
axes_i16b[0,1].plot(x_br, p_br_qei16b_bb, 'k', linewidth=1.5)
axes_i16b[0,1].set_xlim([0,100])
axes_i16b[0,1].fill_between(x_br, p_br_qei16b_bb, color='mediumseagreen')
axes_i16b[0,1].axvline(qe_i16b_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[0,1].text(qe_i16b_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,1].text(qe_i16b_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,1].set_title("Rio Grande do Norte", fontsize=size_subtitle, weight='bold')

####### Alternative 25 #######
qe_i16b_cc = qe_i16.loc[(qe_i16['QE_I16'] == 25)]

mu_br_qei16b_cc, std_br_qei16b_cc = norm.fit(qe_i16b_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_cc = norm.pdf(x_br, mu_br_qei16b_cc, std_br_qei16b_cc)

# Plot histogram
axes_i16b[0,2].plot(x_br, p_br_qei16b_cc, 'k', linewidth=1.5)
axes_i16b[0,2].set_xlim([0,100])
axes_i16b[0,2].fill_between(x_br, p_br_qei16b_cc, color='mediumseagreen')
axes_i16b[0,2].axvline(qe_i16b_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[0,2].text(qe_i16b_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,2].text(qe_i16b_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[0,2].set_title("Paraíba", fontsize=size_subtitle, weight='bold')

####### Alternative 26 #######
qe_i16b_dd = qe_i16.loc[(qe_i16['QE_I16'] == 26)]

mu_br_qei16b_dd, std_br_qei16b_dd = norm.fit(qe_i16b_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_dd = norm.pdf(x_br, mu_br_qei16b_dd, std_br_qei16b_dd)

# Plot histogram
axes_i16b[1,0].plot(x_br, p_br_qei16b_dd, 'k', linewidth=1.5)
axes_i16b[1,0].set_xlim([0,100])
axes_i16b[1,0].fill_between(x_br, p_br_qei16b_dd, color='mediumseagreen')
axes_i16b[1,0].axvline(qe_i16b_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[1,0].text(qe_i16b_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,0].text(qe_i16b_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,0].set_title("Pernambuco", fontsize=size_subtitle, weight='bold')

####### Alternative 28 #######
qe_i16b_ee = qe_i16.loc[(qe_i16['QE_I16'] == 28)]

mu_br_qei16b_ee, std_br_qei16b_ee = norm.fit(qe_i16b_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_ee = norm.pdf(x_br, mu_br_qei16b_ee, std_br_qei16b_ee)

# Plot histogram
axes_i16b[1,1].plot(x_br, p_br_qei16b_ee, 'k', linewidth=1.5)
axes_i16b[1,1].set_xlim([0,100])
axes_i16b[1,1].fill_between(x_br, p_br_qei16b_ee, color='mediumseagreen')
axes_i16b[1,1].axvline(qe_i16b_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[1,1].text(qe_i16b_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,1].text(qe_i16b_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,1].set_title("Sergipe", fontsize=size_subtitle, weight='bold')

####### Alternative 29 #######
qe_i16b_ff = qe_i16.loc[(qe_i16['QE_I16'] == 29)]

mu_br_qei16b_ff, std_br_qei16b_ff = norm.fit(qe_i16b_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_ff = norm.pdf(x_br, mu_br_qei16b_ff, std_br_qei16b_ff)

# Plot histogram
axes_i16b[1,2].plot(x_br, p_br_qei16b_ff, 'k', linewidth=1.5)
axes_i16b[1,2].set_xlim([0,100])
axes_i16b[1,2].fill_between(x_br, p_br_qei16b_ff, color='mediumseagreen')
axes_i16b[1,2].axvline(qe_i16b_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[1,2].text(qe_i16b_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,2].text(qe_i16b_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[1,2].set_title("Bahia", fontsize=size_subtitle, weight='bold')

####### Alternative 31 #######
qe_i16b_gg = qe_i16.loc[(qe_i16['QE_I16'] == 31)]

mu_br_qei16b_gg, std_br_qei16b_gg = norm.fit(qe_i16b_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_gg = norm.pdf(x_br, mu_br_qei16b_gg, std_br_qei16b_gg)

# Plot histogram
axes_i16b[2,0].plot(x_br, p_br_qei16b_gg, 'k', linewidth=1.5)
axes_i16b[2,0].set_xlim([0,100])
axes_i16b[2,0].fill_between(x_br, p_br_qei16b_gg, color='mediumseagreen')
axes_i16b[2,0].axvline(qe_i16b_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[2,0].text(qe_i16b_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,0].text(qe_i16b_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,0].set_title("Minas Gerais", fontsize=size_subtitle, weight='bold')

####### Alternative 33 #######
qe_i16b_hh = qe_i16.loc[(qe_i16['QE_I16'] == 33)]

mu_br_qei16b_hh, std_br_qei16b_hh = norm.fit(qe_i16b_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_hh = norm.pdf(x_br, mu_br_qei16b_hh, std_br_qei16b_hh)

# Plot histogram
axes_i16b[2,1].plot(x_br, p_br_qei16b_hh, 'k', linewidth=1.5)
axes_i16b[2,1].set_xlim([0,100])
axes_i16b[2,1].fill_between(x_br, p_br_qei16b_hh, color='mediumseagreen')
axes_i16b[2,1].axvline(qe_i16b_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[2,1].text(qe_i16b_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,1].text(qe_i16b_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,1].set_title("Espírito Santo", fontsize=size_subtitle, weight='bold')

####### Alternative 35 #######
qe_i16b_ii = qe_i16.loc[(qe_i16['QE_I16'] == 35)]

mu_br_qei16b_ii, std_br_qei16b_ii = norm.fit(qe_i16b_ii['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16b_ii = norm.pdf(x_br, mu_br_qei16b_ii, std_br_qei16b_ii)

# Plot histogram
axes_i16b[2,2].plot(x_br, p_br_qei16b_ii, 'k', linewidth=1.5)
axes_i16b[2,2].set_xlim([0,100])
axes_i16b[2,2].fill_between(x_br, p_br_qei16b_ii, color='mediumseagreen')
axes_i16b[2,2].axvline(qe_i16b_ii['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16b[2,2].text(qe_i16b_ii['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16b_ii['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,2].text(qe_i16b_ii['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16b_ii['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16b[2,2].set_title("São Paulo", fontsize=size_subtitle, weight='bold')

axes_i16b[0,0].set_ylabel('Distribuição')
axes_i16b[1,0].set_ylabel('Distribuição')
axes_i16b[2,0].set_ylabel('Distribuição')

for ax in axes_i16b.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I16B_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%% QE_I16C
############################## QE_I16C ##############################
size_title = 18
size_subtitle = 14
fig_i16c, axes_i16c = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                   sharex=False, sharey=True, figsize=(10,10))
fig_i16c.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socieconômico:Unidade da Federação que concluiu o ensino médio',
                 fontsize=size_title)


####### Alternative 41 #######
qe_i16c_aa = qe_i16.loc[(qe_i16['QE_I16'] == 41)]

# Média e desvio padrão
mu_br_qei16c_aa, std_br_qei16c_aa = norm.fit(qe_i16c_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_aa = norm.pdf(x_br, mu_br_qei16c_aa, std_br_qei16c_aa)

# Plot histogram
axes_i16c[0,0].plot(x_br, p_br_qei16c_aa, 'k', linewidth=1.5)
axes_i16c[0,0].set_xlim([0,100])
axes_i16c[0,0].fill_between(x_br, p_br_qei16c_aa, color='mediumseagreen')
axes_i16c[0,0].axvline(qe_i16c_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[0,0].text(qe_i16c_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,0].text(qe_i16c_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,0].set_title("Paraná", fontsize=size_subtitle, weight='bold')

####### Alternative 42 #######
qe_i16c_bb = qe_i16.loc[(qe_i16['QE_I16'] == 42)]

mu_br_qei16c_bb, std_br_qei16c_bb = norm.fit(qe_i16c_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_bb = norm.pdf(x_br, mu_br_qei16c_bb, std_br_qei16c_bb)

# Plot histogram
axes_i16c[0,1].plot(x_br, p_br_qei16c_bb, 'k', linewidth=1.5)
axes_i16c[0,1].set_xlim([0,100])
axes_i16c[0,1].fill_between(x_br, p_br_qei16c_bb, color='mediumseagreen')
axes_i16c[0,1].axvline(qe_i16c_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[0,1].text(qe_i16c_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,1].text(qe_i16c_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,1].set_title("Santa Catarina", fontsize=size_subtitle, weight='bold')

####### Alternative 43 #######
qe_i16c_cc = qe_i16.loc[(qe_i16['QE_I16'] == 43)]

mu_br_qei16c_cc, std_br_qei16c_cc = norm.fit(qe_i16c_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_cc = norm.pdf(x_br, mu_br_qei16c_cc, std_br_qei16c_cc)

# Plot histogram
axes_i16c[0,2].plot(x_br, p_br_qei16c_cc, 'k', linewidth=1.5)
axes_i16c[0,2].set_xlim([0,100])
axes_i16c[0,2].fill_between(x_br, p_br_qei16c_cc, color='mediumseagreen')
axes_i16c[0,2].axvline(qe_i16c_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[0,2].text(qe_i16c_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,2].text(qe_i16c_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[0,2].set_title("Rio Grande do Sul", fontsize=size_subtitle, weight='bold')

####### Alternative 26 #######
qe_i16c_dd = qe_i16.loc[(qe_i16['QE_I16'] == 50)]

mu_br_qei16c_dd, std_br_qei16c_dd = norm.fit(qe_i16c_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_dd = norm.pdf(x_br, mu_br_qei16c_dd, std_br_qei16c_dd)

# Plot histogram
axes_i16c[1,0].plot(x_br, p_br_qei16c_dd, 'k', linewidth=1.5)
axes_i16c[1,0].set_xlim([0,100])
axes_i16c[1,0].fill_between(x_br, p_br_qei16c_dd, color='mediumseagreen')
axes_i16c[1,0].axvline(qe_i16c_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[1,0].text(qe_i16c_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,0].text(qe_i16c_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,0].set_title("Mato Grosso do Sul", fontsize=size_subtitle, weight='bold')

####### Alternative 51 #######
qe_i16c_ee = qe_i16.loc[(qe_i16['QE_I16'] == 51)]

mu_br_qei16c_ee, std_br_qei16c_ee = norm.fit(qe_i16c_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_ee = norm.pdf(x_br, mu_br_qei16c_ee, std_br_qei16c_ee)

# Plot histogram
axes_i16c[1,1].plot(x_br, p_br_qei16c_ee, 'k', linewidth=1.5)
axes_i16c[1,1].set_xlim([0,100])
axes_i16c[1,1].fill_between(x_br, p_br_qei16c_ee, color='mediumseagreen')
axes_i16c[1,1].axvline(qe_i16c_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[1,1].text(qe_i16c_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,1].text(qe_i16c_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,1].set_title("Mato Grosso", fontsize=size_subtitle, weight='bold')

####### Alternative 52#######
qe_i16c_ff = qe_i16.loc[(qe_i16['QE_I16'] == 52)]

mu_br_qei16c_ff, std_br_qei16c_ff = norm.fit(qe_i16c_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_ff = norm.pdf(x_br, mu_br_qei16c_ff, std_br_qei16c_ff)

# Plot histogram
axes_i16c[1,2].plot(x_br, p_br_qei16c_ff, 'k', linewidth=1.5)
axes_i16c[1,2].set_xlim([0,100])
axes_i16c[1,2].fill_between(x_br, p_br_qei16c_ff, color='mediumseagreen')
axes_i16c[1,2].axvline(qe_i16c_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[1,2].text(qe_i16c_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,2].text(qe_i16c_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[1,2].set_title("Goiás", fontsize=size_subtitle, weight='bold')

####### Alternative 53 #######
qe_i16c_gg = qe_i16.loc[(qe_i16['QE_I16'] == 53)]

mu_br_qei16c_gg, std_br_qei16c_gg = norm.fit(qe_i16c_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_gg = norm.pdf(x_br, mu_br_qei16c_gg, std_br_qei16c_gg)

# Plot histogram
axes_i16c[2,0].plot(x_br, p_br_qei16c_gg, 'k', linewidth=1.5)
axes_i16c[2,0].set_xlim([0,100])
axes_i16c[2,0].fill_between(x_br, p_br_qei16c_gg, color='mediumseagreen')
axes_i16c[2,0].axvline(qe_i16c_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[2,0].text(qe_i16c_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[2,0].text(qe_i16c_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[2,0].set_title("Distrito Federal", fontsize=size_subtitle, weight='bold')

####### Alternative 99 #######
qe_i16c_hh = qe_i16.loc[(qe_i16['QE_I16'] == 99)]

mu_br_qei16c_hh, std_br_qei16c_hh = norm.fit(qe_i16c_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.10)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei16c_hh = norm.pdf(x_br, mu_br_qei16c_hh, std_br_qei16c_hh)

# Plot histogram
axes_i16c[2,1].plot(x_br, p_br_qei16c_hh, 'k', linewidth=1.5)
axes_i16c[2,1].set_xlim([0,100])
axes_i16c[2,1].fill_between(x_br, p_br_qei16c_hh, color='mediumseagreen')
axes_i16c[2,1].axvline(qe_i16c_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i16c[2,1].text(qe_i16c_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i16c_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i16c[2,1].text(qe_i16c_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i16c_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i16c[2,1].set_title("Não se aplica", fontsize=size_subtitle, weight='bold')

axes_i16c[2,2].axis('off')

axes_i16c[0,0].set_ylabel('Distribuição')
axes_i16c[1,0].set_ylabel('Distribuição')
axes_i16c[2,0].set_ylabel('Distribuição')

for ax in axes_i16c.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I16C_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I19
############################## QE_I19 ##############################
fig_i19, axes_i19 = plt.subplots(nrows=3, ncols=3, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i19.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Pessoa que mais incentivou a cursar a graduação',
                 fontsize=size_title)


####### Alternative A #######
qe_i19_aa = qe_i19.loc[(qe_i19['QE_I19'] == 'A')]

# Média e desvio padrão
mu_br_qei19_aa, std_br_qei19_aa = norm.fit(qe_i19_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_aa = norm.pdf(x_br, mu_br_qei19_aa, std_br_qei19_aa)

# Plot histogram
axes_i19[0,0].plot(x_br, p_br_qei19_aa, 'k', linewidth=1.5)

axes_i19[0,0].fill_between(x_br, p_br_qei19_aa, color='mediumseagreen')
axes_i19[0,0].axvline(qe_i19_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,0].text(qe_i19_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[0,0].text(qe_i19_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[0,0].set_title("A:Ninguém", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i19_bb = qe_i19.loc[(qe_i19['QE_I19'] == 'B')]

mu_br_qei19_bb, std_br_qei19_bb = norm.fit(qe_i19_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_bb = norm.pdf(x_br, mu_br_qei19_bb, std_br_qei19_bb)

# Plot histogram
axes_i19[0,1].plot(x_br, p_br_qei19_bb, 'k', linewidth=1.5)

axes_i19[0,1].fill_between(x_br, p_br_qei19_bb, color='mediumseagreen')
axes_i19[0,1].axvline(qe_i19_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,1].text(qe_i19_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[0,1].text(qe_i19_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[0,1].set_title("B:Pais", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i19_cc = qe_i19.loc[(qe_i19['QE_I19'] == 'C')]

mu_br_qei19_cc, std_br_qei19_cc = norm.fit(qe_i19_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_cc = norm.pdf(x_br, mu_br_qei19_cc, std_br_qei19_cc)

# Plot histogram
axes_i19[0,2].plot(x_br, p_br_qei19_cc, 'k', linewidth=1.5)

axes_i19[0,2].fill_between(x_br, p_br_qei19_cc, color='mediumseagreen')
axes_i19[0,2].axvline(qe_i19_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[0,2].text(qe_i19_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[0,2].text(qe_i19_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[0,2].set_title("C:Outros membros da família", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i19_dd = qe_i19.loc[(qe_i19['QE_I19'] == 'D')]

mu_br_qei19_dd, std_br_qei19_dd = norm.fit(qe_i19_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_dd = norm.pdf(x_br, mu_br_qei19_dd, std_br_qei19_dd)

# Plot histogram
axes_i19[1,0].plot(x_br, p_br_qei19_dd, 'k', linewidth=1.5)
axes_i19[1,0].fill_between(x_br, p_br_qei19_dd, color='mediumseagreen')
axes_i19[1,0].axvline(qe_i19_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,0].text(qe_i19_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[1,0].text(qe_i19_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[1,0].set_title("D:Professores", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i19_ee = qe_i19.loc[(qe_i19['QE_I19'] == 'E')]

mu_br_qei19_ee, std_br_qei19_ee = norm.fit(qe_i19_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_ee = norm.pdf(x_br, mu_br_qei19_ee, std_br_qei19_ee)

# Plot histogram
axes_i19[1,1].plot(x_br, p_br_qei19_ee, 'k', linewidth=1.5)

axes_i19[1,1].fill_between(x_br, p_br_qei19_ee, color='mediumseagreen')
axes_i19[1,1].axvline(qe_i19_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,1].text(qe_i19_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[1,1].text(qe_i19_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[1,1].set_title("E:Líder religioso", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i19_ff = qe_i19.loc[(qe_i19['QE_I19'] == 'F')]

mu_br_qei19_ff, std_br_qei19_ff = norm.fit(qe_i19_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_ff = norm.pdf(x_br, mu_br_qei19_ff, std_br_qei19_ff)

# Plot histogram
axes_i19[1,2].plot(x_br, p_br_qei19_ff, 'k', linewidth=1.5)

axes_i19[1,2].fill_between(x_br, p_br_qei19_ff, color='mediumseagreen')
axes_i19[1,2].axvline(qe_i19_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[1,2].text(qe_i19_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_ff['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[1,2].text(qe_i19_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[1,2].set_title("F:Colegas/amigos", fontsize=size_subtitle, weight='bold')

####### Alternative G #######
qe_i19_gg = qe_i19.loc[(qe_i19['QE_I19'] == 'G')]

mu_br_qei19_gg, std_br_qei19_gg = norm.fit(qe_i19_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei19_gg = norm.pdf(x_br, mu_br_qei19_gg, std_br_qei19_gg)

# Plot histogram
axes_i19[2,0].plot(x_br, p_br_qei19_gg, 'k', linewidth=1.5)

axes_i19[2,0].fill_between(x_br, p_br_qei19_gg, color='mediumseagreen')
axes_i19[2,0].axvline(qe_i19_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i19[2,0].text(qe_i19_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i19_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i19[2,0].text(qe_i19_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i19_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i19[2,0].set_title("G:Outras pessoas", fontsize=size_subtitle, weight='bold')

axes_i19[2,1].axis('off')
axes_i19[2,2].axis('off')

axes_i19[0,0].set_ylabel('Distribuição')
axes_i19[1,0].set_ylabel('Distribuição')
axes_i19[2,0].set_ylabel('Distribuição')

for ax in axes_i19.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I19_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()

#%%
############################## QE_I20A ##############################
size_title = 18
size_subtitle = 12
fig_i20a, axes_i20a = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i20a.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                 'Dado socieconômico:Grupo(s) que o ajudou a enfrentar dificuldades',
                 fontsize=size_title)


####### Alternative A #######
qe_i20a_aa = qe_i20.loc[(qe_i20['QE_I20'] == 'A')]

# Média e desvio padrão
mu_br_qei20a_aa, std_br_qei20a_aa = norm.fit(qe_i20a_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_aa = norm.pdf(x_br, mu_br_qei20a_aa, std_br_qei20a_aa)

# Plot histogram
axes_i20a[0,0].plot(x_br, p_br_qei20a_aa, 'k', linewidth=1.5)
axes_i20a[0,0].set_xlim([0,100])
axes_i20a[0,0].fill_between(x_br, p_br_qei20a_aa, color='mediumseagreen')
axes_i20a[0,0].axvline(qe_i20a_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[0,0].text(qe_i20a_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_aa['NT_GER'].mean()), fontstyle='italic', weight='bold')
axes_i20a[0,0].text(qe_i20a_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[0,0].set_title("A:Sem dificuldades", fontsize=size_subtitle, weight='bold')


####### Alternative B #######
# Plot Gaussiana
qe_i20a_bb = qe_i20.loc[(qe_i20['QE_I20'] == 'B')]

mu_br_qei20a_bb, std_br_qei20a_bb = norm.fit(qe_i20a_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_bb = norm.pdf(x_br, mu_br_qei20a_bb, std_br_qei20a_bb)

# Plot histogram
axes_i20a[0,1].plot(x_br, p_br_qei20a_bb, 'k', linewidth=1.5)
axes_i20a[0,1].set_xlim([0,100])
axes_i20a[0,1].fill_between(x_br, p_br_qei20a_bb, color='mediumseagreen')
axes_i20a[0,1].axvline(qe_i20a_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[0,1].text(qe_i20a_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20a[0,1].text(qe_i20a_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[0,1].set_title("B:Não recebi apoio", fontsize=size_subtitle, weight='bold')

####### Alternative C #######
qe_i20a_cc = qe_i20.loc[(qe_i20['QE_I20'] == 'C')]

mu_br_qei20a_cc, std_br_qei20a_cc = norm.fit(qe_i20a_cc['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_cc = norm.pdf(x_br, mu_br_qei20a_cc, std_br_qei20a_cc)

# Plot histogram
axes_i20a[1,0].plot(x_br, p_br_qei20a_cc, 'k', linewidth=1.5)
axes_i20a[1,0].set_xlim([0,100])
axes_i20a[1,0].fill_between(x_br, p_br_qei20a_cc, color='mediumseagreen')
axes_i20a[1,0].axvline(qe_i20a_cc['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[1,0].text(qe_i20a_cc['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_cc['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20a[1,0].text(qe_i20a_cc['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_cc['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[1,0].set_title("C:Pais", fontsize=size_subtitle, weight='bold')

####### Alternative D #######
qe_i20a_dd = qe_i20.loc[(qe_i20['QE_I20'] == 'D')]

mu_br_qei20a_dd, std_br_qei20a_dd = norm.fit(qe_i20a_dd['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_dd = norm.pdf(x_br, mu_br_qei20a_dd, std_br_qei20a_dd)

# Plot histogram
axes_i20a[1,1].plot(x_br, p_br_qei20a_dd, 'k', linewidth=1.5)
axes_i20a[1,1].set_xlim([0,100])
axes_i20a[1,1].fill_between(x_br, p_br_qei20a_dd, color='mediumseagreen')
axes_i20a[1,1].axvline(qe_i20a_dd['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[1,1].text(qe_i20a_dd['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_dd['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20a[1,1].text(qe_i20a_dd['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_dd['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[1,1].set_title("D:Avós", fontsize=size_subtitle, weight='bold')

####### Alternative E #######
qe_i20a_ee = qe_i20.loc[(qe_i20['QE_I20'] == 'E')]

mu_br_qei20a_ee, std_br_qei20a_ee = norm.fit(qe_i20a_ee['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_ee = norm.pdf(x_br, mu_br_qei20a_ee, std_br_qei20a_ee)

# Plot histogram
axes_i20a[2,0].plot(x_br, p_br_qei20a_ee, 'k', linewidth=1.5)
axes_i20a[2,0].set_xlim([0,100])
axes_i20a[2,0].fill_between(x_br, p_br_qei20a_ee, color='mediumseagreen')
axes_i20a[2,0].axvline(qe_i20a_ee['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[2,0].text(qe_i20a_ee['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_ee['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20a[2,0].text(qe_i20a_ee['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_ee['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[2,0].set_title("E:Irmãos/primos/tios", fontsize=size_subtitle, weight='bold')

####### Alternative F #######
qe_i20a_ff = qe_i20.loc[(qe_i20['QE_I20'] == 'F')]

# Média e desvio padrão
mu_br_qei20a_ff, std_br_qei20a_ff = norm.fit(qe_i20a_ff['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20a_ff = norm.pdf(x_br, mu_br_qei20a_ff, std_br_qei20a_ff)

# Plot histogram
axes_i20a[2,1].plot(x_br, p_br_qei20a_ff, 'k', linewidth=1.5)
axes_i20a[2,1].set_xlim([0,100])
axes_i20a[2,1].fill_between(x_br, p_br_qei20a_ff, color='mediumseagreen')
axes_i20a[2,1].axvline(qe_i20a_ff['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20a[2,1].text(qe_i20a_ff['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20a_ff['NT_GER'].mean()), fontstyle='italic', weight='bold')
axes_i20a[2,1].text(qe_i20a_ff['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20a_ff['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20a[2,1].set_title("F:Líder religioso", fontsize=size_subtitle, weight='bold')

axes_i20a[0,0].set_ylabel('Distribuição')
axes_i20a[1,0].set_ylabel('Distribuição')
axes_i20a[2,0].set_ylabel('Distribuição')

for ax in axes_i20a.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I20A_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%%
############################## QE_I20B ##############################
size_title = 18
size_subtitle = 12
fig_i20b, axes_i20b = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i20b.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'
                 'Dado socieconômico:Grupo(s) que o ajudou a enfrentar dificuldades',
                 fontsize=size_title)

####### Alternative G #######
# Plot Gaussiana
qe_i20b_gg = qe_i20.loc[(qe_i20['QE_I20'] == 'G')]

mu_br_qei20b_gg, std_br_qei20b_gg = norm.fit(qe_i20b_gg['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20b_gg = norm.pdf(x_br, mu_br_qei20b_gg, std_br_qei20b_gg)

# Plot histogram
axes_i20b[0,0].plot(x_br, p_br_qei20b_gg, 'k', linewidth=1.5)
axes_i20b[0,0].set_xlim([0,100])
axes_i20b[0,0].fill_between(x_br, p_br_qei20b_gg, color='mediumseagreen')
axes_i20b[0,0].axvline(qe_i20b_gg['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20b[0,0].text(qe_i20b_gg['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20b_gg['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20b[0,0].text(qe_i20b_gg['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20b_gg['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20b[0,0].set_title("G:Colegas de curso/amigos", fontsize=size_subtitle, weight='bold')

####### Alternative H #######
qe_i20b_hh = qe_i20.loc[(qe_i20['QE_I20'] == 'H')]

mu_br_qei20b_hh, std_br_qei20b_hh = norm.fit(qe_i20b_hh['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20b_hh = norm.pdf(x_br, mu_br_qei20b_hh, std_br_qei20b_hh)

# Plot histogram
axes_i20b[0,1].plot(x_br, p_br_qei20b_hh, 'k', linewidth=1.5)
axes_i20b[0,1].set_xlim([0,100])
axes_i20b[0,1].fill_between(x_br, p_br_qei20b_hh, color='mediumseagreen')
axes_i20b[0,1].axvline(qe_i20b_hh['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20b[0,1].text(qe_i20b_hh['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20b_hh['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20b[0,1].text(qe_i20b_hh['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20b_hh['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20b[0,1].set_title("I:Professores do curso", fontsize=size_subtitle, weight='bold')

####### Alternative I #######
qe_i20b_ii = qe_i20.loc[(qe_i20['QE_I20'] == 'I')]

mu_br_qei20b_ii, std_br_qei20b_ii = norm.fit(qe_i20b_ii['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20b_ii = norm.pdf(x_br, mu_br_qei20b_ii, std_br_qei20b_ii)

# Plot histogram
axes_i20b[1,0].plot(x_br, p_br_qei20b_ii, 'k', linewidth=1.5)
axes_i20b[1,0].set_xlim([0,100])
axes_i20b[1,0].fill_between(x_br, p_br_qei20b_ii, color='mediumseagreen')
axes_i20b[1,0].axvline(qe_i20b_ii['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20b[1,0].text(qe_i20b_ii['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20b_ii['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20b[1,0].text(qe_i20b_ii['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20b_ii['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20b[1,0].set_title("I:Profissionais de apoio da IES", fontsize=size_subtitle, weight='bold')

####### Alternative J #######
qe_i20b_jj = qe_i20.loc[(qe_i20['QE_I20'] == 'J')]

mu_br_qei20b_jj, std_br_qei20b_jj = norm.fit(qe_i20b_jj['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20b_jj = norm.pdf(x_br, mu_br_qei20b_jj, std_br_qei20b_jj)

# Plot histogram
axes_i20b[1,1].plot(x_br, p_br_qei20b_jj, 'k', linewidth=1.5)
axes_i20b[1,1].set_xlim([0,100])
axes_i20b[1,1].fill_between(x_br, p_br_qei20b_jj, color='mediumseagreen')
axes_i20b[1,1].axvline(qe_i20b_jj['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20b[1,1].text(qe_i20b_jj['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20b_jj['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20b[1,1].text(qe_i20b_jj['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20b_jj['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20b[1,1].set_title("J:Colegas de trabalho", fontsize=size_subtitle, weight='bold')

######## Alternative K #######
qe_i20b_kk = qe_i20.loc[(qe_i20['QE_I20'] == 'K')]

mu_br_qei20b_kk, std_br_qei20b_kk = norm.fit(qe_i20b_kk['NT_GER'])
# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei20b_kk = norm.pdf(x_br, mu_br_qei20b_kk, std_br_qei20b_kk)

# Plot histogram
axes_i20b[2,0].plot(x_br, p_br_qei20b_kk, 'k', linewidth=1.5)
axes_i20b[2,0].set_xlim([0,100])
axes_i20b[2,0].fill_between(x_br, p_br_qei20b_kk, color='mediumseagreen')
axes_i20b[2,0].axvline(qe_i20b_kk['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i20b[2,0].text(qe_i20b_kk['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i20b_kk['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i20b[2,0].text(qe_i20b_kk['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i20b_kk['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i20b[2,0].set_title("K:Outro", fontsize=size_subtitle, weight='bold')

axes_i20b[2,1].axis('off')

axes_i20b[0,0].set_ylabel('Distribuição')
axes_i20b[1,0].set_ylabel('Distribuição')
axes_i20b[2,0].set_ylabel('Distribuição')

for ax in axes_i20b.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I20B_BR_GAUSS.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
plt.show()
#%% QE_I21
############################## QE_I21 ##############################
fig_i21, axes_i21 = plt.subplots(nrows=2, ncols=2, constrained_layout=True, 
                                 sharex=False, sharey=True, figsize=(10,10))
fig_i21.suptitle('Distribuição de notas do Enade no Brasil de 2014 a 2018\n'    
                  'Dado socioeconômico:Alguém da família concluiu um curso superior',
                 fontsize=size_title)


####### Alternative A #######
qe_i21_aa = qe_i21.loc[(qe_i21['QE_I21'] == 'A')]

# Média e desvio padrão
mu_br_qei21_aa, std_br_qei21_aa = norm.fit(qe_i21_aa['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0, 0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei21_aa = norm.pdf(x_br, mu_br_qei21_aa, std_br_qei21_aa)

# Plot histogram
axes_i21[0,0].plot(x_br, p_br_qei21_aa, 'k', linewidth=1.5)
axes_i21[0,0].fill_between(x_br, p_br_qei21_aa, color='mediumseagreen')
axes_i21[0,0].axvline(qe_i21_aa['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i21[0,0].text(qe_i21_aa['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i21_aa['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i21[0,0].text(qe_i21_aa['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i21_aa['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i21[0,0].set_title("A:Sim", fontsize=size_subtitle, weight='bold')

####### Alternative B #######
qe_i21_bb = qe_i21.loc[(qe_i21['QE_I21'] == 'B')]

mu_br_qei21_bb, std_br_qei21_bb = norm.fit(qe_i21_bb['NT_GER'])

# Limites
min_ylim, max_ylim = plt.ylim(0,0.15)
xmin, xmax = plt.xlim(0,100)
x_br = np.linspace(xmin, xmax, 100)

# Normalizando
p_br_qei21_bb = norm.pdf(x_br, mu_br_qei21_bb, std_br_qei21_bb)

# Plot histogram
axes_i21[0,1].plot(x_br, p_br_qei21_bb, 'k', linewidth=1.5)

axes_i21[0,1].fill_between(x_br, p_br_qei21_bb, color='mediumseagreen')
axes_i21[0,1].axvline(qe_i21_bb['NT_GER'].mean(), color='k', linestyle='dashed', linewidth=1.5)
axes_i21[0,1].text(qe_i21_bb['NT_GER'].mean()*1.2, max_ylim*0.9, 'Média: {:.2f}'
         .format(qe_i21_bb['NT_GER'].mean()),fontsize=10, style='italic', weight='bold')
axes_i21[0,1].text(qe_i21_bb['NT_GER'].mean()*1.2, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(qe_i21_bb['NT_GER'].std()),fontsize=10, style='italic', weight='bold')
axes_i21[0,1].set_title("B:Não", fontsize=size_subtitle, weight='bold')

axes_i21[1,0].axis('off')
axes_i21[1,1].axis('off')

axes_i21[0,0].set_ylabel('Distribuição')

for ax in axes_i21.flat:
    plot_axis_name(ax, fontsize=10)

# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/BR/imagens/menor_impacto/QE_I21_BR_GAUSS.png', dpi=150, bbox_inches='tight', pad_inches=0.015);
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
    9) https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html
'''