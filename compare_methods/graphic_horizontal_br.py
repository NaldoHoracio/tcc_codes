# -*- coding: utf-8 -*-
"""
Gráfico de barras horizontal

DADOS DO BRASIL (EXCLUINDO ALAGOAS)

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

# Lendo os arquivos

importance_fields_br_dt_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_DT_BR.csv')
importance_fields_br_rf_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_RF_BR.csv')
importance_fields_br_ls_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_LS_BR.csv')

# Importância de variáveis

vimp_br_rf = importance_fields_br_rf_t.iloc[4]

vimp_br_dt = importance_fields_br_dt_t.iloc[5]

vimp_br_ls = importance_fields_br_ls_t.iloc[4]

#%%

I01_BR_RF = vimp_br_rf['I01_BR']; I02_BR_RF = vimp_br_rf['I02_BR'];

I03_BR_RF = vimp_br_rf['I03_BR']; I04_BR_RF = vimp_br_rf['I04_BR']; 

I05_BR_RF = vimp_br_rf['I05_BR']; I06_BR_RF = vimp_br_rf['I06_BR'];

I07_BR_RF = vimp_br_rf['I07_BR']; I08_BR_RF = vimp_br_rf['I08_BR'];

I09_BR_RF = vimp_br_rf['I09_BR']; I10_BR_RF = vimp_br_rf['I10_BR']; 

I11_BR_RF = vimp_br_rf['I11_BR']; I12_BR_RF = vimp_br_rf['I12_BR'];

I13_BR_RF = vimp_br_rf['I13_BR']; I14_BR_RF = vimp_br_rf['I14_BR']; 

I15_BR_RF = vimp_br_rf['I15_BR']; I16_BR_RF = vimp_br_rf['I16_BR']; 

I17_BR_RF = vimp_br_rf['I17_BR']; I18_BR_RF = vimp_br_rf['I18_BR'];

I19_BR_RF = vimp_br_rf['I19_BR']; I20_BR_RF = vimp_br_rf['I20_BR']; 

I21_BR_RF = vimp_br_rf['I21_BR']; I22_BR_RF = vimp_br_rf['I22_BR']; 

I23_BR_RF = vimp_br_rf['I23_BR']; I24_BR_RF = vimp_br_rf['I24_BR'];

I25_BR_RF = vimp_br_rf['I25_BR']; I26_BR_RF = vimp_br_rf['I26_BR'];

# Decision Tree
I01_BR_DT = vimp_br_dt['I01_BR']; I02_BR_DT = vimp_br_dt['I02_BR'];

I03_BR_DT = vimp_br_dt['I03_BR']; I04_BR_DT = vimp_br_dt['I04_BR']; 

I05_BR_DT = vimp_br_dt['I05_BR']; I06_BR_DT = vimp_br_dt['I06_BR'];

I07_BR_DT = vimp_br_dt['I07_BR']; I08_BR_DT = vimp_br_dt['I08_BR'];

I09_BR_DT = vimp_br_dt['I09_BR']; I10_BR_DT = vimp_br_dt['I10_BR']; 

I11_BR_DT = vimp_br_dt['I11_BR']; I12_BR_DT = vimp_br_dt['I12_BR'];

I13_BR_DT = vimp_br_dt['I13_BR']; I14_BR_DT = vimp_br_dt['I14_BR']; 

I15_BR_DT = vimp_br_dt['I15_BR']; I16_BR_DT = vimp_br_dt['I16_BR']; 

I17_BR_DT = vimp_br_dt['I17_BR']; I18_BR_DT = vimp_br_dt['I18_BR'];

I19_BR_DT = vimp_br_dt['I19_BR']; I20_BR_DT = vimp_br_dt['I20_BR']; 

I21_BR_DT = vimp_br_dt['I21_BR']; I22_BR_DT = vimp_br_dt['I22_BR']; 

I23_BR_DT = vimp_br_dt['I23_BR']; I24_BR_DT = vimp_br_dt['I24_BR'];

I25_BR_DT = vimp_br_dt['I25_BR']; I26_BR_DT = vimp_br_dt['I26_BR'];

# Lasso
I01_BR_LS = vimp_br_ls['I01_BR']; I02_BR_LS = vimp_br_ls['I02_BR'];

I03_BR_LS = vimp_br_ls['I03_BR']; I04_BR_LS = vimp_br_ls['I04_BR']; 

I05_BR_LS = vimp_br_ls['I05_BR']; I06_BR_LS = vimp_br_ls['I06_BR'];

I07_BR_LS = vimp_br_ls['I07_BR']; I08_BR_LS = vimp_br_ls['I08_BR'];

I09_BR_LS = vimp_br_ls['I09_BR']; I10_BR_LS = vimp_br_ls['I10_BR']; 

I11_BR_LS = vimp_br_ls['I11_BR']; I12_BR_LS = vimp_br_ls['I12_BR'];

I13_BR_LS = vimp_br_ls['I13_BR']; I14_BR_LS = vimp_br_ls['I14_BR']; 

I15_BR_LS = vimp_br_ls['I15_BR']; I16_BR_LS = vimp_br_ls['I16_BR']; 

I17_BR_LS = vimp_br_ls['I17_BR']; I18_BR_LS = vimp_br_ls['I18_BR'];

I19_BR_LS = vimp_br_ls['I19_BR']; I20_BR_LS = vimp_br_ls['I20_BR']; 

I21_BR_LS = vimp_br_ls['I21_BR']; I22_BR_LS = vimp_br_ls['I22_BR']; 

I23_BR_LS = vimp_br_ls['I23_BR']; I24_BR_LS = vimp_br_ls['I24_BR'];

I25_BR_LS = vimp_br_ls['I25_BR']; I26_BR_LS = vimp_br_ls['I26_BR'];

#%% AVERAGE LIST
def mean_list(lst):
    average_lst = sum(lst) / len(lst) 
    return round(average_lst, 4)
#%% #%% TABELA 14 - MAIS IMPORTANTES - BRASIL
# Ref vertical barplot: https://python-graph-gallery.com/11-grouped-barplot/
# Ref horizontal barplot: https://stackoverflow.com/questions/15201386/how-to-plot-multiple-horizontal-bars-in-one-chart-with-matplotlib
# Ref put numer: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

I02_BR_HIGH = [I02_BR_RF,I02_BR_DT,I02_BR_LS]
I08_BR_HIGH = [I08_BR_RF,I08_BR_DT,I08_BR_LS]
I11_BR_HIGH = [I11_BR_RF,I11_BR_DT,I11_BR_LS]
I13_BR_HIGH = [I13_BR_RF,I13_BR_DT,I13_BR_LS]
I17_BR_HIGH = [I17_BR_RF,I17_BR_DT,I17_BR_LS]
I23_BR_HIGH = [I23_BR_RF,I23_BR_DT,I23_BR_LS]
I25_BR_HIGH = [I25_BR_RF,I25_BR_DT,I25_BR_LS]

vi_br_high=(6.32, 8.7, 14.85, 11.17,
            10.23,5.18, 5.2)

df_br = pd.DataFrame(dict(graph=['Cor;raça', 'Renda;total;família', 
                 'Fonte;bolsa;mensalidade', 'Bolsa;acadêmica;graduação',
                 'Tipo;escola;medio', 'Horas; estudo; semana', 
                 'Porque; curso'],
                          vimp_br_high=[100*mean_list(I02_BR_HIGH), 100*mean_list(I08_BR_HIGH), 
                                        100*mean_list(I11_BR_HIGH), 100*mean_list(I13_BR_HIGH), 
                                        100*mean_list(I17_BR_HIGH), 100*mean_list(I23_BR_HIGH), 
                                        100*mean_list(I25_BR_HIGH)]))
ind = np.arange(len(df_br))
width = 0.3
r1 = ind
#r2 = ind + width
#r3 = r2 + width
fig, ax = plt.subplots()
ax.barh(r1, df_br.vimp_br_high, width, color='green')
#ax.barh(r2, df_br.vimp_dt_br, width, color='green', label='Árvore de Decisão')
#ax.barh(r3, df_br.vimp_ls_br, width, color='blue', label='Lasso')

ax.set(yticks=ind, yticklabels=df_br.graph, xlim=(0,25),ylim=[2*width - 1, len(df_br)])

for i, v in enumerate(vi_br_high):
    ax.text(v + 0.1, i - 0.1, str(v), color='black', fontweight='bold')
 
# Create legend & Show graphic
plt.title('Media aritmética das categorias com maior impacto na nota do Enade: dados no Brasil (excluindo Alagoas)',
          fontsize=6);
plt.xlabel('Importância (%)');
#plt.ylabel('Variável');
plt.legend();
plt.savefig('compare_methods/BR/BR_CP_H_T_HIGH.png', dpi=450, bbox_inches='tight');

#%% #%% #%% TABELA 15 - MENOS IMPORTANTE - BRASIL
# Ref vertical barplot: https://python-graph-gallery.com/11-grouped-barplot/
# Ref horizontal barplot: https://stackoverflow.com/questions/15201386/how-to-plot-multiple-horizontal-bars-in-one-chart-with-matplotlib
# Ref put numer: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

I01_BR_LOW = [I01_BR_RF,I01_BR_DT,I01_BR_LS]
I03_BR_LOW = [I03_BR_RF,I03_BR_DT,I03_BR_LS]
I06_BR_LOW = [I06_BR_RF,I06_BR_DT,I06_BR_LS]
I12_BR_LOW = [I12_BR_RF,I12_BR_DT,I12_BR_LS]
I15_BR_LOW = [I15_BR_RF,I15_BR_DT,I15_BR_LS]
I16_BR_LOW = [I16_BR_RF,I16_BR_DT,I16_BR_LS]
I19_BR_LOW = [I19_BR_RF,I19_BR_DT,I19_BR_LS]
I20_BR_LOW = [I20_BR_RF,I20_BR_DT,I20_BR_LS]
I21_BR_LOW = [I21_BR_RF,I21_BR_DT,I21_BR_LS]

vi_br_low=(0.64, 0.01, 1.09, 0.3,
           1.64, 2.05, 1.46, 2.11, 0.82)

df_br = pd.DataFrame(dict(graph=['Estado civil', 'Nacionalidade', 'Onde;com quem;moro', 
                                 'Aux;permanência', 
                                 'Ingresso;cota', 'UF;medio', 'Quem;incentivo;curso',
                                 'Grupo;força;curso', 'Quem;família;superior'],
                         vimp_br_low=[100*mean_list(I01_BR_LOW), 100*mean_list(I03_BR_LOW), 
                                      100*mean_list(I06_BR_LOW), 100*mean_list(I12_BR_LOW), 
                                      100*mean_list(I15_BR_LOW), 100*mean_list(I16_BR_LOW), 
                                      100*mean_list(I19_BR_LOW), 100*mean_list(I20_BR_LOW),
                                      100*mean_list(I21_BR_LOW)]))
ind = np.arange(len(df_br))
width = 0.3
r1 = ind
#r2 = ind + width
#r3 = r2 + width
fig, ax = plt.subplots()
ax.barh(r1, df_br.vimp_br_low, width, color='green')
#ax.barh(r2, df_al.vimp_dt_al, width, color='green', label='Árvore de Decisão')
#x.barh(r3, df_al.vimp_ls_al, width, color='blue', label='Lasso')

ax.set(yticks=ind, yticklabels=df_br.graph, xlim=(0,5), ylim=[2*width - 1, len(df_br)])

for i, v in enumerate(vi_br_low):
    ax.text(v + 0.1, i - 0.1, str(v), color='black', fontweight='bold')
 
# Create legend & Show graphic
plt.title('Media aritmética das categorias com menor impacto na nota do Enade: dados no Brasil (excluindo Alagoas)',
          fontsize=6);
plt.xlabel('Importância (%)');
#plt.ylabel('Variável');
plt.legend();
plt.savefig('compare_methods/BR/BR_CP_H_T_LOW.png', dpi=450, bbox_inches='tight');