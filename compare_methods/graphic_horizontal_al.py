# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:08:03 2021

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


importance_fields_al_dt_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_DT_AL.csv')
importance_fields_al_rf_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_RF_AL.csv')
importance_fields_al_ls_t = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_LS_AL.csv')

#<<<<<<< Updated upstream
# Importância de variáveis
#=======

#%% Importância de variáveis
#>>>>>>> Stashed changes

vimp_al_rf = importance_fields_al_rf_t.iloc[5]

vimp_al_dt = importance_fields_al_dt_t.iloc[5]

vimp_al_ls = importance_fields_al_ls_t.iloc[5]

#

I01_AL_RF = vimp_al_rf['I01_AL']; I02_AL_RF = vimp_al_rf['I02_AL'];

I03_AL_RF = vimp_al_rf['I03_AL']; I04_AL_RF = vimp_al_rf['I04_AL']; 

I05_AL_RF = vimp_al_rf['I05_AL']; I06_AL_RF = vimp_al_rf['I06_AL'];

I07_AL_RF = vimp_al_rf['I07_AL']; I08_AL_RF = vimp_al_rf['I08_AL'];

I09_AL_RF = vimp_al_rf['I09_AL']; I10_AL_RF = vimp_al_rf['I10_AL']; 

I11_AL_RF = vimp_al_rf['I11_AL']; I12_AL_RF = vimp_al_rf['I12_AL'];

I13_AL_RF = vimp_al_rf['I13_AL']; I14_AL_RF = vimp_al_rf['I14_AL']; 

I15_AL_RF = vimp_al_rf['I15_AL']; I16_AL_RF = vimp_al_rf['I16_AL']; 

I17_AL_RF = vimp_al_rf['I17_AL']; I18_AL_RF = vimp_al_rf['I18_AL'];

I19_AL_RF = vimp_al_rf['I19_AL']; I20_AL_RF = vimp_al_rf['I20_AL']; 

I21_AL_RF = vimp_al_rf['I21_AL']; I22_AL_RF = vimp_al_rf['I22_AL']; 

I23_AL_RF = vimp_al_rf['I23_AL']; I24_AL_RF = vimp_al_rf['I24_AL'];

I25_AL_RF = vimp_al_rf['I25_AL']; I26_AL_RF = vimp_al_rf['I26_AL'];

# Decision Tree
I01_AL_DT = vimp_al_dt['I01_AL']; I02_AL_DT = vimp_al_dt['I02_AL'];

I03_AL_DT = vimp_al_dt['I03_AL']; I04_AL_DT = vimp_al_dt['I04_AL']; 

I05_AL_DT = vimp_al_dt['I05_AL']; I06_AL_DT = vimp_al_dt['I06_AL'];

I07_AL_DT = vimp_al_dt['I07_AL']; I08_AL_DT = vimp_al_dt['I08_AL'];

I09_AL_DT = vimp_al_dt['I09_AL']; I10_AL_DT = vimp_al_dt['I10_AL']; 

I11_AL_DT = vimp_al_dt['I11_AL']; I12_AL_DT = vimp_al_dt['I12_AL'];

I13_AL_DT = vimp_al_dt['I13_AL']; I14_AL_DT = vimp_al_dt['I14_AL']; 

I15_AL_DT = vimp_al_dt['I15_AL']; I16_AL_DT = vimp_al_dt['I16_AL']; 

I17_AL_DT = vimp_al_dt['I17_AL']; I18_AL_DT = vimp_al_dt['I18_AL'];

I19_AL_DT = vimp_al_dt['I19_AL']; I20_AL_DT = vimp_al_dt['I20_AL']; 

I21_AL_DT = vimp_al_dt['I21_AL']; I22_AL_DT = vimp_al_dt['I22_AL']; 

I23_AL_DT = vimp_al_dt['I23_AL']; I24_AL_DT = vimp_al_dt['I24_AL'];

I25_AL_DT = vimp_al_dt['I25_AL']; I26_AL_DT = vimp_al_dt['I26_AL'];

# Lasso
I01_AL_LS = vimp_al_ls['I01_AL']; I02_AL_LS = vimp_al_ls['I02_AL'];

I03_AL_LS = vimp_al_ls['I03_AL']; I04_AL_LS = vimp_al_ls['I04_AL']; 

I05_AL_LS = vimp_al_ls['I05_AL']; I06_AL_LS = vimp_al_ls['I06_AL'];

I07_AL_LS = vimp_al_ls['I07_AL']; I08_AL_LS = vimp_al_ls['I08_AL'];

I09_AL_LS = vimp_al_ls['I09_AL']; I10_AL_LS = vimp_al_ls['I10_AL']; 

I11_AL_LS = vimp_al_ls['I11_AL']; I12_AL_LS = vimp_al_ls['I12_AL'];

I13_AL_LS = vimp_al_ls['I13_AL']; I14_AL_LS = vimp_al_ls['I14_AL']; 

I15_AL_LS = vimp_al_ls['I15_AL']; I16_AL_LS = vimp_al_ls['I16_AL']; 

I17_AL_LS = vimp_al_ls['I17_AL']; I18_AL_LS = vimp_al_ls['I18_AL'];

I19_AL_LS = vimp_al_ls['I19_AL']; I20_AL_LS = vimp_al_ls['I20_AL']; 

I21_AL_LS = vimp_al_ls['I21_AL']; I22_AL_LS = vimp_al_ls['I22_AL']; 

I23_AL_LS = vimp_al_ls['I23_AL']; I24_AL_LS = vimp_al_ls['I24_AL'];

I25_AL_LS = vimp_al_ls['I25_AL']; I26_AL_LS = vimp_al_ls['I26_AL'];

#%% AVERAGE LIST
def mean_list(lst):
    average_lst = sum(lst) / len(lst) 
    return round(average_lst, 4)
#%% TABELA 12 - MAIS IMPORTANTES - ALAGOAS
# Ref vertical barplot: https://python-graph-gallery.com/11-grouped-barplot/
# Ref horizontal barplot: https://stackoverflow.com/questions/15201386/how-to-plot-multiple-horizontal-bars-in-one-chart-with-matplotlib
# Ref put numer: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

I02_AL_HIGH = [I02_AL_RF,I02_AL_DT,I02_AL_LS]
I08_AL_HIGH = [I08_AL_RF,I08_AL_DT,I08_AL_LS]
I11_AL_HIGH = [I11_AL_RF,I11_AL_DT,I11_AL_LS]
I13_AL_HIGH = [I13_AL_RF,I13_AL_DT,I13_AL_LS]
I17_AL_HIGH = [I17_AL_RF,I17_AL_DT,I17_AL_LS]
I18_AL_HIGH = [I18_AL_RF,I18_AL_DT,I18_AL_LS]
I23_AL_HIGH = [I23_AL_RF,I23_AL_DT,I23_AL_LS]

df_al = pd.DataFrame(dict(graph=['Cor;raça', 'Renda;total;família', 
                 'Fonte;bolsa;mensalidade', 'Bolsa;acadêmica;graduação',
                 'Tipo;escola;medio', 'Modalidade;medio', 
                 'Horas;estudo;semana'],
                         vimp_al_high=[100*mean_list(I02_AL_HIGH), 100*mean_list(I08_AL_HIGH), 
                                       100*mean_list(I11_AL_HIGH), 100*mean_list(I13_AL_HIGH), 
                                       100*mean_list(I17_AL_HIGH), 100*mean_list(I18_AL_HIGH), 
                                       100*mean_list(I23_AL_HIGH)]))
vi_al_high=(10.74, 7.55, 9.77, 15.53,
           8.58, 3.57, 6.25)

ind = np.arange(len(df_al))
width = 0.3
r1 = ind
#r2 = ind + width
#r3 = r2 + width
fig, ax = plt.subplots()
ax.barh(r1, df_al.vimp_al_high, width, color='green')
#ax.barh(r2, df_al.vimp_dt_al, width, color='green', label='Árvore de Decisão')
#ax.barh(r3, df_al.vimp_ls_al, width, color='blue', label='Lasso')

ax.set(yticks=ind, yticklabels=df_al.graph, xlim=(0,20),ylim=[2*width - 1, len(df_al)])

for i, v in enumerate(vi_al_high):
    ax.text(v + 0.1, i - 0.1, str(v), color='black', fontweight='bold')
 
# Create legend & Show graphic
plt.title('Media aritmética das categorias com maior impacto na nota do Enade: dados de Alagoas',
          fontsize=8);
plt.xlabel('Importância (%)');
#plt.ylabel('Variável');
plt.legend();
plt.savefig('compare_methods/AL/AL_CP_H_T_HIGH.png', dpi=450, bbox_inches='tight');

#%% #%% #%% TABELA 13 - MENOS IMPORTANTES - ALAGOAS
# Ref vertical barplot: https://python-graph-gallery.com/11-grouped-barplot/
# Ref horizontal barplot: https://stackoverflow.com/questions/15201386/how-to-plot-multiple-horizontal-bars-in-one-chart-with-matplotlib
# Ref put numer: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

I01_AL_LOW = [I01_AL_RF,I01_AL_DT,I01_AL_LS]
I03_AL_LOW = [I03_AL_RF,I03_AL_DT,I03_AL_LS]
I12_AL_LOW = [I12_AL_RF,I12_AL_DT,I12_AL_LS]
I15_AL_LOW = [I15_AL_RF,I15_AL_DT,I15_AL_LS]
I16_AL_LOW = [I16_AL_RF+I16_AL_DT,+I16_AL_LS]
I19_AL_LOW = [I19_AL_RF,I19_AL_DT,I19_AL_LS]
I21_AL_LOW = [I21_AL_RF,I21_AL_DT,I21_AL_LS]

vi_al_low=(0.95, 0.22, 1.9, 1.38,
           0.0, 1.07, 0.5)

df_al = pd.DataFrame(dict(graph=['Estado civil', 'Nacionalidade', 'Aux;permanência', 
                 'Ingresso;cota', 'UF;medio', 'Quem;incentivo;curso','Quem;família;superior'],
                         vimp_al_low=[100*mean_list(I01_AL_LOW), 100*mean_list(I03_AL_LOW), 
                                      100*mean_list(I12_AL_LOW), 100*mean_list(I15_AL_LOW), 
                                      100*mean_list(I16_AL_LOW), 100*mean_list(I19_AL_LOW), 
                                      100*mean_list(I21_AL_LOW)]))
ind = np.arange(len(df_al))
width = 0.3
#r1 = ind
#r2 = ind + width
#r3 = r2 + width
fig, ax = plt.subplots()
ax.barh(r1, df_al.vimp_al_low, width, color='green')
#ax.barh(r2, df_al.vimp_dt_al, width, color='green', label='Árvore de Decisão')
#ax.barh(r3, df_al.vimp_ls_al, width, color='blue', label='Lasso')

ax.set(yticks=ind, yticklabels=df_al.graph, xlim=(0,5),ylim=[2*width - 1, len(df_al)])

for i, v in enumerate(vi_al_low):
    ax.text(v + 0.1, i - 0.1, str(v), color='black', fontweight='bold')
 
# Create legend & Show graphic
plt.title('Media aritmética das categorias com menor impacto na nota do Enade: dados de Alagoas',
          fontsize=6);
plt.xlabel('Importância (%)');
#plt.ylabel('Variável');
plt.legend();
plt.savefig('compare_methods/AL/AL_CP_H_T_LOW.png', dpi=450, bbox_inches='tight');

'''
#%%

categories_al = ['Estado civil', 'Cor;raça', 'Nacionalidade', 'Escolarização;pai', 'Escolarização;mãe', 
                 'Onde;com quem;moro', 'Qtde;moram;comigo', 'Renda;total;família', 
                 'Situação;financeira;atual', 'Situação;atual;trabalho', 
                 'Fonte;bolsa;mensalidade', 'Aux;permanência', 
                 'Bolsa;acadêmica;graduação',
                 'Atividade;exterior', 'Ingresso;cota', 'UF;medio', 'Tipo;escola;medio', 'Modalidade;medio', 
                 'Quem;incentivo;curso','Grupo;força;curso', 'Quem;família;superior', 
                 'Quantos;livros;ano', 'Horas;estudo;semana', 'Oportunidade;idioma;estrang',
                 'Por que;curso', 'Por que;IES']
vimp_al_rf_vals = [I01_AL_RF, I02_AL_RF, I03_AL_RF, I04_AL_RF, I05_AL_RF, I06_AL_RF, I07_AL_RF,
                  I08_AL_RF, I09_AL_RF, I10_AL_RF, I11_AL_RF, I12_AL_RF, I13_AL_RF, I14_AL_RF,
                  I15_AL_RF, I16_AL_RF, I17_AL_RF, I18_AL_RF, I19_AL_RF, I20_AL_RF, I21_AL_RF,
                  I22_AL_RF, I23_AL_RF, I24_AL_RF, I25_AL_RF, I26_AL_RF]
vimp_al_dt_vals = [I01_AL_DT, I02_AL_DT, I03_AL_DT, I04_AL_DT, I05_AL_DT, I06_AL_DT, I07_AL_DT,
                  I08_AL_DT, I09_AL_DT, I10_AL_DT, I11_AL_DT, I12_AL_DT, I13_AL_DT, I14_AL_DT,
                  I15_AL_DT, I16_AL_DT, I17_AL_DT, I18_AL_DT, I19_AL_DT, I20_AL_DT, I21_AL_DT,
                  I22_AL_DT, I23_AL_DT, I24_AL_DT, I25_AL_DT, I26_AL_DT]
vimp_al_ls_vals = [I01_AL_LS, I02_AL_LS, I03_AL_LS, I04_AL_LS, I05_AL_LS, I06_AL_LS, I07_AL_LS,
                  I08_AL_LS, I09_AL_LS, I10_AL_LS, I11_AL_LS, I12_AL_LS, I13_AL_LS, I14_AL_LS,
                  I15_AL_LS, I16_AL_LS, I17_AL_LS, I18_AL_LS, I19_AL_LS, I20_AL_LS, I21_AL_LS,
                  I22_AL_LS, I23_AL_LS, I24_AL_LS, I25_AL_LS, I26_AL_LS]
df_vimp_al = pd.DataFrame({'Floresta Aleatória': vimp_al_rf_vals,
                           'Árvore de Decisão': vimp_al_dt_vals,
                           'Lasso': vimp_al_ls_vals}, index=categories_al)

figure = df_vimp_al.plot.barh()

 Visualization of Variable Importances
# QE_I27a
bar_width = 0.2;
#fig27a = plt.figure();
#ax27aa = fig27a.add_axes([0,0,1,1]);
#bar_width = 0.1;

ax27a = ['Cor;raça', 'Nacionalidade', 'Escolarização;pai', 'Escolarização;mãe', 
         'Onde;com quem;moro', 'Qtde;moram;comigo', 'Renda;total;família', 'Situação;financeira;atual', 
         'Situação;atual;trabalho', 'Fonte;bolsa;mensalidade', 'Aux;permanência', 
         'Bolsa;acadêmica;graduação'];
y27a_rf = [np.sum(I01_AL_RF),np.sum(I02_AL_RF),np.sum(I03_AL_RF),np.sum(I04_AL_RF),
          np.sum(I05_AL_RF),np.sum(I06_AL_RF),np.sum(I07_AL_RF),np.sum(I08_AL_RF),
          np.sum(I09_AL_RF),np.sum(I10_AL_RF),np.sum(I11_AL_RF),np.sum(I12_AL_RF),
          np.sum(I13_AL_RF)];
y27a_rf = list(map(lambda t:t*100, y27a_rf));
y27a_dt = [np.sum(I01_AL_DT),np.sum(I02_AL_DT),np.sum(I03_AL_DT),np.sum(I04_AL_DT),
          np.sum(I05_AL_DT),np.sum(I06_AL_DT),np.sum(I07_AL_DT),np.sum(I08_AL_DT),
          np.sum(I09_AL_DT),np.sum(I10_AL_DT),np.sum(I11_AL_DT),np.sum(I12_AL_DT),
          np.sum(I13_AL_DT)];
y27a_dt = list(map(lambda t:t*100, y27a_dt));
y27a_ls = [np.sum(I01_AL_LS),np.sum(I02_AL_LS),np.sum(I03_AL_LS),np.sum(I04_AL_LS),
          np.sum(I05_AL_LS),np.sum(I06_AL_LS),np.sum(I07_AL_LS),np.sum(I08_AL_LS),
          np.sum(I09_AL_LS),np.sum(I10_AL_LS),np.sum(I11_AL_LS),np.sum(I12_AL_LS),
          np.sum(I13_AL_LS)];

y27a_ls = list(map(lambda t:t*100, y27a_ls));


# Configurando a posição no eixo x
axis27a = np.arange(len(y27a_rf))
y27a1 = [x + bar_width for x in axis27a]
y27a2 = [x + bar_width for x in y27a1]
y27a3 = [x + bar_width for x in y27a2]

# Fazendo o plot
plt.barh(y27a1, color='red', width=bar_width, edgecolor='white', label='Floresta Aleatória')
plt.barh(y27a2, color='green', width=bar_width, edgecolor='white', label='Árvore de Decisão')
plt.barh(y27a3, color='blue', width=bar_width, edgecolor='white', label='Lasso')
    
# Nomeando o eixo x
plt.xlabel('group', fontweight='bold')
plt.xticks([k + bar_width for k in range(len(y27a_rf))], \
           ax27a,\
           rotation=90, fontsize=9)
plt.ylabel('Importância (%)');
plt.xlabel('Variável (AL)');
plt.title('Categorias QE_I01 a QE_I13');
plt.legend();
plt.savefig('AL/AL_CP_H_T12.png', dpi=450, bbox_inches='tight');

'''

