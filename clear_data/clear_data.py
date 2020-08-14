# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:59:35 2020

@author: horacio
"""

import os
import csv
import random
import numpy as np
import pandas as pd

'''
path_al = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/tcc_data/AL_data.csv'
path_br = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/tcc_data/Brazil_Random_data.csv'

data_al = pd.read_csv(path_al, sep = ',')
data_br = pd.read_csv(path_br, sep = ',')

# Selecionando apenas as colunas de interesse 26-60; 70-150
# Removendo as colunas de 0 a 25
data_al_clean = data_al.drop(data_al.iloc[:, 0:26], inplace = True, axis = 1)
data_br_clean = data_br.drop(data_br.iloc[:, 0:26], inplace = True, axis = 1)

# Selecionando apenas as colunas de interesse 26-60; 70-150
# Removendo as colunas de 61 a 95 do arquivo original
data_al_clean = data_al.drop(data_al.iloc[:, 35:44], inplace = True, axis = 1)
data_br_clean = data_br.drop(data_br.iloc[:, 35:44], inplace = True, axis = 1)

data_al_clean = data_al.to_csv('AL_data_clean.csv')
data_br_clean = data_al.to_csv('Brazil_Random_data_clean.csv')

'''

path_2014 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2014.csv'

data_2014 = pd.read_csv(path_2014, sep = ',')

#%%

# Selecionando apenas as colunas de interesse 26-60; 70-150
# Removendo as colunas de 0 a 25
data_al_2014 = data_2014.drop(data_2014.ix[:,], axis = 1)

# Selecionando apenas as colunas de interesse 26-60; 70-150
# Removendo as colunas de 61 a 95 do arquivo original
data_al_2014 = data_2014.drop(data_2014.iloc[:, 35:44], inplace = True, axis = 1)

data_al_2014 = data_2014.to_csv('AL_2014r.csv')