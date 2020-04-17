# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:56 2020

@author: horacio
"""

import os
import csv
import math
import random
import numpy as np
import pandas as pd

path_al = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/clear_data/AL_data_clean.csv'
path_br = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/clear_data/Brazil_Random_data_clean.csv'

data_al = pd.read_csv(path_al, sep = ',')# Lendo dados de Alagoas
#data_br = pd.read_csv(path_br, sep = ',')# Lendo dados do Brasil

# Extraindo as notas do campo NT_GER
aux_data_al = []
element = ""
for idx in data_al.index:
    element = data_al['NT_GER'][idx]
    if isinstance(element, str) == True:
        aux_data_al.append(element.replace(',','.'))
    elif isinstance(element, str) == False:
        aux_data_al.append(element)

# Transformando string em float
for idx in range(len(aux_data_al)):
    element = aux_data_al[idx]
    if isinstance(aux_data_al[idx], str) == True:
        aux_data_al[idx] = float(element)

# Somando os valores da coluna
s_list = 0.0
cont_nan = 0
for idx in range(len(aux_data_al)):
    element = aux_data_al[idx]
    if math.isnan(float(aux_data_al[idx])) == False:
        s_list += element
    elif math.isnan(float(aux_data_al[idx])) == True:
        cont_nan += 1

print("Soma da lista: ", s_list)
print("Nan number: ", cont_nan)
        