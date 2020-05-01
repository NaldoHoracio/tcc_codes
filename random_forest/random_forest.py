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

# IDENTIFY ANOMALIES / MISSING DATA

path_al = 'G:/Meu Drive/UFAL/TCC/CODES/tcc_codes/tcc_data/AL_data.csv'

features_al = pd.read_csv(path_al)

del features_al['Unnamed: 0']


# Choosing only the columns of interest
features_al = features_al.loc[:,'NT_GER':'QE_I26']
features_al = features_al.drop(features_al.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

#%% Observando os dados
print('O formato dos dados é: ', features_al.shape)

describe_al = features_al.describe()

print('Descrição para as colunas: ', describe_al)
print(describe_al.columns)

#%% Números que são strings por float
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

#%% Trocando nan pela mediana

for idx in range()

#%%
features_al['NT_FG'] = features_al['NT_FG'].fillna(features_al['NT_FG'].median())

features_al['NT_OBJ_FG'] = features_al['NT_OBJ_FG'].fillna(features_al['NT_OBJ_FG'].median())

features_al['NT_DIS_FG'] = features_al['NT_DIS_FG'].fillna(features_al['NT_DIS_FG'].median())

features_al['NT_CE'] = features_al['NT_CE'].fillna(features_al['NT_CE'].median())

features_al['NT_OBJ_CE'] = features_al['NT_OBJ_CE'].fillna(features_al['NT_OBJ_CE'].median())

features_al['NT_DIS_CE'] = features_al['NT_DIS_CE'].fillna(features_al['NT_DIS_CE'].median())