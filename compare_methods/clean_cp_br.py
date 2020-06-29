# -*- coding: utf-8 -*-
"""
Título: Módulo de preparação dos dados

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

def clean_data_br(path_br):
    features_br = pd.read_csv(path_br)
    del features_br['Unnamed: 0']
    # Escolhendo apenas as colunas de interesse
    features_br = features_br.loc[:,'NT_GER':'QE_I26']
    features_br = features_br.drop(features_br.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

    #Observando os dados
    #print('O formato dos dados é: ', features_br.shape)

    describe_br = features_br.describe()

    #print('Descrição para as colunas: ', describe_br)
    #print(describe_br.columns)

    # Números que são strings para float
    # Colunas NT_GER a NT_DIS_FG ^ NT_CE a NT_DIS_CE
    features_br['NT_GER'] = features_br['NT_GER'].str.replace(',','.')
    features_br['NT_GER'] = features_br['NT_GER'].astype(float)

    features_br['NT_FG'] = features_br['NT_FG'].str.replace(',','.')
    features_br['NT_FG'] = features_br['NT_FG'].astype(float)

    features_br['NT_OBJ_FG'] = features_br['NT_OBJ_FG'].str.replace(',','.')
    features_br['NT_OBJ_FG'] = features_br['NT_OBJ_FG'].astype(float)

    features_br['NT_DIS_FG'] = features_br['NT_DIS_FG'].str.replace(',','.')
    features_br['NT_DIS_FG'] = features_br['NT_DIS_FG'].astype(float)

    features_br['NT_CE'] = features_br['NT_CE'].str.replace(',','.')
    features_br['NT_CE'] = features_br['NT_CE'].astype(float)

    features_br['NT_OBJ_CE'] = features_br['NT_OBJ_CE'].str.replace(',','.')
    features_br['NT_OBJ_CE'] = features_br['NT_OBJ_CE'].astype(float)

    features_br['NT_DIS_CE'] = features_br['NT_DIS_CE'].str.replace(',','.')
    features_br['NT_DIS_CE'] = features_br['NT_DIS_CE'].astype(float)
    # Substituindo valores nan pela mediana (medida resistente)
    features_br_median = features_br.iloc[:,0:16].median()

    features_br.iloc[:,0:16] = features_br.iloc[:,0:16].fillna(features_br.iloc[:,0:16].median())
    # Observando os dados
    #print('O formato dos dados é: ', features_br.shape)

    describe_br = features_br.describe()

    #print('Descrição para as colunas: ', describe_br)
    #print(describe_br.columns)

    # Convertendo os labels de predição para arrays numpy
    labels_br = np.array(features_br['NT_GER'])
    print('Media das labels: %.2f' %(labels_br.mean()) )
    # Removendo as features de notas
    features_br = features_br.drop(['NT_GER','NT_FG','NT_OBJ_FG','NT_DIS_FG',
                               'NT_FG_D1','NT_FG_D1_PT','NT_FG_D1_CT',
                               'NT_FG_D2','NT_FG_D2_PT','NT_FG_D2_CT',
                               'NT_CE','NT_OBJ_CE','NT_DIS_CE',
                               'NT_CE_D1','NT_CE_D2','NT_CE_D3'], axis = 1)
    # Salvando e convertendo
    # Salvando os nomes das colunas (features) com os dados para uso posterior
    # antes de codificar
    features_br_list = list(features_br.columns)


    # One hot encoding - QE_I01 a QE_I26
    features_br = pd.get_dummies(data=features_br, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
    # Salvando os nomes das colunas (features) com os dados para uso posterior
    # depois de codificar
    features_br_list_oh = list(features_br.columns)
    #
    # Convertendo para numpy
    features_br = np.array(features_br)
    
    return features_br