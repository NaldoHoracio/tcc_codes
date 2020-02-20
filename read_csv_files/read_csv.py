import os
import csv
import numpy as np
import pandas as pd


path_file = '/home/horacio/Documentos/UFAL/TCC/tcc_data/microdados_censo_superior_2016/DADOS/DM_CURSO.csv'
column1 = ['NO_MUNICIPIO_CURSO']

data = pd.read_csv(path_file, sep = '|')
uf_al = data[data['SGL_UF_CURSO'] == 'AL']
#data = pd.read_csv(path_file, sep = '|', skipinitialspace=True, usecols=column1)



    
