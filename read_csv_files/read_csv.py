import os
import csv
import numpy as np
import pandas as pd


path_file_ufal = '/home/horacio/Documentos/UFAL/TCC/tcc_data/UFAL.csv'
path_file_brazil = '/home/horacio/Documentos/UFAL/TCC/tcc_data/Random.csv'

data_ufal = pd.read_csv(path_file_ufal, sep = ',')
data_brazil = pd.read_csv(path_file_brazil, sep = ',')

#data[data['SGL_UF_CURSO'] == 'AL']
#without_al = data[data['SGL_UF_CURSO'] != 'AL']
#data_random = without_al.sample(10000)



    
