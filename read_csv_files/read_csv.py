import os
import csv
import random
import numpy as np
import pandas as pd


#path_file_al = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/tcc_data/UFAL.csv'
#path_file_brazil = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/tcc_data/Random.csv'
path = '/home/horacio/Documentos/UFAL/ENADE/microdados_Enade_2017_portal_2018.10.09/3.DADOS/MICRODADOS_ENADE_2017.txt'

#data_brazil = pd.read_csv(path_file_brazil, sep = ',')
#data_al = pd.read_csv(path_file_al, sep = ',')
data = pd.read_csv(path, sep = ';')# Lendo

data_al = data[data['QE_I16'] == 27]# Get apenas em Alagoas
data_al_save = data_al.to_csv('AL_data.csv')# Salvando como .csv

data_br = data[data['QE_I16'] != 27]
data_br_br = data_br.sample(7048)# Escolhendo aleatoriamente no DataFrame
data_br_save = data_br_br.to_csv('Brazil_Random_data.csv')

    
#def main():
#    read_file()#main()

    
