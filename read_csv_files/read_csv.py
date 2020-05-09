import os
import csv
import random
import numpy as np
import pandas as pd


path = 'G:/Meu Drive/UFAL/TCC/DADOS/microdados_Enade_2017_portal_2018.10.09/3.DADOS/MICRODADOS_ENADE_2017.txt'

data = pd.read_csv(path, sep = ';')# Lendo

#data_al = data[data['QE_I16'] == 27]# Get apenas em Alagoas
#data_al_save = data_al.to_csv('AL_data.csv')# Salvando como .csv

#%% Excluindo Alagoas
data_br = data[data['QE_I16'] != 27]

#%% Removendo linhas com Nan
data_br_no_nan = data_br.dropna(subset = ["QE_I01"], inplace = True)

#%%
data_br_br = data_br.sample(7084)# Escolhendo aleatoriamente no DataFrame
data_br_save = data_br_br.to_csv('BR_data.csv')

    
#def main():
#    read_file()#main()

    
