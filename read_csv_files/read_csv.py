import os
import csv
import random
import numpy as np
import pandas as pd
import time

def seconds_transform(seconds_time):
  hours = int(seconds_time/3600)
  rest_1 = seconds_time%3600
  minutes = int(rest_1/60)
  seconds = rest_1 - 60*minutes
  #print(seconds)
  print("Time: ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")

start_time = time.time() # Time start

#path = 'G:/Meu Drive/UFAL/TCC/DADOS/microdados_Enade_2017_portal_2018.10.09/3.DADOS/MICRODADOS_ENADE_2017.txt'
path_2014 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/microdados_enade_2014/3.DADOS/MICRODADOS_ENADE_2014.txt'
path_2015 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/microdados_enade_2015/3.DADOS/MICRODADOS_ENADE_2015.txt'
path_2016 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/microdados_enade_2016/3.DADOS/MICRODADOS_ENADE_2016.txt'
path_2018 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/microdados_enade_2018/3.DADOS/MICRODADOS_ENADE_2018.txt'

#data = pd.read_csv(path, sep = ';')# Lendo 2017
data_2014 = pd.read_csv(path_2014, sep=';')
data_2015 = pd.read_csv(path_2015, sep=';')
data_2016 = pd.read_csv(path_2016, sep=';')
data_2018 = pd.read_csv(path_2018, sep=';')

#data_al = data[data['QE_I16'] == 27]# Get apenas em Alagoas 2017
data_al_2014 = data_2014[data_2014['QE_I16'] == 27]
data_al_2015 = data_2015[data_2015['QE_I16'] == 27]
data_al_2016 = data_2016[data_2016['QE_I16'] == 27]
data_al_2018 = data_2018[data_2018['QE_I16'] == 27]

data_al_save_2014 = data_al_2014.to_csv('AL_2014.csv')
data_al_save_2015 = data_al_2015.to_csv('AL_2015.csv')
data_al_save_2016 = data_al_2016.to_csv('AL_2016.csv')
data_al_save_2018 = data_al_2018.to_csv('AL_2018.csv')
#data_al_save = data_al.to_csv('AL_data.csv')# Salvando como .csv 2017
seconds_al = (time.time() - start_time) # Time end
seconds_transform(seconds_al)

#% Excluindo Alagoas
#data_br = data[data['QE_I16'] != 27]

#% Removendo linhas com Nan
#data_br_no_nan = data_br.dropna(subset = ["QE_I01"], inplace = True)

#%
#data_br_br = data_br.sample(7084)# Escolhendo aleatoriamente no DataFrame
#data_br_save = data_br_br.to_csv('BR_data.csv')

    
#def main():
#    read_file()#main()

    
