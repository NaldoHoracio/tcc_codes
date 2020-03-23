# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:56 2020

@author: horacio
"""

import os
import csv
import random
import numpy as np
import pandas as pd


path_al = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/clear_data/AL_data_clean.csv'
path_br = '/home/horacio/Documentos/UFAL/TCC/tcc_codes/clear_data/Brazil_Random_data_clean.csv'

data_al = pd.read_csv(path_al, sep = ',')# Lendo dados de Alagoas
data_br = pd.read_csv(path_br, sep = ',')# Lendo dados do Brasil

sum_data_al = 0.0

list_al_nt_ger = data_al['NT_GER']# Get apenas em Alagoas
aux = list_al_nt_ger


    