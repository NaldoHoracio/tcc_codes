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
import matplotlib.pyplot as plt

path = 'G:/Meu Drive/UFAL/2020.1/Aprendizado de Máquina/Databases/temps.csv'

features = pd.read_csv(path)

anomalies = features.describe()

# Janeiro
jan = features[features['month'] == 1]
jan_average = jan['average']
print('Average Jan %.2f' %(jan_average.sum()/jan_average.size))

# Fevereiro
feb = features[features['month'] == 2]
feb_average = feb['average']
print('Average Feb %.2f' %(feb_average.sum()/feb_average.size))

# Março
mar = features[features['month'] == 3]
mar_average = mar['average']
print('Average Mar %.2f' %(mar_average.sum()/mar_average.size))

# Abril
abr = features[features['month'] == 4]
abr_average = abr['average']
print('Average Abr %.2f' %(abr_average.sum()/abr_average.size))

# Maio
may = features[features['month'] == 5]
may_average = may['average']
print('Average May %.2f' %(may_average.sum()/may_average.size))

# Junho
jun = features[features['month'] == 6]
jun_average = jun['average']
print('Average Jun %.2f' %(jun_average.sum()/jun_average.size))

# Julho
jul = features[features['month'] == 7]
jul_average = jul['average']
print('Average Jul %.2f' %(jul_average.sum()/jul_average.size))

# Agosto
aug = features[features['month'] == 8]
aug_average = aug['average']
print('Average Aug %.2f' %(aug_average.sum()/aug_average.size))

# Setembro
sept = features[features['month'] == 9]
sept_average = sept['average']
print('Average Sept %.2f' %(sept_average.sum()/sept_average.size))

# Outubro
octb = features[features['month'] == 10]
octb_average = octb['average']
print('Average Octb %.2f' %(octb_average.sum()/octb_average.size))

# Novembro
nov = features[features['month'] == 11]
nov_average = nov['average']
print('Average Nov %.2f' %(nov_average.sum()/nov_average.size))

# Dezembro
dec = features[features['month'] == 12]
dec_average = dec['average']
print('Average Dec %.2f' %(dec_average.sum()/dec_average.size))

        

