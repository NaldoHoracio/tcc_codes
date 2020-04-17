# -*- coding: utf-8 -*-
"""
Created on Thu 23 15:37:56 2020

@author: edvonaldo
"""
import os
import csv
import math
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

regr = RandomForestRegressor(max_depth=2, random_state=0)

regr.fit(X, y)

predit = regr.predict(X)