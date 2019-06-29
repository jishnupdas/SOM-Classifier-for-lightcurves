#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:10:13 2019

@author: jishnu
"""
#%%

import SOM
import glob
import numpy as np
import seaborn as sns
from minisom import MiniSom 
import matplotlib.pyplot as plt

#%%
dpath = '/home/jishnu/Documents/TESS/tess_data/1D_lc/'

#%%

som   = SOM.SOM()

som.set_files(dpath)
som.set_data()
som.set_som(sigma=0.1,learning_rate=1.5)
som.train_som(number=10000)
som.save_model('som2_model')
som.load_model('som2_model.p')

x,y   = som.get_coords()

som.plot_winners()
