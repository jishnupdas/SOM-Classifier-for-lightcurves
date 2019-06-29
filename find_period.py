#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:38:29 2019

@author: jishnu
"""

#%%
import Lomb_scargle
import pandas as pd

#%%
fpath = '/media/astronomy/External HDD/TESSDATA/lc_sector1/clean/'
file_list = '/media/astronomy/External HDD/TESSDATA/lc_sector1/clean/list_sec1'

with open(file_list,'r') as fl_list:
    f_list = fl_list.readlines()[50000:100000]

F = [i.strip('\n') for i in f_list]

files = [fpath+f for f in F]
per = []

#%%
with open('log','a+') as rec:
    for file,fname in zip(files,F):
        t,y,e = Lomb_scargle.get_data(file)
        Period = Lomb_scargle.lomb_scargle(t,y)
        per.append(Period)
        rec.write(fname+','+str(Period)+'\n')
        print('period:',Period)

#%%
df = pd.DataFrame()
df = pd.read_csv('log',sep=',')
df.columns = ['files','period']
lp = df[(df.period <= 1.5)]

#%%
files,period = list(lp.files),list(lp.period)

with open('filtered_files','a+') as flog:
    for f,p in zip(files,period):
        flog.write(f+','+str(p)+'\n')
