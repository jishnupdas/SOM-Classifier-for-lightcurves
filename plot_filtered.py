#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:23:54 2018

@author: jishnu
"""
#%%
import Lomb_scargle
import os

#%%
fpath     = '/media/astronomy/External HDD/TESSDATA/lc_sector1/clean/'
file_list = 'filtered_files'

with open(file_list,'r') as f_list:
    '''read the file containing filtered filenames'''
    f_list    = f_list.readlines()
    filenames = [i.split(',')[0] for i in f_list] #get only the filenames
    files     = [fpath+f for f in filenames] #make a list with fullpath to file
    impath    = '/home/astronomy/Documents/tess_sec1_lc/lc_image/' #path to write output images
    fpath     = '/home/astronomy/Documents/tess_sec1_lc/lc_data/'
    
    for file,fname in zip(files,filenames):
        if os.path.isfile(impath+fname):
            print('file %s already exists!'%fname)
        else:
            t,y,e  = Lomb_scargle.get_data(file)
            Period = Lomb_scargle.lomb_scargle(t,y)
            Lomb_scargle.phase_fold(t,y,Period,impath+fname)
            print('period:',Period)

