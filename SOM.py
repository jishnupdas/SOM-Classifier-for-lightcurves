#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:00:36 2019

@author: jishnu
"""

#%%
import glob
import pickle
import numpy as np
import seaborn as sns
from minisom import MiniSom 
import matplotlib.pyplot as plt


#%%
class SOM:
    '''
    Class to easily train, use, save and load the SOM clustering/classifier
    
    packages used;
        Pickle     #to save and load files
        Minisom    #minimal implementation of Self Organizing maps
        
    '''
    
    def __init__(self,files=None,data=None,som=None,network_h=None,
                 network_w=None,coords=None,x=None,y=None,fnames=None):
        '''initializing all the necessary values
        
        Parameters:
        -----------
        
        files      : list of files, which contain the 1D phase binned data
        data       : An array of arrays,each element is a 1D phase binned LC
        som        : self organising map NN with hxw neurons
        network_h  : height of the network
        network_w  : width of the network
        coords     : An array which contains som.winner for all LCs in data
        x,y        : x,y coords from coords
        
        '''
        
        self.files     = files
        self.fnames    = fnames
        self.data      = data
        self.som       = som
        self.network_h = 50
        self.network_w = 50
        self.coords    = coords
        self.x         = x
        self.y         = y
        
        
        
    def set_files(self,path):
        '''
        takes path to the files as arg; returns list of files in the path
        '''
        self.files  = glob.glob(path+'*')
        
        
        
    def get_arr(self,file):
        '''
        Get data from a file as an np array:
        reject files which has nan values in them
        nan can break the SOM classifier
        '''
        data  = np.loadtxt(file)
        if np.isnan(data).any() == True:
            return np.nan
        else:
            return data



    def set_data(self):
        '''
        opens each file in the folder and reads the data
        into an array, and appends that to the data array
        if it doesnt contain any nan values
        '''
        
        self.fnames,self.data,err_f = [],[],[]

        for f in self.files:
            arr = self.get_arr(f)
            if arr is not np.nan:
                self.fnames.append(f)
                self.data.append(arr)
            else:
                err_f.append(f)
                
        
    def set_som(self,sigma,learning_rate):
        '''
        initializes the network:
        by default 50x50 with 0.1 sigma and 1.5 lr is initialized
        '''
        
        self.som = MiniSom(x = self.network_h,y = self.network_w,
                           input_len = 32, sigma = sigma,
                           learning_rate = learning_rate)
        
        self.som.random_weights_init(self.data)
        #initialize random weights to the network
        
        
    
    def train_som(self,number):
        '''
        tains the network with 'number' iterations by randomly taking
        'number' of elements from the data array
        '''
        self.som.train_random(self.data, number)
        
        
        
    def save_model(self,outfile):
        '''
        Save the trained model
        '''
        with open(outfile+'.p', 'wb') as outfile:
                pickle.dump(self.som, outfile)
    
    
    def load_model(self,som_file):
        '''
        Load the saved model
        '''
        with open(som_file, 'rb') as infile:
            self.som = pickle.load(infile)
            
    
    
    def get_coords(self):
        '''
        Runs each of the elements of the dataset through the SOM
        and gets the winner and appends it to the coords array
        '''
        
        self.coords = []
        err         = []
        
        for d in self.data:
            try:
                self.coords.append(np.array(self.som.winner(d)))
            except:
                #print("err with ",str(d))
                err.append(d)
        
        #getting x,y points
        self.x   = [i[0] for i in self.coords]
        self.y   = [i[1] for i in self.coords]
        
        return self.x,self.y
        
        
    
    def plot_winners(self):
        
        x,y = self.x,self.y
        
        plt.style.use('seaborn')
        plt.figure(figsize=(9,9))
        plt.plot(x,y,'.',alpha=0.15)
        sns.kdeplot(x,y,cmap='Blues',shade=True,bw=1.5,shade_lowest=False,                    alpha=0.8)
        plt.show()
        plt.close()

