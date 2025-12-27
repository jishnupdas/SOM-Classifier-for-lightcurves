#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:00:36 2019

@author: jishnu
"""

#%%
import glob
import pickle
import logging
from typing import Optional, List, Tuple
import numpy as np
import seaborn as sns
from minisom import MiniSom 
import matplotlib.pyplot as plt

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
DEFAULT_NETWORK_SIZE = 50
DEFAULT_INPUT_LENGTH = 32

#%%
class SOM:
    '''
    Class to easily train, use, save and load the SOM clustering/classifier
    
    packages used;
        Pickle     #to save and load files
        Minisom    #minimal implementation of Self Organizing maps
        
    '''
    
    def __init__(self, files: Optional[List[str]] = None, data: Optional[List[np.ndarray]] = None,
                 som: Optional[MiniSom] = None, network_h: Optional[int] = None,
                 network_w: Optional[int] = None, coords: Optional[List[np.ndarray]] = None,
                 x: Optional[List[int]] = None, y: Optional[List[int]] = None,
                 fnames: Optional[List[str]] = None) -> None:
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
        self.network_h = network_h if network_h is not None else DEFAULT_NETWORK_SIZE
        self.network_w = network_w if network_w is not None else DEFAULT_NETWORK_SIZE
        self.coords    = coords
        self.x         = x
        self.y         = y
        
        
        
    def set_files(self, path: str) -> None:
        '''
        takes path to the files as arg; returns list of files in the path
        '''
        self.files = glob.glob(path + '*')
        logger.info(f"Found {len(self.files)} files in {path}")
        
        
        
    def get_arr(self, file: str) -> Optional[np.ndarray]:
        '''
        Get data from a file as an np array:
        reject files which has nan values in them
        nan can break the SOM classifier
        '''
        try:
            data = np.loadtxt(file)
            if np.isnan(data).any():
                logger.warning(f"File contains NaN values: {file}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
            return None



    def set_data(self) -> None:
        '''
        opens each file in the folder and reads the data
        into an array, and appends that to the data array
        if it doesnt contain any nan values
        '''
        
        self.fnames, self.data, err_f = [], [], []

        for f in self.files:
            arr = self.get_arr(f)
            if arr is not None:
                self.fnames.append(f)
                self.data.append(arr)
            else:
                err_f.append(f)
        
        logger.info(f"Successfully loaded {len(self.data)} valid files")
        if err_f:
            logger.warning(f"Rejected {len(err_f)} files with NaN values")
                
        
    
    def set_som(self, sigma: float, learning_rate: float) -> None:
        '''
        initializes the network:
        by default 50x50 with 0.1 sigma and 1.5 lr is initialized
        '''
        # Use DEFAULT_INPUT_LENGTH for input_len (maintains backward compatibility)
        self.som = MiniSom(x=self.network_h, y=self.network_w,
                           input_len=DEFAULT_INPUT_LENGTH, sigma=sigma,
                           learning_rate=learning_rate)
        
        self.som.random_weights_init(self.data)
        logger.info(f"Initialized SOM network: {self.network_h}x{self.network_w}, "
                   f"sigma={sigma}, lr={learning_rate}")
        
        
    
    def train_som(self, number: int) -> None:
        '''
        trains the network with 'number' iterations by randomly taking
        'number' of elements from the data array
        '''
        logger.info(f"Starting SOM training for {number} iterations")
        self.som.train_random(self.data, number)
        logger.info("SOM training completed")
        
        
        
    def save_model(self, outfile: str) -> None:
        '''
        Save the trained model
        '''
        try:
            with open(outfile + '.p', 'wb') as f:
                pickle.dump(self.som, f)
            logger.info(f"Model saved to {outfile}.p")
        except Exception as e:
            logger.error(f"Failed to save model to {outfile}.p: {e}")
            raise
    
    
    
    def load_model(self, som_file: str) -> None:
        '''
        Load the saved model
        '''
        try:
            with open(som_file, 'rb') as infile:
                self.som = pickle.load(infile)
            logger.info(f"Model loaded from {som_file}")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {som_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model from {som_file}: {e}")
            raise
            
    
    
    def get_coords(self) -> Tuple[List[int], List[int]]:
        '''
        Runs each of the elements of the dataset through the SOM
        and gets the winner and appends it to the coords array
        '''
        
        self.coords = []
        err = []
        self.x = []
        self.y = []
        
        
        for d in self.data:
            try:
                coord = np.array(self.som.winner(d))
                self.coords.append(coord)
                self.x.append(coord[0])
                self.y.append(coord[1])
            except Exception as e:
                logger.error(f"Error getting coordinates for data point: {e}")
                err.append(d)
        
        logger.info(f"Retrieved coordinates for {len(self.coords)} data points")
        if err:
            logger.warning(f"Failed to get coordinates for {len(err)} data points")
        
        return self.x, self.y
        
        
    
    def plot_winners(self) -> None:
        
        x, y = self.x, self.y
        
        plt.style.use('seaborn')
        plt.figure(figsize=(9, 9))
        plt.plot(x, y, '.', alpha=0.15)
        sns.kdeplot(x, y, cmap='Blues', shade=True, bw=1.5, shade_lowest=False, alpha=0.8)
        plt.show()
        plt.close()
        logger.debug("Generated SOM winners plot")

