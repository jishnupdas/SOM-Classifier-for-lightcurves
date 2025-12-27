#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:23:05 2019

@author: jishnu
"""
#%%
import logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
DEFAULT_BIN_LENGTH = 64
DEFAULT_POLYFIT_DEGREE = 30
MAX_FREQUENCY_DEFAULT = 8.0
VARIANCE_THRESHOLD = 0.0001

#%%
class Lctools:
    '''
    A set of tools to analyse timeseries data(lightcurves)

    '''
    def __init__(self, file: Optional[str] = None, jd: Optional[np.ndarray] = None,
                 mag: Optional[np.ndarray] = None, err: Optional[np.ndarray] = None,
                 period: Optional[float] = None, phase: Optional[np.ndarray] = None,
                 df: Optional[pd.DataFrame] = None, synth: Optional[np.ndarray] = None,
                 magx2: Optional[np.ndarray] = None, noise: Optional[np.ndarray] = None,
                 phase2: Optional[np.ndarray] = None, pshift: Optional[float] = None,
                 bin1D: Optional[np.ndarray] = None, binlen: Optional[int] = None,
                 binarr: Optional[np.ndarray] = None) -> None:
        '''
        Parameters:
        -----------
        file       : the file which contains the timeseries data
        jd,mag,err : HJD,Magnitude,error from the file as arrays
        noise      : set of frequencies to be removed from periodogram
                     which are contributed by external factors
        period     : period of the data computed using LS
        synth      : synthetic lc, computed using a polynomial fit
        phases     : array which holds all phases
        
        df         : dataframe which contains hjd,phase,mag,err
        '''
        self.file   = file
        self.jd     = jd
        self.mag    = mag
        self.err    = err
        self.period = period
        self.phase  = phase
        self.df     = df
        self.synth  = synth
        self.magx2  = magx2
        self.phase2 = phase2
        self.pshift = pshift
        self.bin1D  = bin1D
        self.noise  = noise
        self.binlen = binlen if binlen is not None else DEFAULT_BIN_LENGTH
        self.binarr = np.linspace(0, 1, self.binlen + 1)
        
    def set_lc(self, file: str) -> None:
        '''
        Parameters:
        ----------
        takes a textfile with 3 columns, namely HJD,mag and error:
            
        Returns:
        HJD,mag and error as 1-d arrays 
        
        '''
        try:
            data = np.loadtxt(file, delimiter=' ').astype(np.float64)
            if data.shape[1] != 3:
                raise ValueError(f"Expected 3 columns, got {data.shape[1]}")
            self.file = file
            self.jd = data[:, 0]
            self.mag = data[:, 1]
            self.err = data[:, 2]
            logger.info(f"Successfully loaded lightcurve from {file}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file}")
            raise
        except ValueError as e:
            logger.error(f"Invalid data format in {file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
            raise

    
    def set_noise_frequencies(self, noise_file: str) -> None:
        '''
        Parameters: file contining noisy frequencies
        ----------
        
        There is a provision to reject a set of frequencies from periodogram
        produced by lombscargle, this is to eliminate dominant frequencies 
        present in the data due to instrumental effects such as frequency of 
        the thruster firing (incase data is from any satellite) or the 
        frequency of cadence of the dataset.
        '''
        try:
            self.noise = np.loadtxt(noise_file).astype('float')
            logger.info(f"Loaded {len(self.noise)} noise frequencies from {noise_file}")
        except FileNotFoundError as e:
            logger.error(f"Noise frequency file not found: {noise_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading noise frequencies from {noise_file}: {e}")
            raise
        
    
    def lomb_scargle(self, maximum_frequency: float = MAX_FREQUENCY_DEFAULT) -> float:
        '''
        The LombScargle module from astropy.timeseries is used to compute
        the period of the given data.
        '''
        t, y = self.jd, self.mag
        freq, power = LombScargle(t, y).autopower(maximum_frequency=maximum_frequency)
        
        if self.noise is not None:
            idx = [np.where(freq == i)[0][0] for i in self.noise if i in freq]
            freq = np.delete(freq, idx)
            power = np.delete(power, idx)
            logger.debug(f"Removed {len(idx)} noise frequencies from periodogram")
        
        self.period = 1 / freq[np.argmax(power)]
        self.phase = np.remainder(self.jd, self.period) / self.period
        
        self.phase2 = np.concatenate((self.phase, self.phase + 1))
        self.magx2 = np.concatenate((self.mag, self.mag))
        
        logger.info(f"Detected period: {self.period:.6f}")
        return self.period
                
    
    def polyfit_lc(self, phase: np.ndarray, mag: np.ndarray) -> np.ndarray:
        '''
        Fits an nth order polynomial to the phased lightcurve
        
        Parameters:
        -----------
        phase & magnitude as lists/arrays
        
        Returns:
        A polynomial fit array of the same length as phase/mag
        '''
        cf = np.polyfit(phase, mag, DEFAULT_POLYFIT_DEGREE)
        sc = np.poly1d(cf)
        synth = sc(phase)
        
        return synth


    def phase_shift(self, shift: float) -> List[float]:
        '''
        A small function to shift the calculated phases by a certain
        amount'''
        phases = [i + shift for i in self.phase]
        phshft = []
        for i in phases:
            if i >= 1:
                phshft.append(i - 1)
            else:
                phshft.append(i)
                
        return phshft
    
    
    def build_df(self) -> pd.DataFrame:
        '''returns a pandas dataframe sorted by increasing phases'''
        self.df = pd.DataFrame()
        self.phase = np.remainder(self.jd, self.period) / self.period
        self.df = self.df.assign(**{'MJD': self.jd, 'phase': self.phase,
                                    'mag': self.mag, 'err': self.err})
        self.df = self.df.sort_values(by=['phase'])
        self.df = self.df.reset_index()
        self.df = self.df.drop(['index'], axis=1)
        
        self.jd = np.array(self.df.MJD).astype('float')
        self.phase = np.array(self.df.phase).astype('float')
        self.mag = np.array(self.df.mag).astype('float')
        self.err = np.array(self.df.err).astype('float')
        self.phase2 = np.concatenate((self.phase, self.phase + 1))
        self.magx2 = np.concatenate((self.mag, self.mag))
        
        logger.debug("Built dataframe with sorted phases")
        return self.df

    
    def update_df(self) -> None:
        self.df.phase = self.phase
        self.df = self.df.sort_values(by=['phase'])
        
    
    def phase_correction(self) -> np.ndarray:
        '''
        Parameters:
        -----------
        phase,mag
        
        Returns:
        shifted phase; so that minima falls at zero phase
        
        updates the internal dataframe with the shifted phase
        '''
        self.synth = self.polyfit_lc(self.phase2, self.magx2)
        idx_mn = np.where(self.synth == max(self.synth))[0][0]
        self.pshift = self.phase2[idx_mn]
        if self.pshift > 1:
            self.pshift -= 1
        
        self.phase = self.phase_shift(1 - self.pshift)
        self.update_df()
        
        logger.debug(f"Phase corrected with shift: {self.pshift:.6f}")
        return self.phase

    
    def check_double_period(self) -> bool:
        '''
        Check if the given phase folded lightcurve has double or half period,
        takes a dataframe which contains the columns ['JD','phase','mag','err']
        
        process:
            > dataframe is sorted by period
            > A polynomial is fitted on to the phased lc
            > Variance of the residuals of fit is computed
            > if variance > VARIANCE_THRESHOLD
                > check variance at double and half
                > pick the option with lowest variance
        '''
        sctr = np.var(self.synth - self.magx2)
        
        if sctr < VARIANCE_THRESHOLD:
            logger.info('No period doubling detected')
            return False
            
        else:
            period = self.period * 2
            phase = np.remainder(self.jd, period) / period
            phase_x2 = np.concatenate((phase, phase + 1))
            synth_phse = self.polyfit_lc(phase_x2, mag=self.magx2)
            
            if np.var(synth_phse - self.magx2) < sctr:
                self.period = period
                self.phase = phase
                logger.warning('Period doubling detected - period has been doubled')
                return True
            else:
                logger.info('No period doubling detected')
                return False

    
    def phase_bin(self) -> np.ndarray:
        x = self.binarr
        df = self.df
        bn = []
        
        for i in range(1, len(x)):
            bins = df[df.phase.between(x[i-1], x[i])].mag
            if len(bins) > 0:
                bn.append(np.mean(bins))
            else:
                logger.warning(f"Empty bin at index {i}, using NaN")
                bn.append(np.nan)
        self.bin1D = np.array(bn).astype('float')
        
        logger.debug(f"Phase binned lightcurve into {len(bn)} bins")
        return self.bin1D
    
        
    def normalise(self, array: np.ndarray) -> np.ndarray:
        max_val = max(array)
        if max_val == 0:
            logger.warning("Cannot normalize array with zero maximum")
            return array
        array = np.array(array) / max_val
        return array
        
        
        
    def phased_plot(self, phases: np.ndarray, mag: np.ndarray) -> None:
        phases = np.concatenate((phases, [p + 1 for p in phases]))
        y = np.concatenate((mag, mag))
        plt.style.use('seaborn')
        plt.figure(figsize=(9, 6))
        plt.title('Period : %.6f' % self.period)
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.plot(phases, y, '.k', alpha=0.8)
        plt.show()
        plt.close()
        logger.debug("Generated phased lightcurve plot")
