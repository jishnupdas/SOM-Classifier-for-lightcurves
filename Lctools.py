#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:23:05 2019

@author: jishnu
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

#%%
class Lctools:
    '''
    A set of tools to analyse timeseries data(lightcurves)

    '''
    def __init__(self,file=None,jd=None,mag=None,err=None,period=None,
                 phase=None,df=None,synth=None,magx2=None,noise=None,
                 phase2=None,pshift=None,bin1D=None,binlen=None,binarr=None):
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
        self.binlen = 64
        self.binarr = np.linspace(0,1,self.binlen+1)
        
    def set_lc(self,file):
        '''
        Parameters:
        ----------
        takes a textfile with 3 columns, namely HJD,mag and error:
            
        Returns:
        HJD,mag and error as 1-d arrays 
        
        '''
        data=np.loadtxt(file,delimiter=' ').astype(np.float)
        self.file   = file
        self.jd     = data[:,0]
        self.mag    = data[:,1]
        self.err    = data[:,2]

    
    def set_noise_frequencies(self,noise_file):
        '''
        Parameters: file contining noisy frequencies
        ----------
        
        There is a provision to reject a set of frequencies from periodogram
        produced by lombscargle, this is to eliminate dominant frequencies 
        present in the data due to instrumental effects such as frequency of 
        the thruster firing (incase data is from any satellite) or the 
        frequency of cadence of the dataset.
        '''
        self.noise  = np.loadtxt(noise_file).astype('float')
        
    
    def lomb_scargle(self):
        '''
        The LombScargle module from astropy.timeseries is used to compute
        the period of the given data.
        '''
        t,y = self.jd,self.mag
        freq, power = LombScargle(t,y).autopower(maximum_frequency=8)
        
        if self.noise is not None:
            idx   = [np.where(freq == i)[0][0] for i in self.noise if i in freq]
            freq  = np.delete(freq,idx) #  / #
            power = np.delete(power,idx)
        else:
            pass
        
        self.period  = 1/freq[np.argmax(power)]
        self.phase   = np.remainder(self.jd,self.period)/self.period
        
        self.phase2  = np.concatenate((self.phase,self.phase+1))
        self.magx2   = np.concatenate((self.mag,self.mag))
        
        return self.period
                
    
    def polyfit_lc(self,phase,mag):
        '''
        Fits an nth order polynomial to the phased lightcurve
        
        Parameters:
        -----------
        phase & magnitude as lists/arrays
        
        Returns:
        A polynomial fit array of the same length as phase/mag
        '''
        cf = np.polyfit(phase,mag, 30) #20 order polyfit
        sc = np.poly1d(cf) #fitting function
        synth = sc(phase)
        
        return synth


    def phase_shift(self,shift):
        '''
        A small function to shift the calculated phases by a certain
        amount'''
        phases = [i+shift for i in self.phase]
        phshft = []
        for i in phases:
            if i>=1:
                phshft.append(i-1)
            else:
                phshft.append(i)
                
        return phshft
    
    
    def build_df(self):
        '''returns a pandas dataframe sorted by increasing phases'''
        self.df     = pd.DataFrame()
        self.phase  = np.remainder(self.jd,self.period)/self.period
        self.df     = self.df.assign(**{'MJD':self.jd,'phase':self.phase,
                                        'mag':self.mag,'err':self.err})
        self.df     = self.df.sort_values(by=['phase'])
        self.df     = self.df.reset_index()
        self.df     = self.df.drop(['index'],axis=1)
        
        self.jd     = np.array(self.df.MJD).astype('float')
        self.phase  = np.array(self.df.phase).astype('float')
        self.mag    = np.array(self.df.mag).astype('float')
        self.err    = np.array(self.df.err).astype('float')
        self.phase2 = np.concatenate((self.phase,self.phase+1))
        self.magx2  = np.concatenate((self.mag,self.mag))
        
        return self.df

    
    def update_df(self):
        self.df.phase = self.phase
        self.df       = self.df.sort_values(by=['phase'])
        
    
    def phase_correction(self):
        '''
        Parameters:
        -----------
        phase,mag
        
        Returns:
        shifted phase; so that minima falls at zero phase
        
        updates the internal dataframe with the shifted phase
        '''
        self.synth  = self.polyfit_lc(self.phase2,self.magx2)
        idx_mn      = np.where(self.synth == max(self.synth))[0][0]
        self.pshift = self.phase2[idx_mn]
        if self.pshift > 1:
            self.pshift -= 1
        else:
            pass
        self.phase  = self.phase_shift(1-self.pshift)
        self.update_df()
        
        return self.phase

    
    def check_double_period(self):
        '''
        Check if the given phase folded lightcurve has double or half period,
        takes a dataframe which contains the columns ['JD','phase','mag','err']
        
        process:
            > dataframe is sorted by period
            > A polynomial is fitted on to the phased lc
            > Variance of the residuals of fit is computed
            > if variance > 0.0001 
                > check variance at double and half
                > pick the option with lowest variance
        '''
        sctr  = np.var(self.synth-self.magx2)
        
        if sctr < 0.0001:
            print('the lc has no doubling')
            return False
            
        else:
            period     = self.period*2
            phase      = np.remainder(self.jd,period)/period
            phase_x2   = np.concatenate((phase,phase+1))
            synth_phse = self.polyfit_lc(phase_x2,mag=self.magx2)
            
            if np.var(synth_phse-self.magx2) < sctr:
                self.period   = period
                self.phase    = phase
                print('''the lc has doubling 
                      and period value has been doubled''')
                return True
            else:
                return False

    
    def phase_bin(self):
        x  = self.binarr
        df = self.df
        bn = []
        
        for i in range(1,len(x)):
            bins = df[df.phase.between(x[i-1],x[i])].mag
            bn.append(np.mean(bins))
        self.bin1D = np.array(bn).astype('float')
        
        return self.bin1D
    
        
    def normalise(self,array):
        array = np.array(array)/max(array) 
        return array
        
        
        
    def phased_plot(self,phases,mag):
        phases=np.concatenate((phases,[p+1 for p in phases]))
        y=np.concatenate((mag,mag))
        plt.style.use('seaborn')
        plt.figure(figsize = (9,6))
        plt.title('Period : %.6f'%self.period)
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.plot(phases,y,'.k',alpha=0.8)
        plt.show()
        plt.close()
