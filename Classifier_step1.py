#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:37:52 2019

@author: jishnu
"""

#%%
import os
import Lctools
import pandas as pd
import matplotlib.pyplot as plt

#%%

fpath      = '/media/jishnu/ExternalHDD/TESSDATA/lc_sector1/clean/'

df         = pd.read_csv('low_period.csv')

filenames  = list(df.filename)
files      = [fpath+f for f in filenames] #make a list with fullpath to file


#svpath     = '/home/astronomy/Documents/test_tess/classifierNN/1D_lc/'
#impath     = '/home/astronomy/Documents/test_tess/classifierNN/1D_lc_plots/'

svpath     = '/home/jishnu/Documents/TESS/tess_data/1D_lc/'
impath     = '/home/jishnu/Documents/TESS/tess_data/1D_lc_plots/'


#%%
def main(file):
    '''
    parameters:
    -----------
    file   : a text file with 3 columns; MJD,mag & mag_error
    
    function:
    ---------
    performs lombscargle on the timeseries data, 
    computes the period after removing any noise frequencies
    checks if the period is double the actual one
    shifts the phase so the primary minima falls at 0 phase
    perform a phase binning with 64 bins (mean)
    writes the 1d array to a file
    which will be fed into the classifier NN
    
    Returns:
    -------
    1D phase binned array
    
    writes phased lc image to disk
    
    '''
    fname  = file.split('/')[-1]
    
    tgt    = Lctools.Lctools()
    tgt.set_lc(file)
    tgt.set_noise_frequencies('noise_freq')
    
    period = tgt.lomb_scargle()
    df     = tgt.build_df()
    ph     = tgt.phase_correction()
    
    status = tgt.check_double_period()
#    status = tgt.check_period_spline()
    
    
    if status is not False:
        df = tgt.build_df()
        ph = tgt.phase_correction()
    else:
        pass
    
    binned = tgt.normalise(tgt.phase_bin())
#    print('normalized binned lc of %s'%fname)
    
    with open('logs_LS','a+') as log:
        log.write(fname+','+str(tgt.period)+','+str(status)+'\n')
        
#    print('writing 1D phase binned lc to file')
    with open(svpath+'1D_'+fname,'w+') as LCarray:
        for mag in binned:
            LCarray.write(str(mag)+'\n')
    
#    plot_phased_lc(df.phase,df.mag,tgt.period,impath+fname)
#    del period,df,ph
    
    
    
#%%
def plot_phased_lc(phases,mag,period,name):
    plt.style.use('seaborn')
    plt.figure(figsize = (4,3))
    plt.title('Period : %.6f'%period)
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.gca().invert_yaxis()
    plt.plot(phases,mag,'.k',alpha=0.8)
    plt.savefig(name+'_s.png',dpi=100)
#    plt.show()
    plt.close()

#%%
for file in files:
    try:
        fname  = file.split('/')[-1]
        if os.path.exists(svpath+'1D_'+fname) == False:
            main(file)
    except:
        with open('errorlog','a+') as errlog:
            errlog.write(file+'\n')
            print('encountered an error with file %s'%file.split('/')[-1])
print('Done!')

  
