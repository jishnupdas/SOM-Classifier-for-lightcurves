#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:51:23 2019

@author: jishnu
"""

#%%
import os
import glob
import pickle
import numpy as np
import seaborn as sns
from minisom import MiniSom 
import matplotlib.pyplot as plt
from scipy import signal
#%%
dpath = '/home/jishnu/Documents/TESS/tess_data/1D_lc/'

files = glob.glob(dpath+'*')

#%%
get_array = lambda file: np.loadtxt(file)


def get_arr(file):
    
    data  = np.loadtxt(file)
    if np.isnan(data).any() == True:
#        files.remove(file)
        return np.nan
    else:
        return data

#%%
def OneDplot(xlen,y):
    xarr  = np.linspace(0,1,xlen)
    plt.plot(xarr,y)
    plt.show()
    plt.close()

#%%

name  = [f for f in files if get_arr(f) is not np.nan]

data  = [get_arr(f) for f in name]


#%%
#for f in data[:99]:
#    OneDplot(32,f)

#%%
som   = MiniSom(50,50, 32, sigma=0.1,learning_rate=1.5) 
# initialization of 40x40 SOM
#som.pca_weights_init(data)
#%%
som.random_weights_init(data)
som.train_random(data, 10000) # trains the SOM with 100 iterations


#with open('som.p', 'wb') as outfile:
#    pickle.dump(som, outfile)
    
#%%    
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
    

#%%
coords = []
err    = []
for d in data:
    try:
        coords.append(np.array(som.winner(d)))
    except:
        print("err with ",str(d))
        err.append(d)

#%%
x,y  = [i[0] for i in coords],[i[1] for i in coords]
plt.style.use('seaborn')
plt.plot(x,y,'.',alpha=0.15)
sns.kdeplot(x,y,cmap='Blues',shade=True,bw=2,shade_lowest=False,alpha=0.8)
plt.show()
plt.close()


#%%
"""REF: https://towardsdatascience.com/an-introduction-to-clustering-
    algorithms-in-python-123438574097"""

from sklearn.cluster import KMeans

points = coords

kmeans = KMeans(n_clusters=15)

kmeans.fit(points)

y_km   = kmeans.fit_predict(points)

#%%
def cluster_plot(l,mask1):
    for i in range(l):
        plt.plot(np.ma.masked_array(data=x,mask = np.invert((mask1 ==i,0)[0])), 
                 np.ma.masked_array(data=y,mask = np.invert((mask1 ==i,1)[0])),
                 '.')

#%%

cluster_x  = [i[0] for i in kmeans.cluster_centers_]
cluster_y  = [i[1] for i in kmeans.cluster_centers_]

#plt.plot(x,y,'.',alpha=0.15)

#plt.plot(cluster_x,cluster_y,'o')
cluster_plot(15,y_km)
sns.kdeplot(x,y,cmap='Blues',shade=True,shade_lowest=False,bw=2,alpha=0.6)
plt.show()
plt.close()

#%%
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))

# create clusters
hc = AgglomerativeClustering(n_clusters=15, affinity = 'euclidean',
                             linkage = 'ward')

# save clusters for chart
y_hc = hc.fit_predict(points)

#%%
cluster_plot(6,y_hc)
sns.kdeplot(x,y,cmap='Blues',shade=False,bw=2,alpha=0.5)
plt.show()
plt.close()
