#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:25:17 2017

@author: adrien
"""

# Importations
import time
import random
import numpy as np
import pandas as pd
import sklearn as sk

####### Variables quantitatives

# Paramètres
n = 800
d = 2
test=False
validation=False
criterion = 'bic'
iter = 200
m_depart = 10

# Génération données d'apprentissage
random.seed(1)
x = np.array(np.random.uniform(size=(n,d)))
xd = np.ndarray.copy(x)
cuts = ([0,0.333,0.666,1])

for i in range(d):
    xd[:,i] = pd.cut(x[:,i],bins=cuts,labels=[0,1,2])
    
theta = np.array([[1]*d]*(len(cuts)-1))
theta[1,:] = 2
theta[2,:] = -2

log_odd = np.array([0]*n)
for i in range(n):
    for j in range(d):
        log_odd[i] += theta[int(xd[i,j]),j]

p = 1/(1+np.exp(-log_odd))
y = np.random.binomial(1,p)


# Entraînement de la discrétisation
start_time = time.clock()
essai = disc_sem(x,y,iter,m_depart)
print(time.clock() - start_time, "seconds")

# Discrétisation de l'ensemble d'apprentissage
emap = discretize_link(essai[2],x)

# Résultats
import matplotlib.pyplot as plt

plt.plot(emap.astype(str)[:,0],x[:,0].reshape(-1,1),'ro')

plt.plot(emap.astype(str)[:,1],x[:,1].reshape(-1,1),'ro')

plt.plot(emap.astype(str)[:,2],x[:,2].reshape(-1,1),'ro')

plt.plot(emap.astype(str)[:,3],x[:,3].reshape(-1,1),'ro')