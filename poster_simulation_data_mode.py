#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:55:21 2017

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
n = 1000
d = 2
test=False
validation=False
criterion = 'bic'
iter = 300
m_depart = 10

m_1 = []
m_2 = []

for j in range(50):
    
    # Génération données d'apprentissage
    random.seed(j)
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
    essai = discretize_sem(x,[],y,iter,m_depart,criterion)
    print(time.clock() - start_time, "seconds")
    
    # Discrétisation de l'ensemble d'apprentissage
    emap = discretize_link(essai[1],x,[],[])
    
    m_1.append(len(np.unique(emap[:,0])))
    m_2.append(len(np.unique(emap[:,1])))

