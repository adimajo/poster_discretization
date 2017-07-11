#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:05:45 2017

@author: adrien
"""

def discretize_link(link,predictors1,predictors2,affectations):
    
    import collections
    import numpy as np
    import sklearn
    from scipy import stats
    
    try:
        n = predictors1.shape[0]
    except AttributeError:
        n = predictors2.shape[0]

    try:
        d_1 = predictors1.shape[1]
    except AttributeError:
        d_1 = 0

    try:
        d_2 = predictors2.shape[1]
    except AttributeError:
        d_2 = 0
    
    d_1bis = [isinstance(x,sklearn.linear_model.logistic.LogisticRegression) for x in link]
    d_2bis = [isinstance(x,collections.Counter) for x in link]
    
    if d_1!=sum(d_1bis): raise ValueError('Shape of predictors1 does not match provided link function')
    if d_2!=sum(d_2bis): raise ValueError('Shape of predictors2 does not match provided link function')
    
    emap = np.array([0]*n*(d_1+d_2)).reshape(n,d_1+d_2)
    
    for j in range(d_1+d_2):
        if d_1bis[j]:
            emap[np.invert(np.isnan(predictors1[:,j])),j] = link[j].predict(predictors1[np.invert(np.isnan(predictors1[:,j])),j].reshape(-1,1))
            emap[np.isnan(predictors1[:,j]),j] = stats.describe(emap[:,j]).minmax[1] + 1
        elif d_2bis[j]:
            m = max(link[j].keys(),key=lambda key: key[1])[1]
            t = np.zeros((n,int(m)+1))
            
            for l in range(n):
                for k in range(int(m)+1):
                    t[l,k] = link[j][(int((affectations[j].transform(np.ravel(predictors2[l,j-d_1])))),k)]/n
        
            emap[:,j] = np.argmax(t,axis=1)
            
        else: raise ValueError('Not quantitative nor qualitative?')
    
#        emap[:,j] = link[j].predict(predictors1[:,j].reshape(-1,1))
#    
#    for j in range(d_2):
#        m = max(link[j+d_1].keys(),key=lambda key: key[1])[1]
#        t = np.zeros((n,int(m)+1))
#        
#        for l in range(n):
#            for k in range(int(m)+1):
#                t[l,k] = link[j+][(int((affectations[j].transform(np.ravel(predictors2[l,j])))),k)]/n
#    
#        emap[:,j] = np.argmax(t,axis=1)

    return emap