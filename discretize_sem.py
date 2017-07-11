#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:15:58 2017

@author: adrien
"""

def discretize_sem(predictors1,predictors2,labels,iter=1000,m_depart=10,criterion="aic"):
    "This discretizes features in predictors according to the labels"
    
    import numpy as np
    import sklearn as sk
    import sklearn.preprocessing
    import sklearn.linear_model
    import time
#    from numpy import *
#    from sklearn import *
    from numpy.random import permutation
    from scipy import stats
    from collections import Counter
    from math import log


    # Calculate shape of predictors (re-used multiple times)
    n = labels.shape[0]
    try:
        d1 = predictors1.shape[1]
    except AttributeError:
        d1 = 0

    try:
        d2 = predictors2.shape[1]
    except AttributeError:
        d2 = 0

    continu_complete_case = np.invert(np.isnan(predictors1))
    sum_continu_complete_case = np.zeros((n,d1))

    for j in range(d1):
        sum_continu_complete_case[0,j] = continu_complete_case[0,j]*1
        for l in range(1,n):
            sum_continu_complete_case[l,j] = sum_continu_complete_case[l-1,j]+continu_complete_case[l,j]*1

    # Initialization for following the performance of the discretization
    criterion_iter=[]
    current_best = 0
    
    # Initial random "discretization"
    affectations = [None]*(d1+d2)
    edisc = np.random.choice(list(range(m_depart)),size=(n,d1+d2))
    
    for j in range(d1):
        edisc[np.invert(continu_complete_case[:,j]),j] = m_depart
              
    predictors_trans = np.zeros((n,d2))
    
    for j in range(d2):
        affectations[j+d1] = sk.preprocessing.LabelEncoder().fit(predictors2[:,j])
        if (m_depart > stats.describe(sk.preprocessing.LabelEncoder().fit(predictors2[:,j]).transform(predictors2[:,j])).minmax[1]):
            edisc[:,j+d1] = np.random.choice(list(range(stats.describe(sk.preprocessing.LabelEncoder().fit(predictors2[:,j]).transform(predictors2[:,j])).minmax[1]-1)), size = n)
        else:
            edisc[:,j+d1] = np.random.choice(list(range(m_depart)),size = n)
        
        predictors_trans[:,j] = (affectations[j+d1].transform(predictors2[:,j])).astype(int)
    
    emap = np.ndarray.copy(edisc)

    # Loop till iter is reached
    for i in range(iter):

        model_reglog_eq = sk.linear_model.LogisticRegression(solver='liblinear',C=1e40,tol=1e-8,max_iter=25,warm_start=True)
        model_reglog = model_reglog_eq.fit(X=sk.preprocessing.OneHotEncoder().fit_transform(X=emap.astype(str)),y=labels)
        
        logit_eq = sk.linear_model.LogisticRegression(solver='liblinear',C=1e40,tol=1e-8,max_iter=25,warm_start=True)
        logit = logit_eq.fit(X=sk.preprocessing.OneHotEncoder().fit_transform(X=edisc.astype(str)),y=labels)
        
        # criterion_iter: for now, supports only AIC criterion
        if criterion == "aic":
            criterion_iter.append(2*model_reglog.coef_.shape[1] + 2*sk.metrics.log_loss(labels,model_reglog.predict_proba(sk.preprocessing.OneHotEncoder().fit_transform(X=emap.astype(str))), normalize=False))
        else:
            criterion_iter.append(log(n)*model_reglog.coef_.shape[1] + 2*sk.metrics.log_loss(labels,model_reglog.predict_proba(sk.preprocessing.OneHotEncoder().fit_transform(X=emap.astype(str))), normalize=False))

        
        # if AIC better than the current best model, update current best
        if (criterion_iter[i] <= criterion_iter[current_best]):
            
            # Update current best logistic regression
            best_reglog = model_reglog
            
            # For the first iteration, link is not present in the environment yet
            try:
                best_link = link
            except NameError:
                best_link = []
            current_best = i
        
        # Initialize the list of link functions (regressions fitted from the continuous variables on the discretized ones) at each iteration
        link_eq = [None]*(d1+d2)
        link = [None]*(d1+d2)
        # Number of factor levels
#        m = stats.describe(edisc).minmax[1]
        m=[None]*(d1+d2)
        
        for j in range(d1+d2):
            m[j] = np.unique(edisc[:,j])

    
        base_disjonctive = sk.preprocessing.OneHotEncoder().fit_transform(X=edisc.astype(str)).toarray()

        for j in permutation(d1+d2):
            
            if (j<d1):
            
                start_time = time.clock()
                
                link_eq[j] = sk.linear_model.LogisticRegression(C=1e40,multi_class='multinomial',solver='newton-cg',max_iter=25,warm_start=True)
                link[j] = link_eq[j].fit(y=edisc[continu_complete_case[:,j],j],X=predictors1[continu_complete_case[:,j],j].reshape(-1,1))
                
                print(time.clock() - start_time, "seconds : polynomial logistic regression")

                start_time = time.clock()

                y_p = np.zeros((n,len(m[j])))
                
                for k in range(len(m[j])):
                    modalites = np.zeros((n,len(m[j])))
                    modalites[:,k] = np.ones((n,))
    
#                    try:
#                        y_p[:,k] = logit.predict_proba(np.column_stack((sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,0:j].astype(str)).toarray(),modalites,sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,(j+1):(d1+d2)].astype(str)).toarray())))[:,1]*(2*labels-1)-labels+1
#                    except ValueError:
#                        if (j<1):
#                            if ((d1+d2)>1):
#                                y_p[:,k] = logit.predict_proba(np.column_stack((modalites,sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,(j+1):(d1+d2)].astype(str)).toarray())))[:,1]*(2*labels-1)-labels+1
#                            else:
#                                y_p[:,k] = logit.predict_proba(modalites)[:,1]*(2*labels-1)-labels+1
#                        else:
#                            y_p[:,k] = logit.predict_proba(np.column_stack((sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,0:j].astype(str)).toarray(),modalites)))[:,1]*(2*labels-1)-labels+1
    
                    y_p[:,k] = logit.predict_proba(np.column_stack((base_disjonctive[:,0:(sum(list(map(len,m[0:j]))))],modalites,base_disjonctive[:,(sum(list(map(len,m[0:(j+1)])))):(sum(list(map(len,m))))])))[:,1]*(2*np.ravel(labels)-1)-np.ravel(labels)+1
    
                print(time.clock() - start_time, "seconds : p(y|e) calculation")
                
                start_time1 = time.clock()

                t = link[j].predict_proba(predictors1[(continu_complete_case[:,j]),j].reshape(-1,1))
                
                print(time.clock() - start_time1, "seconds : t calculation")

                start_time = time.clock()

                emap[(continu_complete_case[:,j]),j] = np.argmax(t,axis=1)
                emap[np.invert(continu_complete_case[:,j]),j] = m[j][-1]
                
                print(time.clock() - start_time, "seconds : emap calculation")

                start_time = time.clock()

                if (np.invert(continu_complete_case[:,j]).sum() == 0):
                    t = t*y_p
                else:
                    t = t*y_p[continu_complete_case[:,j],0:(len(m[j])-1)]
                
                t = t/(t.sum(axis=1)[:,None])
                
                print(time.clock() - start_time, "seconds : new t calculation")

                start_time = time.clock()
                
                for l in range(n):
                    if continu_complete_case[l,j]:
                        edisc[l,j] = (np.random.multinomial(1,t[sum_continu_complete_case[l,j]-1,:],size=1)==1).argmax()
                    else:
                        edisc[l,j] = m[j][-1]

                print(time.clock() - start_time, "seconds : edisc calculation")
                             
                print(time.clock() - start_time1, "seconds : updating emap edisc")

            else:

                start_time = time.clock()
                
                link[j] = Counter([tuple(element) for element in np.column_stack((affectations[j].transform(predictors2[:,j-d1]),edisc[:,j]))])
                y_p = np.zeros((n,len(m[j])))

                print(time.clock() - start_time, "seconds : tableau de contingence")

                start_time = time.clock()
                
                for k in range(len(m[j])):
                    modalites = np.zeros((n,len(m[j])))
                    modalites[:,k] = np.ones((n,))
    
#                    try:
#                        y_p[:,k] = logit.predict_proba(np.column_stack((sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,0:j].astype(str)).toarray(),modalites,sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,(j+1):(d1+d2)].astype(str)).toarray())))[:,1]*(2*labels-1)-labels+1
#                    except ValueError:
#                        if (j<1):
#                            if ((d1+d2)>1):
#                                y_p[:,k] = logit.predict_proba(np.column_stack((modalites,sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,(j+1):(d1+d2)].astype(str)).toarray())))[:,1]*(2*labels-1)-labels+1
#                            else:
#                                y_p[:,k] = logit.predict_proba(modalites)[:,1]*(2*labels-1)-labels+1
#                        else:
#                            y_p[:,k] = logit.predict_proba(np.column_stack((sk.preprocessing.OneHotEncoder().fit_transform(X=edisc[:,0:j].astype(str)).toarray(),modalites)))[:,1]*(2*labels-1)-labels+1

                    y_p[:,k] = logit.predict_proba(np.column_stack((base_disjonctive[:,0:(sum(list(map(len,m[0:j]))))],modalites,base_disjonctive[:,(sum(list(map(len,m[0:(j+1)])))):(sum(list(map(len,m))))])))[:,1]*(2*np.ravel(labels)-1)-np.ravel(labels)+1


                print(time.clock() - start_time, "seconds : p(y|e) calculation")

                start_time = time.clock()
                
                t = np.zeros((n,int(len(m[j]))))
    
                start_time = time.clock()
    
                for l in range(n):
                    for k in range(int(len(m[j]))):
                        t[l,k] = link[j][(predictors_trans[l,j-d1],k)]/n
                         
                emap[:,j] = np.argmax(t,axis=1)
                
                t = t*y_p
                
                t = t/(t.sum(axis=1)[:,None])
                
                for l in range(n):
                    edisc[l,j] = (np.random.multinomial(1,t[l,:],size=1)==1).argmax()

                print(time.clock() - start_time, "seconds : updating emap edisc")

        
    return [criterion_iter,best_link,best_reglog,affectations]
