#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# In[2]:


def create_random_preferences(M, W, T=True, I=True):
    prefs = np.zeros((M,W))
    
    for i in range(M):
        row = np.empty(W)
        row[:] = np.inf
        rank = 0
        non_I = np.random.randint(W//2,W+1) if I else W
        for j in range(non_I):
            row[j] = rank
            rank += np.random.randint(0,2) if T else 1
        prefs[i] = np.random.permutation(row)
    return prefs


# In[3]:


def stable_marriage2(men_prefs, women_prefs):
    M = men_prefs.shape[0]
    W = women_prefs.shape[0]
    x = np.zeros((M,W))
    men_prefs = np.copy(men_prefs)
    women_prefs = np.copy(women_prefs)
    
    bachelors = [i for i in range(M)]
    
    while bachelors:
        proposals = [np.random.choice(np.where(i==np.min(i))[0]) for i in men_prefs]
        free_men = [i for i in bachelors if not (men_prefs[i]==np.inf).all()]
        if not free_men:
            break

        for man in free_men:
            wife = proposals[man]
            if women_prefs[wife,man] == np.inf:
                men_prefs[man,wife] = np.inf
            else:
                if np.sum(x[:,wife])>0:
                    old_husband = np.where(x[:,wife]==1)[0][0]
                    x[old_husband,wife]=0
                    bachelors.append(old_husband)
    
                x[man,wife] = 1
                successors = women_prefs[wife] >= women_prefs[wife,man]
                men_prefs[successors,wife] = np.inf
                women_prefs[wife][successors] = np.inf
                bachelors.remove(man)
        
            
    return x


# In[4]:


def is_weakly_stable(x, men_prefs, women_prefs):
    M,W = x.shape
    man_ismatched = np.sum(x,axis=1)
    woman_ismatched = np.sum(x,axis=0)
    if (((women_prefs.T) + men_prefs)[np.where(x==1)] == np.inf).any():
        print('impossible match')
        return False
    for i in range(M):
        for j in range(W):
            if men_prefs[i,j] == np.inf or women_prefs[j,i] == np.inf:
                None
            else:
                
                current_wife = np.where(x[i]==1)[0] if man_ismatched[i]==1 else 0
                current_husband = np.where(x[:,j]==1)[0] if woman_ismatched[j]==1 else 0

                if (men_prefs[i,j]<men_prefs[i,current_wife] or man_ismatched[i]==0) and (women_prefs[j,i]<women_prefs[j,current_husband] or woman_ismatched[j]==0):
                    return False
                
    return True


# In[5]:

def stability_constraints(men_preferences,women_preferences):
    M = men_preferences.shape[0]
    W = women_preferences.shape[0]
    
    F = [((men_pref[i]!=np.inf)*(women_pref[:,i]!=np.inf)).nonzero()[0] for i in range(M)]
    C = [((women_pref[i]!=np.inf)*(men_pref[:,i]!=np.inf)).nonzero()[0] for i in range(W)]
    
    men_stability = [[(men_preferences[j] <= men_preferences[j,i]).nonzero()[0] for i in F[j]] for j in range(M)]
    women_stability = [[(women_preferences[j] <= women_preferences[j,i]).nonzero()[0] for i in C[j]] for j in range(W)]
    
    constraints = np.zeros((M*W,M*W))
    
    for i in range(M):
        for ind, j in enumerate(F[i]):
            x = np.zeros((M,W))
            
            if women_preferences[j][i] != np.inf:
                man_ind = np.where(C[j]==i)[0][0]
                #print(man_ind)
                w = women_stability[j][man_ind]
            else:
                w=[]
            
            m = men_stability[i][ind]
            for k in w:
                x[k,j] += 1
            for k in m:
                x[i,k] += 1

            #x[i,j] -= 1
            constraints[i*M+j,:] = x.reshape(1,-1)
            
    #print(constraints)
    return constraints


# In[168]:


def other_constraints(M,W):
    x = np.zeros((M+W,M*W))
    row = np.pad(np.ones(W), (0,(M-1)*W))
    for i in range(M):
        x[i,:] = row
        row = np.roll(row,W)
        
    row = np.tile(np.identity(W)[0],M)
    for i in range(W):
        x[M+i,:] = row
        row = np.roll(row,1)
    return x


# In[8]:


from pulp import *


def summary(model, variables, M, W):
    print(model.status,'\n') #1 if optimal solution was found
    print(f"objective: {np.round(model.objective.value(),4)}\n") #objective function optimal value
    x = np.array([i.value() for i in variables]).reshape(M,W)
    print(x)
    return x


# In[320]:


def IP_MAX_SMTI(men_prefs, women_prefs):
    M, W = men_prefs.shape
    s_constraints = stability_constraints(men_prefs, women_prefs)
    o_constraints = other_constraints(M,W)
    impossible_matches = (((women_prefs.T) + men_prefs)==np.inf).reshape(1,-1)[0]
    
    prob = LpProblem("MAX-SMTI", sense=LpMaximize)
    matches = LpVariable.dicts("match", (range(M), range(W)), cat="Binary")
    prob += lpSum(matches)
    
    variables = sum([list(matches[i].values()) for i in matches], [])
    for i in range(len(s_constraints)):
        prob += lpSum([s_constraints[i][j]*variables[j] for j in range(M*W)]) >= 1

    for i in range(M+W):
        prob += lpSum([o_constraints[i][j]*variables[j] for j in range(M*W)]) <= 1

    for i in range(M*W):
        if impossible_matches[i] != 0:
            prob += impossible_matches[i]*variables[i] == 0
            
    return prob, variables



