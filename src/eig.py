import numpy as np
from pfeautil import *
from math import *
import scipy as sp

#The sparse solver
import cvxopt as co
from cvxopt import cholmod

#Note to self I am going to transcibe exactly the frame3dd subspace method
#Must come back and make better using default functions and so forth
def subspace(K,M,tot_dof,n_modes,V,tol):
    
    if (n_modes>tot_dof):
        print("subspace: Number of eigen-values must be less than the problem dimension.\n Desired number of eigen-values=%d \n Dimension of the problem= %d \n",n_modes,tot_dof)
        
    d = co.matrix(0.0,(K.size[1],1))
    u = co.matrix(0.0,(K.size[1],1))
    v = co.matrix(0.0,(K.size[1],1))
    Kb = co.matrix(0.0,(n_modes,n_modes))
    Mb = co.matrix(0.0,(n_modes,n_modes))
    Xb = co.matrix(0.0,(n_modes,n_modes))
    Qb = co.matrix(0.0,(n_modes,n_modes))
    idx = co.matrix(0.0,(n_modes,1))
    
    if 0.5*n_modes>n_modes-8:
        modes = n_modes/2.0
    else:
        modes = n_nodes-8
    
    try:
        cholmod.linsolve(K,v)
        xq = v
    except Exception,e:
        print(type(e))
        print(e)
        print("Warning: Cholesky did not work")
        xq = sp.linalg.solve(K,v)
    
    I = co.matrix(range(0,len(K),K.size[1]))
    d = co.div(K[I],M[I])
    
    km_old = 0.0
    for k in range(1,n_modes):
        km = d[1]
        for i in range(1,tot_dof):
            if km <= d[i] & d[i] <= km:
                ok = True
                for j in range(1,k-1):
                    if i == idx[j]:
                        ok = False
                if ok:
                    km = d[i]
                    idx[k] = i
        if idx[k] == 0:
            i = idx[1]
            for j in range(1,k):
                if i < idx[j]:
                    i = idx[j]
            idx[k] = i+1
            km = d[i+1]
        km_old = km
        
    for k in range(1,n_modes):
        V[idx[k]][k] = 1.0
        idxMod = idx[k]%6
        if idxMod == 1:
            i = 1
            j = 2
        elif idxMod == 2:
            i = -1
            j = 1
        elif idxMod == 3:
            i = -1
            j = -2
        elif idxMod == 4:
            i = 1
            j = 2
        elif idxMod == 5:
            i = -1
            j = 1
        elif idxMod == 6:
            i = -1
            j = -2
        V[idx[k]+1][k] = 0.2
        V[idx[k]+j][k] = 0.2
        
    iter = 0
    error = 1
    while error>tol:
        for k in range(1,n_modes):
            v = M*V[:,j];
            try:
                cholmod.linsolve(K,v)
                d = v
            except Exception,e:
                print(type(e))
                print(e)
                print("Warning: Cholesky did not work")
                d = sp.linalg.solve(K,v)
            #Come back and start with the ldl_mprove because adding an i to the begining would make it clear...
    return w
