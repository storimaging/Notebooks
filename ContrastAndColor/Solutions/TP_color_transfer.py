import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def todo_specification_separate_channels(u,v):
    nrowu,ncolu,nchu = u.shape
    w = np.zeros(u.shape)
    for i in range(3):
        uch = u[:,:,i]
        vch = v[:,:,i]
        u_sort,index_u=np.sort(uch,axis=None),np.argsort(uch,axis=None)
        v_sort,index_v=np.sort(vch,axis=None),np.argsort(vch,axis=None)
        uspecifv= np.zeros(nrowu*ncolu)
        uspecifv[index_u] = v_sort
        uspecifv = uspecifv.reshape(nrowu,ncolu)   
        w[:,:,i] = uspecifv.reshape(nrowu,ncolu)
    return w


def transport1D(X,Y):
    sx = np.argsort(X) #argsort retourne les indices des valeurs s'ils étaient ordonnés par ordre croissant   
    sy = np.argsort(Y)
    return((sx,sy)) 

def todo_transport3D(X,Y,N,eps): #X,y,Z are nx3 matrices
    Z=np.copy(X) # output
    for k in range(N):
        u=np.random.randn(3,3)
        q=np.linalg.qr(u)[0] #orthonormal basis with uniform distibution on the sphere 
        for i in range(3):
            # projection on the basis 
            Yt=np.dot(Y,q[:,i])
            Zt=np.dot(Z,q[:,i])
            #Permutations
            [sZ,sY]=transport1D(Zt,Yt)
            Z[sZ,:] += eps * (Yt[sY]-Zt[sZ])[:,None] * q[:,i][None,:] # 3D transport
            # equivalent to
            #for j in range(X.shape[0]):
            #    Z[sZ[j],:]=Z[sZ[j],:]+e*(Yt[sY[j]]-Zt[sZ[j]])*(q[:,i]) #transport 3D
        
    return Z,sZ,sY


def average_filter(u,r):    # implementation with integral images
    # uniform filter with a square (2*r+1)x(2*r+1) window 
    # u is a 2d image
    # r is the radius for the filter
   
    (nrow, ncol)                                      = u.shape
    big_uint                                          = np.zeros((nrow+2*r+1,ncol+2*r+1))
    big_uint[r+1:nrow+r+1,r+1:ncol+r+1]               = u
    big_uint                                          = np.cumsum(np.cumsum(big_uint,0),1)       # integral image
        
    out = big_uint[2*r+1:nrow+2*r+1,2*r+1:ncol+2*r+1] + big_uint[0:nrow,0:ncol] - big_uint[0:nrow,2*r+1:ncol+2*r+1] - big_uint[2*r+1:nrow+2*r+1,0:ncol]
    out = out/(2*r+1)**2
    
    return out

def todo_guided_filter(u,guide,r,eps):
    C           = average_filter(np.ones(u.shape), r)   # to avoid image edges pb 
    mean_u      = average_filter(u, r)/C
    mean_guide  = average_filter(guide, r)/C
    corr_guide  = average_filter(guide*guide, r)/C
    corr_uguide = average_filter(u*guide, r)/C
    var_guide   = corr_guide - mean_guide * mean_guide
    cov_uguide  = corr_uguide - mean_u * mean_guide

    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide

    mean_alph = average_filter(alph, r)/C
    mean_beta = average_filter(beta, r)/C

    q = mean_alph * guide + mean_beta
    return q
