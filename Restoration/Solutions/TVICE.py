import numpy as np
import scipy.special

def logerfc(t):
    """
    out = logerfc(t): compute an accurate a estimate of log(erfc(t))
    """
    out = np.zeros(t.shape) 
    id = t < 20.
    out[id] = np.log(scipy.special.erfc(t[id])) 
    
    c = np.cumprod(np.arange(1,16,2)/2) 
    t2n0 = t[id==0]**2
    t2n = np.copy(t2n0) 
    S = np.ones((t2n.size,)) 
    
    p = -1
    for n in range(8):
        S = S + (p * c[n]) / t2n 
        t2n = t2n*t2n0 
        p = -p 
    out[id==0] = -t2n0 + np.log(S/(t[id==0]*np.sqrt(np.pi))) 
    return out


def logerf2(a,b):
    """
    usage: out = logerf2(a,b) with a < b
    computes an accurate estimate of log(erf(b)-erf(a))
    """
    a0 = np.copy(a)
    id = (b < 0)
    a[id] = -b[id]
    b[id] = -a0[id]

    out = np.zeros(a.shape) 
    id1 = (b-a)/(np.abs(a)+np.abs(b)) < 1e-14 
    out[id1] = np.log(2*(b[id1]-a[id1])/np.sqrt(np.pi)) - b[id1]**2 

    id2 = (id1==0) & (a<1) 
    out[id2] = np.log(scipy.special.erf(b[id2])-scipy.special.erf(a[id2])) 

    id3 = (id1==0) & (id2==0) 
    m = logerfc(b[id3]) 
    out[id3] = m + np.log(np.expm1(logerfc(a[id3])-m)) 
    
    return out


def tvice(u0,sig,lambd,niter): 
    """
    usage: out = tvice(u0,sigma,lambda,niter) 
    TV-ICE denoising algorithm (vectorized version)
    """
    u = np.copy(u0)
    nr,nc = u0.shape
    v = np.zeros((nr,nc,4)) 
      
    log_X = np.zeros((nr,nc,5)) 
    lambda_nrm = lambd/(2*sig**2) 
    sigma_nrm = sig*np.sqrt(2) 
    

    ## main loop 
    for iter in range(niter):
        
        # update image
        v[:,:,0] = np.hstack([u[:,0:1],u[:,0:-1]])  # right
        v[:,:,1] = np.hstack([u[:,1:] ,u[:,-1:]])    # left
        v[:,:,2] = np.vstack([u[0:1,:],u[0:-1,:]])  # bottom
        v[:,:,3] = np.vstack([u[1:,:] ,u[-1:,:]])    # top
        s = np.sort(v,2)
        A = np.reshape(s[:,:,0],[nr,nc])
        B = np.reshape(s[:,:,1],[nr,nc])
        C = np.reshape(s[:,:,2],[nr,nc])
        D = np.reshape(s[:,:,3],[nr,nc])
        
        log_X[:,:,0] = logerfc((u0-A+2*lambd*2*sig**2)/sigma_nrm) + 2*lambd*(2*(u0+lambd*2*sig**2)-A-B) # log_Xm2
        log_X[:,:,1] = logerf2((A-u0-lambd*2*sig**2)/sigma_nrm,(B-u0-lambd*2*sig**2)/sigma_nrm) + lambd*(2*(u0-B)+2*sig**2*lambd) # log_Xm1
        log_X[:,:,2] = logerf2((B-u0)/sigma_nrm , (C-u0)/sigma_nrm) # log_X0
        log_X[:,:,3] = logerf2((C-u0+lambd*2*sig**2)/sigma_nrm,(D-u0+lambd*2*sig**2)/sigma_nrm) + lambd*(2*(C-u0)+2*sig**2*lambd) # log_Xp1
        log_X[:,:,4] = logerfc((D-u0+2*lambd*2*sig**2)/sigma_nrm) + 2*lambd*(C+D-2*(u0-2*sig**2*lambd)) # log_Xp2
            
        M = np.reshape(np.max(log_X,2),[nr,nc])
        XXm2 = np.exp(log_X[:,:,0]-M)
        XXm1 = np.exp(log_X[:,:,1]-M)
        XX0  = np.exp(log_X[:,:,2]-M)
        XXp1 = np.exp(log_X[:,:,3]-M)
        XXp2 = np.exp(log_X[:,:,4]-M)
        u = u0 + 2*sig**2*lambd * (2*XXm2 + XXm1 - XXp1 - 2*XXp2) / (XXm2 + XXm1 + XX0 + XXp1 + XXp2) 
 
    return u