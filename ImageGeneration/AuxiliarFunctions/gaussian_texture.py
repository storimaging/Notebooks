# Copyright Arthur Leclaire (c), 2019.

import numpy as np

def adsn(s,mu):
    # Compute an Asymptotic Discrete Spot Noise texture.
    #
    #   out = adsn(s,mu) computes a realization of the Gaussian stationary 
    #   random field of mean mu and whose covariance function is 
    #   the autocorrelation of s.
    #   The output is of same size as the mean image mu.
    #
    #   Notice that the covariance of the resulting field is 
    #   the non-periodic autocorrelation of s.
    #
    #   NB :
    #   - The mean value of s is not substracted.
    #   - The input s can have multiple channels.
    
    M,N,C = mu.shape
    m,n,c = s.shape
    out = adsn_periodic(s,np.zeros((M+m,N+n,c)))
    out = mu + out[0:M,0:N,:]
    return out

def adsn_periodic(s,mu):
    # Compute a periodic Asymptotic Discrete Spot Noise texture.
    #
    #   out = adsn_periodic(s,mu) 
    #
    #   computes a realization of the Gaussian 
    #   circularly stationary random field of mean mu and whose 
    #   covariance function is the periodic autocorrelation of s.
    #
    #   Notice that the covariance of the resulting field is the
    #   periodic autocorrelation of s.
    #
    #   NB :
    #   - The mean value of s is not substracted.
    #   - If size(s,1)>M or size(s,2)>N , then s is cropped.
    #   - The input s can have multiple channels.
    #
    #   This texture model is presented in the paper
    #       "Random Phase Textures: Theory and Synthesis", 
    #       (B. Galerne, Y. Gousseau, J.-M. Morel), 
    #       IEEE Transactions on Image Processing, 2011.
    
    M,N,C = mu.shape
    m,n,c = s.shape
    if m>M:
        s = s[0:M,:,:]
    elif n>N:
        s = s[:,0:N,:]
    s = zeropad(s,M,N)
    m,n,c = s.shape
    out = np.zeros((M,N,C))
    W = np.random.randn(M,N,1)
    W = np.tile(W,(1,1,C))
    fW = np.fft.fft2(W,axes=(0,1))
    fs = np.fft.fft2(s,axes=(0,1))
    out = mu+np.real(np.fft.ifft2(fW*fs,axes=(0,1)))
    return out

def estimate_adsn_model(u):
    # Compute the mean and texton associated with u
    #   [ t,m ] = estimate_adsn_model( u,Mb,Nb,maskrgb )
    #   
    #   INPUT
    #   u       Original texture image
    #   Mb,Nb   (optional) The texton is embedded in a MbxNb image
    #   maskrgb (optional) Estimate the ADSN model outside a mask
    #       
    #   OUTPUT
    #   t       Texton of the ADSN model
    #   m       Mean value of the ADSN model
    
    M,N,C = u.shape
    mv = np.mean(u,(0,1))
    t = (u-mv)/np.sqrt(M*N)
    return t,mv

def zeropad(u,M,N):
    # v = zeropad( u,M,N )
    #   Extend u by zero on a domain of size M x N
    
    if u.ndim==3:
        m,n,C = u.shape
        v = np.zeros((M,N,C))
        v[0:m,0:n,:] = np.copy(u)
    else:
        m,n = u.shape
        v = np.zeros((M,N))
        v[0:m,0:n] = np.copy(u)
    
    return v

def get_covariance_adsn(t,ind,per=0):
    # S = get_covariance_adsn(t,ind,per=0):
    #   Compute the covariance matrix associated to a Gaussian texture model.
    #
    #   INPUT
    #   t        Texton of the ADSN model
    #   ind      Indicator function of the desired points
    #   per      1 to work with periodic ADSN or 0 otherwise.
    
    M,N,C = t.shape
    if per==0:
        M *= 2
        N *= 2
        t = zeropad(t,M,N)
        ind = zeropad(ind,M,N)
    
    covt = np.zeros((M,N,C,C))
    ft = np.fft.fft2(t, axes=(0,1))
    for a in range(0,C):
        for b in range(0,C):
            covt[:,:,a,b] = np.real(np.fft.ifft2(np.conj(ft[:,:,a])*ft[:,:,b], axes=(0,1)))
    ca, cb = np.nonzero(ind>0.5)
    cn = ca.size
    S = np.zeros((cn*C,cn*C))
    for j in range(0,cn):
        for a in range(0,C):
            for b in range(0,C):
                sa,sb = cn*a,cn*b
                covtt = np.roll(covt[:,:,a,b],(ca[j],cb[j]))
                S[sa+j,sb:sb+cn] = covtt[ca,cb].reshape(cn)
    return S

def mixing(t0, mv0, t1, mv1, alpha):
    # t,m = mixing(t0, mv0, t1, mv1, alpha)
    # Compute the barycenter of Gaussian texture models
    #  (t0, mv0), (t1, mv1) with coefficient alpha in [0,1].
    
    m0, n0, c = t0.shape
    m1, n1, _ = t1.shape
    m = max(m0, m1)
    n = max(n0, n1)
    
    t0 = zeropad(t0, m, n)
    t1 = zeropad(t1, m, n)

    # Mean value of mixed model
    mv = (1-alpha)*mv0 + alpha*mv1

    # Texton of mixed model
    ft0 = np.fft.fft2(t0, axes=(0,1))
    ft1 = np.fft.fft2(t1, axes=(0,1))
    phshift = np.exp(1j* np.angle(np.sum(np.conj(ft0)*ft1, axis=2)))
    phshift = np.tile(phshift[:,:,None], (1,1,c))
    ft = (1-alpha)*phshift*ft0 + alpha*ft1
    t = np.real(np.fft.ifft2(ft, axes=(0,1)))

    return t, mv

