import numpy as np
import scipy.linalg as spl
import scipy.stats as sps
import ot

# need POT
# pip install POT

def GaussianW2(m0,m1,Sigma0,Sigma1):    # Compute the OT distance between two Gaussians
    Sigma00  = spl.sqrtm(Sigma0).real
    Sigma010 = spl.sqrtm(Sigma00@Sigma1@Sigma00).real
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d

# Compute the OT map between two Gaussians, 
# m0 and m1 must be 2D arrays of size dx1
# Sigma0 and Sigma1 must be 2D arrays of size dxd
# x can be a matrix of size d x n, 
# and each column of x is a vector to which the function is applied 
def GaussianMap(m0,m1,Sigma0,Sigma1,x):
    sqS = spl.sqrtm(Sigma0@Sigma1).real
    Sigma  = np.linalg.pinv(Sigma0, hermitian=True)@sqS
    Tx        = m1+Sigma@(x-m0)
    #Tx         = x-m0+m1     # just translations
    return Tx


# Compute the OT plan between two GMM
def DW2(pi0,pi1,mu0,mu1,S0,S1):       # return the OT map
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    M = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library        
    # Wd = ot.emd2(pi0,pi1,M)            # discrete transport distance
    wstar  = ot.emd(pi0,pi1,M)         # discrete transport plan
    return wstar

# Compute the mixing of two GMM
def MixGMM(pi0,pi1,mu0,mu1,S0,S1,t):
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d = mu0.shape[1]
    M = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library        
    # Wd = ot.emd2(pi0,pi1,M)            # discrete transport distance
    wstar  = ot.emd(pi0,pi1,M)         # discrete transport plan
    # Store mixed GMM
    ncomp = np.sum(wstar>1e-10)
    print("Number of components of GMM OT plan = ", ncomp)
    pi = np.zeros(ncomp)
    mu = np.zeros((ncomp, d))
    S = np.zeros((ncomp, d, d))
    n = 0
    for k in range(K0):
        for l in range(K1):
            if wstar[k, l] > 1e-10:
                pi[n] = wstar[k, l]
                mu[n, :] = (1-t)*mu0[k, :] + t*mu1[l, :]
                sig0 = S0[k,:,:]
                sig1 = S1[l,:,:]
                sqrsig1 = spl.sqrtm(sig1).real
                sqrinv = np.linalg.pinv(spl.sqrtm(sqrsig1 @ sig0 @ sqrsig1).real, hermitian=True)
                C = sqrsig1 @ sqrinv @ sqrsig1
                Ct = (1-t) * np.eye(d) + t * C
                S[n, :, :] = Ct @ sig0 @ Ct
                n = n+1
    return pi, mu, S, wstar
    
# Compute the mixing of two scikit-learn GMM
def MixGMMsk(gmm0, gmm1, t):
    pi0, mu0, S0 = gmm0.weights_, gmm0.means_, gmm0.covariances_
    pi1, mu1, S1 = gmm1.weights_, gmm1.means_, gmm1.covariances_
    pi, mu, S = MixGMM(pi0,pi1,mu0,mu1,S0,S1,t)
    #########
    # XXX Actually not so interesting because one should initialize sklearn GMM with precision matrices.
    #########
