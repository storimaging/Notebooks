import numpy as np
import torch
import scipy.linalg as spl
import scipy.stats as sps
import ot
from sklearn.neighbors import NearestNeighbors


##################################################
# Functions for GMM Optimal Transport
##################################################

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

def GMM_Optimal_Transport_NN(gmm0,gmm1, X, Y):
    # Extract components
    pi0,mu0,S0 = gmm0.weights_, gmm0.means_, gmm0.covariances_
    pi1,mu1,S1 = gmm1.weights_, gmm1.means_, gmm1.covariances_
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    
    # Compute the K0xK1 OT matrix between the members of the mixtures
    wstar = DW2(pi0/np.sum(pi0),pi1/np.sum(pi1),mu0,mu1,S0,S1)

    # Predict class probabilities in source GMM
    ProbaClassesX = gmm0.predict_proba(X)

    # Apply transport map to all patches
    # Compute mean color transfer on all points
    Np = X.shape[0]
    pdim = X.shape[1]
    Tmeanx = np.zeros((pdim,Np))
    for k in range(K0):
        for l in range(K1):
            GM = GaussianMap(mu0[k,:].reshape(pdim,1),mu1[l,:].reshape(pdim,1),S0[k,:,:],S1[l,:,:],X.T) 
            Tmeanx += wstar[k,l]/pi0[k]*ProbaClassesX[:,k].T* GM

    # Reproject on original patches (in order to upsample)
    # (projection done slice by slice in order to fit in memory)
    # XXX : withdraw the slicing procedure?
    lenslice = Np
    nslice = int(np.ceil(Np/lenslice))
    Psynth = np.zeros((Np, pdim))
    ind = [None] * Np

    nbrs = NearestNeighbors(n_neighbors=1).fit(Y)
    
    for q in range(0,nslice):
        idxslice = slice(q*lenslice, min((q+1)*lenslice,Np))
        _, j = nbrs.kneighbors(Tmeanx.T[idxslice,:])
        ind[idxslice]=j[:,0]
        Psynth[idxslice,:] = Y[ind[idxslice],:]
        
    return Tmeanx.T, Psynth, ind


##################################################
# Functions for Patch Extraction/Aggregation
##################################################

def patches(im,w,s, npatches = None, selec = None):

    if (npatches is not None) and (selec is not None):
        print('wrong usage')

    m,n,nc= im.shape
    patches_im = im2patch(im,w,s)
    Np = patches_im.shape[0]
    
    if patches_im.ndim>2:
        patches_im = patches_im[:,:,0]

    if npatches is not None:
        ntarget = min(Np, npatches)
        print(f'Estimate target measure with {ntarget} patches')
        rperm = np.random.permutation(Np)
        selec = rperm[0:ntarget]

    if selec is not None:
        patches_im = patches_im[selec,:]

    return patches_im, selec


# patch extract with tensor reshaping included
# u must be a torch tensor of image size (m,n,c)
def im2patch(u,w,s=1):
    u_ = torch.tensor(u).permute(2,0,1).unsqueeze(0)
    return np.array(torch.nn.Unfold(kernel_size=w, dilation=1, padding=0, stride=s)(u_).squeeze(0).permute(1,0))

# patch agreggregation with tensor reshaping included
# (m, n) is the size of the desired output image
def patch2im(P, w, m, n, s=1):
    P_ = torch.tensor(P).permute(1,0).unsqueeze(0)
    sump = torch.nn.Fold((m,n), w, dilation=1, padding=0, stride=s)(P_)
    count = torch.nn.Fold((m,n), w, dilation=1, padding=0, stride=s)(P_*0+1)
    u_ = sump/count
    return np.array(u_.squeeze().permute(1,2,0))


######################################
#  Old version without pytorch
# 
# class patch:

#     def __init__(self,m,n,nc,w,s=1):
#         # Initialize instance of class patch.
#         #  P = patch(m,n,nc,w,s=1)
#         #
#         #  INPUT
#         #   m,n    Spatial dimensions
#         #   nc     Number of channels
#         #   w      Patch size (w x w)
#         #   s      Stride
        
#         self.m = m
#         self.n = n
#         self.nc = nc
#         self.w = w
#         self.stride = s
#         self.pdim = w*w*nc
#         [x,y] = np.mgrid[0:m-w+1:s,0:n-w+1:s]
#         Np = x.shape[0]*x.shape[1]  # number of patches
#         self.Np = Np
#         [dx,dy] = np.mgrid[0:w,0:w]
#         X = x[:,:,np.newaxis,np.newaxis]+dx[np.newaxis,np.newaxis,:]
#         Y = y[:,:,np.newaxis,np.newaxis]+dy[np.newaxis,np.newaxis,:]
#         px = X.reshape((Np,w*w))
#         py = Y.reshape((Np,w*w))
#         if nc>1:
#             px = np.tile(px[:,:,np.newaxis],(1,nc))
#             py = np.tile(py[:,:,np.newaxis],(1,nc))
#             px = px.reshape(Np,self.pdim)
#             py = py.reshape(Np,self.pdim)
#             pc = np.tile(np.arange(0,nc),(w*w,Np))
#             self.pc = pc.reshape((Np,w*w*nc))
#             #self.pc = pc.T
#         self.px = px
#         self.py = py

#     def im2patch(self,u):
#         # Extract patches from image u.
#         #  Pu = P.im2patch(u)
        
#         if self.nc>1:
#             pu = u[self.px,self.py,self.pc]
#         else:
#             pu = u[self.px,self.py]
#         return pu
        
#     def patch2im(self,p):
#         # Blend patches to form back an image.
#         # (The blending is a simple average)
#         #  v = P.patch2im(Pu)
        
#         m,n,nc = self.m,self.n,self.nc
#         if nc>1:
#             px,py,pc = self.px,self.py,self.pc
#             u = np.zeros((m,n,nc))
#             z = np.zeros((m,n,nc))
#             for j in range(0,self.pdim):
#                 u[px[:,j],py[:,j],pc[:,j]] += p[:,j]
#                 z[px[:,j],py[:,j],pc[:,j]] += 1
#         else:
#             px,py = self.px,self.py
#             u = np.zeros((m,n))
#             z = np.zeros((m,n))
#             for j in range(0,self.pdim):
#                 u[px[:,j],py[:,j]] += p[:,j]
#                 z[px[:,j],py[:,j]] += 1
#         u = u/z
#         return u


# def patches(im,w,s, npatches = None, selec = None):

#     if (npatches is not None) and (selec is not None):
#         print('wrong usage')

#     m,n,nc= im.shape
#     P = patch(m,n,nc,w,s)
#     patches_im = P.im2patch(im)
    
#     if patches_im.ndim>2:
#         patches_im = patches_im[:,:,0]

#     if npatches is not None:
#         ntarget = min(P.Np, npatches)
#         print(f'Estimate target measure with {ntarget} points')
#         rperm = np.random.permutation(P.Np)
#         selec = rperm[0:ntarget]

#     if selec is not None:
#         patches_im = patches_im[selec,:]

#     return patches_im, P, selec



##################################################
# Functions for Gaussian Texture Synthesis
##################################################

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


def gaussian_synthesis(u):
    (t,mv) = estimate_adsn_model(u)
    mv = np.reshape(mv,(1,1,3))*np.ones((u.shape[0],u.shape[1],3))
    v = adsn(t,mv)
    return v


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
