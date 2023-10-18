import numpy as np
import scipy.signal

# basic functions div, grad, laplacian, no periodicity

def div(cx,cy):
    """
    cy and cy are coordonates of a vector field.
    #the function computes the discrete divergence of this vector field
    """
    nr,nc=cx.shape

    ddx=np.zeros((nr,nc))
    ddy=np.zeros((nr,nc))

    ddx[:,1:-1]=cx[:,1:-1]-cx[:,0:-2]
    ddx[:,0]=cx[:,0]
    ddx[:,-1]=-cx[:,-2]
  
    ddy[1:-1,:]=cy[1:-1,:]-cy[0:-2,:]
    ddy[0,:]=cy[0,:]
    ddy[-1,:]=-cy[-2,:]
 
    d=ddx+ddy

    return d


def grad(im):
    """
    computes the gradient of the image 'im'
    """
    nr,nc=im.shape
  
    gx = im[:,1:]-im[:,0:-1]
    gx = np.block([gx,np.zeros((nr,1))])

    gy =im[1:,:]-im[0:-1,:]
    gy=np.block([[gy],[np.zeros((1,nc))]])
    return gx,gy


def laplacian(im):
    """
    computes the laplacian of the image 'im'
    """

    cx, cy = grad(im)
    return div(cx, cy)


# basic functions, periodicity

def div_per(cx,cy):
    """
    cy and cy are coordonates of a vector field.
    #the function computes the discrete divergence of this vector field
    """
    nr,nc=cx.shape

    ddx=np.zeros((nr,nc))
    ddy=np.zeros((nr,nc))

    ddx[:,1:]=cx[:,1:]-cx[:,0:-1]
    ddx[:,0]=cx[:,0] - cx[:,-1]
  
    ddy[1:,:]=cy[1:,:]-cy[0:-1,:]
    ddy[0,:]=cy[0,:] - cy[-1,:]
 
    d=ddx+ddy

    return d


def grad_per(im):
    """
    computes the gradient of the image 'im'
    """
    nr,nc=im.shape
    gx = np.zeros((nr,nc))
    gy = np.zeros((nr,nc))
    
    gx[:,:-1] = im[:,1:]-im[:,0:-1]
    gx[:,-1] = im[:,0]-im[:,-1]

    gy[:-1,:] =im[1:,:]-im[0:-1,:]
    gy[-1,:] = im[0,:]-im[-1,:]

    return gx,gy






# restoration

def inpainting(v, epsilon, tau, niter, mask):
    """
    Parameters:
    v: degraded image
    epislon: small value to avoid division by 0
    tau: gradient step 
    niter: number of iterations 
    mask: mask of missing pixels.
    """
    
    u = np.copy(v)

    for i in range(niter):
        ux,uy  = grad(u)
        normgrad = np.sqrt(ux**2 + uy**2 + epsilon**2)
        u = u - tau * ( - div(ux/normgrad,uy/normgrad))
        u[mask] = v[mask]
    
    return u


def convol_aperiodic(a,b):
    return scipy.signal.convolve2d(a,b, boundary='symm', mode='same')


def tvdeconv(ub,k,lambd,niter):
    """
    Deconvolution with double splitting and known kernel
    """

    # Kernel
    k = k/np.sum(k)
    normk = np.sum(np.abs(k))
    kstar = k[::-1,::-1]

    # Initialization
    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ub)

    p = np.zeros((nr,nc,2))
    q = np.zeros((nr,nc))
    tau   = 0.9/np.sqrt(8*lambd**2 + normk**2)
    sigma = tau
    theta = 1
    
    # Double splitting
    for i in range(niter):
        #INSERT YOUR CODE HERE  

        # ProxF for p
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # ProxF for q
        q = q + sigma*convol_aperiodic(ubar, k)
        q = (q-sigma*ub)/(1+sigma)

        # Subgradient step on u
        d=div(p[:,:,0],p[:,:,1])
        unew = (ut + tau*lambd*d - tau*convol_aperiodic(q, kstar)) 
    
        # Extragradient step on u 
        ubar = unew + theta*(unew-ut)
        ut = np.copy(unew) 

        #END INSERT YOUR CODE HERE  

    return ut


def chambolle_pock_prox_TV(TV,ub,lambd,niter, **opts):
    """
    the function solves the problem (for a non periodic image u)
    - TVL2
       argmin_u   1/2 \| u - ub\|^2 + \lambda TV(u)
    - or TVL2A
       argmin_u   1/2 \| Au - ub\|^2 + \lambda TV(u)
       with A = diagonal matrix represented by the mask send as an opt on parameters
    - or TVL1
       argmin_u   1/2 \| u - ub\|_1 + \lambda TV(u)
    with TV(u) = \sum_i \|\nabla u (i) \|_2
    uses niter iterations of Chambolle-Pock
    """

    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ut)
    p = np.zeros((nr,nc,2))
    tau = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    
    # For TVL2A case
    mask = opts.get('mask', np.ones_like(ub))
    
    for k in range(niter):
        # Calcul de proxF
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # Calcul de proxG
        d=div(p[:,:,0],p[:,:,1])
        if (TV == "TVL2"):
            unew = 1/(1+tau)*(ut+tau*lambd*d+tau*ub) 
        elif (TV == "TVL2A"):    
            unew = 1/(1+tau*mask)*(ut+tau*lambd*d+tau*mask*ub)
        else:
            uaux = ut+tau*lambd*d
            unew = (uaux-tau)*(uaux-ub>tau)+(uaux+tau)*(uaux-ub<-tau)+ub*(abs(uaux-ub)<=tau)
        
        # Extragradient step
        ubar = unew+theta*(unew-ut)
        ut = np.copy(unew)
           
    return ut


# Convolution assuming a periodic signal
def convol_periodic(a,b):
    return np.real(np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))) 

def IdplustauATA_inv(x,tau,h): 
    return np.real(np.fft.ifft2(np.fft.fft2(x)/(1+tau*np.abs(np.fft.fft2(h))**2)))

def chambolle_pock_deblurring_TVL2(ub,h,lambd,niter):
    # the function solves the problem
    # argmin_u   1/2 \| Au - ub\|^2 + \lambda TV(u)
    # with TV(u) = \sum_i \|\nabla u (i) \|_2
    # and A = blur given by a kernel h
    # uses niter iterations of Chambolle-Pock

    nr,nc = ub.shape
    ut = np.copy(ub)

    p = np.zeros((nr,nc,2))
    tau   = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    ubar = np.copy(ut)

    # Conjugate of the kernel h
    h_fft = np.fft.fft2(h)
    hc_fft = np.conj(h_fft)
    hc = np.fft.ifft2(hc_fft)

    for k in range(niter):
        
        # Subgradient step on p 
        ux,uy  = grad_per(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep
        
        # Subgradient step on u
        d=div_per(p[:,:,0],p[:,:,1])
        unew = (ut+tau*lambd*d+tau*convol_periodic(ub, hc)) 
        unew = IdplustauATA_inv(unew, tau,h)
    
        # Extragradient step on u 
        ubar = unew+theta*(unew-ut)
        ut = np.copy(unew)
        
    return ut