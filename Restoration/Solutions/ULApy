import math as m
import numpy as np

def ULA_gauss(niter,delta,x0=0):
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X 
        X = X - delta*grad + np.sqrt(2*delta)*Z
        Y[i] = X
    return Y


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



def grad_htv(I, eps):
    """
    This function allows to compute the gradient of the HTV energy of an image.
    Inputs:
        - I: the image
        - eps (int): regularization parameter
    """
    grad_x, grad_y = grad(I)
    norme = np.sqrt(grad_x**2 + grad_y**2)
    temp1 = -1/(eps)*div(grad_x,grad_y)
    temp2 = (-div(grad_x/(norme+1e-15), grad_y/(norme+1e-15)))
    return  temp1*(norme < eps) + temp2*(norme>=eps)


def ula_deblurring(im_blurred, im_orig, sigma, delta, lambd, epsilon, h, inter= 10, n_iter = 10000, n_burn_in = 1000):
    """
    Inputs:
        - im_blurred: noisy image
        - im_orig: original image (to compute output MMSE_error)
        - sigma: standard deviation of noise of noisy image. 
        - delta: parameter of ULA iteration
        - lambd: TV regularization parameter lambd
        - epsilon: Regularization for HTV (since ULA needs differentiable potentials) 
        - h: kernel of convolution
        - inter: Interval to save samples
        - n_iter: Total number of iterations 
        - n_burn_in: Number of iterations for the burn-in phase. 
    Outputs:
        - X_mean: The average posterior
        - Xf: Last sample
        - std: The posterior std
        - X_samples: sequence of samples through iterations
        - MMSE_error: Average error between X_mean and u
    """
    
    # Initializations
    n_Rows, n_Col = im_blurred.shape
    samples = np.zeros((n_Rows, n_Col,int(n_iter/inter)))
    X_mean = np.copy(im_blurred)
    X_2 = np.zeros((n_Rows, n_Col))
    MMSE_error = []
    X = np.zeros((n_Rows, n_Col)) # Markov chain initialization
    
    # Conjugate of the kernel h
    h_fft = np.fft.fft2(h)
    hc_fft = np.conj(h_fft)
    hc = np.fft.ifft2(hc_fft)
    
    for i in tqdm(range(n_iter)):
        # Gradient of the potential
        Z = np.random.randn(n_Rows, n_Col)
        ATA_x = np.real(np.fft.ifft2((hc_fft*h_fft)*np.fft.fft2(X)))
        AT_y = np.real(np.fft.ifft2(hc_fft*np.fft.fft2(im_blurred)))
        grad = (ATA_x- AT_y)/sigma**2 + lambd*grad_htv(X,epsilon)      
        
        # ULA step
        X = X - delta*grad + np.sqrt(2*delta)*Z        
        
        # Keep samples 
        if (i%inter==0):
            samples[:,:,int(i/inter)] = np.copy(X) 

        # Compute mean of X, and mean of X**2 
        if i>=n_burn_in:                                
            i_b = i - n_burn_in
            X_mean = i_b/(i_b+1)*X_mean + 1/(i_b+1)*X
            X_2 = i_b/(i_b+1)*X_2 + 1/(i_b+1)*(X**2)
            MMSE_error = MMSE_error + [1/(n_Rows*n_Col)*np.sqrt(np.sum((X_mean - im_orig)**2))]
        
    # Variance computation
    var = X_2 - X_mean**2
    std = np.sqrt(var*(var>=0))
            
    return X_mean, X, std, samples, MMSE_error, 
           
def PSNR(im_0, im_rest):
    """
    This function allows to compute the PSNR.
    Inputs:
        -im_0: the original image
        -im_rest: the restored image
    Output:
        -the PSNR
    """
    N = im_0.shape[0]
    M = im_0.shape[1]
    EQM = np.linalg.norm(im_0 - im_rest, 'fro')**2/N/M
    
    return -10*m.log10(EQM)


def convol_periodic(a,b):
    """
    Convolution assuming a periodic signal
    """
    return np.real(np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))) 

def IdplustauATA_inv(x,tau,h): 
    return np.real(np.fft.ifft2(np.fft.fft2(x)/(1+tau*np.abs(np.fft.fft2(h))**2)))

def chambolle_pock_deblurring_TVL2(ub,h,lambd,niter):
    """
    the function solves the problem
    argmin_u   1/2 \| Au - ub\|^2 + \lambda TV(u)
    with TV(u) = \sum_i \|\nabla u (i) \|_2
    and A = blur given by a kernel h
    uses niter iterations of Chambolle-Pock
    """

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
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep
        
        # Subgradient step on u
        d=div(p[:,:,0],p[:,:,1])
        unew = (ut+tau*lambd*d+tau*convol_periodic(ub, hc)) 
        unew = IdplustauATA_inv(unew, tau,h)
    
        # Extragradient step on u 
        ubar = unew+theta*(unew-ut)
        ut = np.copy(unew)
        
    return ut