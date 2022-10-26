import matplotlib.pyplot as plt
import numpy as np

# 1. Periodic plus smooth image decomposition

def ppluss(U):

    n,m = U.shape
    U = U.astype(np.float64)

    #Compute the derivative across the boundary.

    B = np.zeros_like(U)
    B[:,0] += U[:,-1] - U[:,0]
    B[:,-1] += U[:,0] - U[:,-1]
    B[0,:] += U[-1,:] - U[0,:]
    B[-1,:] += U[0,:] - U[-1,:]

    #Compute the Laplacian filter over the Fourier domain.

    q = np.arange(n).reshape(n, 1).astype(B.dtype)
    r = np.arange(m).reshape(1, m).astype(B.dtype)
    L = (2*np.cos( np.divide((2*np.pi*q), n) ) + 2*np.cos( np.divide((2*np.pi*r), m) ) - 4)
    L[0, 0] = 1
    
    #Compute the smooth part S by solving Laplacian(S)=B. This is easily solved over the Fourier domain.

    S = np.fft.fft2(B)
    S = np.divide(S, L, out=np.zeros_like(S), where=L!=0)
    S[0, 0] = 0
    S = np.real(np.fft.ifft2(S))

    #Compute the periodic part P.

    P = U - S

    return P,S

# 2.1 

def ChangeToPeriodic(U):
    U_p, _ = ppluss(U)
    mean_u_P = np.mean(U_p)
    U_p -= mean_u_P
    return U_p, mean_u_P

# 2.2

def ComputeTexton(U):
    m,n = U.shape

    # Texton of u_gray_periodic
    Texton = np.fft.ifft2(np.abs(np.fft.fft2(U)))
    STI = np.fft.fftshift(Texton)

    return STI


def PlotSlices(real_STI):
    m,n = real_STI.shape

    # Plot slices
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot( np.arange(m)-m/2, real_STI[:,int(n/2)])
    axes[0].set_title('Signal x -> T(u0)(x,0)')
    axes[1].plot( np.arange(n)-n/2, real_STI[int(m/2),:])
    axes[1].set_title('Signal y -> T(u0)(0,y)')
    fig.tight_layout()
    plt.figure()

# 2.3

def GenerateGaussianTexture(STI, mean, W=None):
    m, n= STI.shape

    if W is None:
        W= np.fft.fft2(np.random.randn(m,n))/np.sqrt(m*n)
        W[0,0] = 1

    # Generate texture of same size
    u_hat = np.fft.fft2(STI)*W
    Gaussian_texture = np.real(np.fft.ifft2(u_hat)) + mean

    return Gaussian_texture


def GenerateGaussianTextureDouble(STI, mean):
    m, n= STI.shape

    # Generate texture of 2n x 2m
    Texton2= np.zeros((2*m,2*n)).astype(STI.dtype)
    Texton2[int(m/2):int(3*m/2),int(n/2):int(3*n/2)]= STI

    W2= np.fft.fft2(np.random.randn(2*m,2*n))/np.sqrt(m*n)
    W2[0,0] = 1

    u_hat_2 = np.fft.fft2(Texton2)*W2
    Gaussian_texture_2 = np.real(np.fft.ifft2(u_hat_2)) + mean

    return Gaussian_texture_2

# 3.1

def ChangeToPeriodic_Color(u):
    u_periodic = np.zeros_like(u)
    mean_u_periodic = np.zeros((3))

    for i in range(3):
        u_periodic[:,:,i], mean_u_periodic[i] = ChangeToPeriodic(u[:,:,i])

    return u_periodic, mean_u_periodic

# 3.2 

def ComputeTexton_Color(u_periodic):
    alpha=[0.33, 0.5, 0.17]
    alphaTexton = np.zeros_like(u_periodic)
    alphaTexton = alpha[0]*u_periodic[:,:,0] + alpha[1]*u_periodic[:,:,1] + alpha[2]*u_periodic[:,:,2]

    STI_color = np.zeros_like(u_periodic).astype(complex)
    oppPhase = np.conj(np.fft.fft2(alphaTexton)) / np.abs(np.fft.fft2(alphaTexton))

    for i in range(3):
        STI_color[:,:,i]= np.fft.fftshift( np.fft.ifft2(np.fft.fft2(u_periodic[:,:,i])*oppPhase))

    return STI_color

# 3.3

def GenerateGaussianTexture_Color(STI_color, mean, W = None):
    m,n,c = STI_color.shape

    # Generate white noise
    if W is None:
        W= np.fft.fft2(np.random.randn(m,n))/np.sqrt(m*n)
        W[0,0] = 1

    # Generate texture of same size
    Gaussian_texture_color = np.zeros_like(STI_color)

    for i in range(c):
        Gaussian_texture_color[:,:,i] = GenerateGaussianTexture(STI_color[:,:,i], mean[i], W)
    
    return Gaussian_texture_color

# 4.1

def Crop(u_orig, v_orig, gray):
    u, v = np.copy(u_orig), np.copy(v_orig)
    u_row, u_col = u.shape[0], u.shape[1]
    v_row, v_col = v.shape[0], v.shape[1]

    if gray == True:
        if u_row<v_row:
          v = v[0:u_row,:]
        else:
          u = u[0:v_row,:]
        if u_col<v_col:
          v = v[:,0:u_col]
        else:
          u = u[:,0:v_col]
    else:
        if u_row<v_row:
          v = v[0:u_row,:,:]
        else:
          u = u[0:v_row,:,:]
        if u_col<v_col:
          v = v[:,0:u_col,:]
        else:
          u = u[:,0:v_col,:]
    return u,v
    
    
def GaussianTextureMixing_gray(u_0_gray, u_1_gray, n_p):
    # Crop if sizes are different
    u_0_gray, u_1_gray = Crop(u_0_gray, u_1_gray, gray=True)
    nh, nw = u_1_gray.shape

    # Replace images for its periodic Component
    u_0_gray, mean_u_0_gray = ChangeToPeriodic(u_0_gray)
    u_1_gray, mean_u_1_gray = ChangeToPeriodic(u_1_gray)

    # Generate textons
    Texton0 = ComputeTexton(u_0_gray)
    Texton1 = ComputeTexton(u_1_gray)

    GTrho_list = list()

    # Foreach rho, generate mix texture
    for r in np.linspace(0,1, n_p +1):
        Trho = (1-r)*Texton0 + r*Texton1
        mean_Trho = (1-r)*mean_u_0_gray + r*mean_u_1_gray
        
        # Generate Gaussian texture from mix Texton
        Gaussian_texture_mixture = GenerateGaussianTexture(Trho, mean_Trho)
        GTrho_list.append(Gaussian_texture_mixture)
    
    return GTrho_list

# 4.2

def GaussianTextureMixing_color(u_0, u_1, n_p):
    # Crop if sizes are different
    u_0, u_1 = Crop(u_0, u_1, gray=False)
    nh, nw, c = u_1.shape

    # Replace images for its periodic Component
    u_0, mean_u_0 = ChangeToPeriodic_Color(u_0)
    u_1, mean_u_1 = ChangeToPeriodic_Color(u_1)

    Grho_list = list()

    F0 = np.zeros((nh,nw,c)).astype(complex)
    F1 = np.zeros((nh,nw,c)).astype(complex)
    CR = 0

    for i in range(c):
        F0[:,:,i] = np.fft.fft2(u_0[:,:,i])
        F1[:,:,i] = np.fft.fft2(u_1[:,:,i])
        CR += np.conj(F0[:,:,i]) * F1[:,:,i]

    CR /= np.abs(CR)

    # white noise
    W= np.fft.fft2(np.random.randn(nh,nw))/np.sqrt(nh*nw)
    W[0,0] = 1
    
    # Foreach rho, generate mix texture
    for r in np.linspace(0,1,n_p +1):
        Grho = np.zeros((nh,nw,c))
        mean_Trho = (1-r)*mean_u_1 + r*mean_u_0
        for i in range(c):
            u_hat_mix_color = ((1-r)*F1[:,:,i] + r*F0[:,:,i]*CR) * W
            Grho[:,:,i] = np.real(np.fft.ifft2(u_hat_mix_color)) + mean_Trho[i]
        Grho_list.append(Grho)
    
    return Grho_list