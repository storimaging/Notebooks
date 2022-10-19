import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.signal as scs
from sklearn.cluster import KMeans

def todo_HSI_after_compression(imrgb):
   
    imred   = imrgb[:,:,0]
    imgreen = imrgb[:,:,1]
    imblue  = imrgb[:,:,2]
    O1 = (imred-imgreen)/np.sqrt(2)
    O2 = (imred+imgreen-2*imblue)/np.sqrt(6)
    O3 = (imred+imgreen+imblue)/np.sqrt(3)
    H=np.arctan(O1/(O2+0.001))
    S=np.sqrt(O1**2+O2**2)
    I=O3

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

    axes[0].imshow(H, cmap='gray')
    axes[0].set_title('H')
    axes[1].imshow(S, cmap='gray')
    axes[1].set_title('S')
    axes[2].imshow(I, cmap='gray')
    axes[2].set_title('I')
    fig.tight_layout()

def todo_color_quantization_with_kmeans(imrgb,k):
    [nrow,ncol,nch]=imrgb.shape
    X = imrgb.reshape((nrow*ncol,3))
    x_pred = KMeans(n_clusters=k).fit_predict(X)
    mu     = np.zeros((k,3))
    pi     = np.zeros((k))
    for i in range(k): 
        Xi       = X[x_pred==i]
        mu[i,:]  = np.mean(Xi,0)
        pi[i]    = Xi.shape[0]/X.shape[0]
    
    fig = plt.figure(figsize=(15, 7))
    axis = fig.add_subplot(1, 2, 1)
    axis.imshow(mu[x_pred].reshape((nrow,ncol,3)),cmap='gray')
    axis = fig.add_subplot(1, 2, 2, projection="3d")
    axis.scatter(mu[:, 0], mu[:,1],mu[:, 2],c=mu,s=5e4*pi)
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


def todo_gray_world(imrgb):

    e = np.zeros((3,))
    e[0] = np.mean(imrgb[:,:,0]*255)/128
    e[1] = np.mean(imrgb[:,:,1]*255)/128
    e[2] = np.mean(imrgb[:,:,2]*255)/128
    #e = e/np.mean(e)   # ensure that the image mean gray level won't change after multiplication by e 

    grey_world = np.zeros(imrgb.shape).astype(float)
    grey_world[:,:,0] = imrgb[:,:,0]/e[0]
    grey_world[:,:,1] = imrgb[:,:,1]/e[1]
    grey_world[:,:,2] = imrgb[:,:,2]/e[2]
    print(e)
    
    f, axe = plt.subplots(1,2,figsize=(15,15))
    axe[1].imshow(grey_world)
    axe[1].set_title('Grey-World correction')
    axe[0].imshow(imrgb)
    axe[0].set_title('Original')


def todo_shades_of_gray(imrgb,p):

    e = np.zeros((3,))
    e[0] = np.mean((imrgb[:,:,0]*255)**p)**(1/p)
    e[1] = np.mean((imrgb[:,:,1]*255)**p)**(1/p)
    e[2] = np.mean((imrgb[:,:,2]*255)**p)**(1/p)
    
    new = np.zeros(imrgb.shape)
    new[:,:,0] = imrgb[:,:,0]
    new[:,:,1] = imrgb[:,:,1]/(e[1]/e[0])
    new[:,:,2] = imrgb[:,:,2]/(e[2]/e[0])
    print(e)
    
    f, axe = plt.subplots(1,2,figsize=(15,15))
    axe[1].imshow(new)
    axe[1].set_title('Shades-of-gray correction')
    axe[0].imshow(imrgb)
    axe[0].set_title('Original')

def todo_interpolate_3_channels(raw,imrgb):

    R   = raw[:, :, 0]
    G   = raw[:, :, 1] 
    B   = raw[:, :, 2]
    lap = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/4
    lap2 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])/4
    G   = G + scs.convolve2d(G, lap, mode='same') # Interpolation of the green channel at the missing points
    B   = B + scs.convolve2d(B, lap2, mode='same')
    B   = B + scs.convolve2d(B, lap, mode='same') # Interpolation of the blue channel at the missing points
    R   = R + scs.convolve2d(R, lap2, mode='same')
    R   = R + scs.convolve2d(R, lap, mode='same') # Interpolation of the red channel at the missing points
    
    output = np.zeros(imrgb.shape, imrgb.dtype)
    output[:, :, 0] = R
    output[:, :, 1] = G
    output[:, :, 2] = B
    return output
