

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # for kmeans


#############
# computing and displaying histograms
#############

def comp_histos(im):
    
    imhisto, bins = np.histogram(im, range=(0,1), bins = 256)
    imhisto       = imhisto/np.sum(imhisto)
    imhistocum = np.cumsum(imhisto)
    
    return imhisto, bins, imhistocum

def plot_histos(im):
    
    imhisto, bins, imhistocum = comp_histos(im)
    values = (bins[1:]+bins[:-1])/2
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].imshow(im,cmap='gray',vmin=0,vmax=1)
    axes[0].axis('off')
    axes[1].bar(values,imhisto,width=1/256)
    axes[1].set_title('histogram')
    axes[2].bar(values,imhistocum,width=1/256)
    axes[2].set_title('cumulative histogram')
    fig.tight_layout()
    plt.show()

#############
# histogram equalization
#############

def todo_equalization(imrgb):

    if (len(imrgb.shape)==3):
        imgray = 1/3*(imrgb[:,:,0]+imrgb[:,:,1]+imrgb[:,:,2])
    else:
        imgray = imrgb
    [nrow,ncol] = imgray.shape
        
    imhisto,bins= np.histogram(imgray, range=(0,1), bins = 256)
    imhisto      = imhisto/np.sum(imhisto)
    imhistocum = np.cumsum(imhisto)

    imeq = imhistocum[np.uint8(imgray*255)]
    imeq=imeq.reshape(nrow,ncol)
    
    return imeq

#############
# noise on histograms
#############

def add_gaussian_noise(u,sigma):
    
    nrow, ncol = u.shape
    res = u + sigma * np.random.randn(nrow,ncol)
    #res = np.maximum(0,np.minimum(1,res))
    
    return res


def todo_noise_histograms(imgray):
    
    for sigma in [0,0.2,0.6]:
        imgray_noise = add_gaussian_noise(imgray,sigma)
        plot_histos(imgray_noise)

def add_uniform_noise(u,alpha):
    
    nrow, ncol = u.shape
    res = u + (np.random.rand(nrow,ncol)-.5)*2*alpha
    #res = np.maximum(0,np.minimum(1,res))
    
    return res

def todo_uniform_noise_histograms(imgray):
    
    for sigma in [0,0.1,0.5]:
        imgray_noise = add_uniform_noise(imgray,sigma)
        plot_histos(imgray_noise)


def add_impulse_noise(u,p):
    
    nrow, ncol = u.shape
    Y = np.random.rand(nrow,ncol)
    tab = np.random.rand(nrow,ncol)
    X = np.zeros((nrow,ncol))
    X[tab<p] = 1
    res = (1-X)*u+X*Y
    return res

def todo_impulse_noise_histograms(imgray):
    
    for p in [0,.3,.7]:
        imgray_noise = add_impulse_noise(imgray,p)
        plot_histos(imgray_noise)



##################
#### quantization and dithering
##################

# we define a function to perform the quantization of an image u given the values of t_i and q_i :
def quantize(u,t,q):
    K = len(q)
    u_quant = np.zeros(u.shape)
    for k in range(K):
        test = (u>=t[k])
        u_quant[test] = q[k]
    return u_quant

def todo_histogram_quantize(u,K):
    nrow, ncol = u.shape
    # we need to compute t_i = H_u^-1(i/K)
    # First we sort the values of u:
    S_u = np.sort(u,axis=None)
    # S_u gives the sorted values of u. We have S_u[j]=lambda <=> j/|Omega| = H_u(lambda)
    # So we have : t_i = H_u^-1(i/K) = S_u[|Omega|*i/K]
    # So first we need to compute the indices j = |Omega|*i/K for i=0,1,...,K :
    tab_i = np.arange(K+1)/K
    tab_j = ((nrow*ncol-1)*tab_i).astype(int)
    # now we compute the t_i = S_u[j]
    t = S_u[tab_j]
    # finally we compute the q_i = (t_{i+1}+t_i)/2 :
    q = .5 * (t[1:]+t[:-1])
    # finally we compute the quantization given t and q :
    u_quant = quantize(u,t,q)
    return u_quant


def todo_Lloyd_Max_quantize(u,K,plot=False):
    imhisto, bins, imhistocum = comp_histos(u)
    imcumesp = np.cumsum(np.arange(256)*imhisto/256)
    t = np.arange(K+1)/K
    test = True
    eps = 1e-6
    k = 0
    while test:
        k+=1
        indt = (255*t).astype(np.int32)
        q = np.diff(imcumesp[indt]) / np.diff(imhistocum[indt])
        t_old = t.copy()
        t[1:-1] = .5 * (q[1:]+q[:-1])
        test = (np.linalg.norm(t-t_old) > eps)
    if K==2:
        # N.B. we prefer to use values 0 and 1 for a binary image
        q[0], q[1] = 0, 1
    u_quant = quantize(u,t,q)
    if plot:
        plt.figure(figsize=(16, 7))
        plt.subplot(1,2,1)
        plt.title('Lloyd-Max quantization with ' + str(K) + ' levels')
        plt.axis('off')
        plt.imshow(u_quant,cmap='gray',vmin=0,vmax=1)
        plt.subplot(1,2,2)
        plt.title('histogram of u, points t_i (in blue) and q_i (in red)')
        plt.plot(np.arange(256)/256,imhisto)
        plt.plot(t,np.zeros(t.shape),'.b')
        plt.plot(q,np.zeros(q.shape),'+r')
        plt.show()
    return u_quant


def dither_quantize(u,K,sigma):
    u = u + sigma * np.random.randn(u.shape[0],u.shape[1])
    u_quant = todo_Lloyd_Max_quantize(u,K)
    return u_quant

def todo_dither(u,K,sigma):
    im_quant = todo_Lloyd_Max_quantize(u,K)
    plt.figure(figsize=(14, 7))
    plt.subplot(1,2,1)
    plt.title('Lloyd-Max quantization with '+str(K)+' levels')
    plt.axis('off')
    plt.imshow(im_quant,cmap='gray',vmin=0,vmax=1)
    im_dither_quant = dither_quantize(u,K,sigma)
    plt.subplot(1,2,2)
    plt.title('same, with dithering (Gaussian noise of std {:f})'.format(sigma))
    plt.axis('off')
    plt.imshow(im_dither_quant,cmap='gray',vmin=0,vmax=1)
    plt.show()
