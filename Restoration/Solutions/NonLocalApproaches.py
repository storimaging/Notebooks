import numpy as np
from skimage.util import view_as_windows

def moyCov(Y):
    
    mY = np.mean(Y, axis=1)
    Yc = Y-mY.reshape(-1,1)
    C = np.cov(Y)
    return mY,C,Yc


def denoisePCA(Y,eta):
    
    # PCA
    mY,C,Yc = moyCov(Y)
    D, X = np.linalg.eig(C)
    idx = D.argsort()[::-1] 
    D = D[idx]
    X = X[:,idx]
    
    # Hard thresholding     
    Yproj = np.dot(X.T,Yc)                       
    Yproj[np.abs(Yproj)<eta] = 0  
    
    Z = np.dot(X, Yproj) + mY[:, np.newaxis]

    return Z


def denoisePCAsoft(Y,eta):
    
    # PCA
    mY,C,Yc = moyCov(Y)
    D, X = np.linalg.eig(C)
    idx = D.argsort()[::-1] 
    D = D[idx]
    X = X[:,idx]
    
    # Soft thresholding 
    Yproj= np.dot(X.T,Yc)  
    Yproj = np.sign(Yproj)*np.maximum(0, np.abs(Yproj) - eta)
    
    Z = np.dot(X, Yproj) + mY[:, np.newaxis]
    
    return Z


def Patch_Extraction(f, v):
    
    patch_shape = (2*f+1, 2*f+1)
    return view_as_windows(v, patch_shape).reshape(-1, patch_shape[0]*patch_shape[1]).T[:,::]


def Image_Reconstruction2(Ydenoised, nrow, ncol, f):
    
    tmp = np.zeros(shape=(nrow, ncol,(2*f+1)*(2*f+1)))
    count = np.zeros(shape=(nrow, ncol))
    
    # For iter in rows of Ydenoised
    i = 0    
    for x in range(2*f+1):
        for y in range(2*f+1):   
            w = Ydenoised[i,:].reshape(nrow-2*f,ncol-2*f)    # Extract image of row i
            tmp[x:nrow-2*f+x, y:ncol-2*f+y,i] = w    # Put on channel i of tmp the image with int the correct position   
            count [x:nrow-2*f+x, y:ncol-2*f+y] += 1    # Indicate on count the pixels contained in the image w. it will be necessary for calculate average
            i += 1

    return np.divide(np.sum(tmp,axis=2), count)


def PCADenoising(f, v, sigma, denoise):

    nrow, ncol = v.shape

    # Decomposition of v into Patches Y
    Y = Patch_Extraction(f, v)
    
    # PCA of Y and hard thresholding
    eta = 3*sigma 
    Z = denoise(Y,eta)

    # Reconstruction of vdenoised
    vdenoised = Image_Reconstruction2(Z, nrow, ncol, f)
    return vdenoised


def CutImage(lengthSquare, v,step):
    
    square_shape = (lengthSquare, lengthSquare)
    return view_as_windows(v, square_shape, step)


def PCADenoisingLocalization(f, v, sigma, denoise, step, length):

    nrow, ncol = v.shape
    eta = 3*sigma

    imageRectangles = CutImage(length, v, step)
    tmp = np.zeros(shape=(nrow, ncol, imageRectangles.shape[0]*imageRectangles.shape[1]))
    count = np.zeros(shape=(nrow, ncol))
    aux = 0

    for i in range(imageRectangles.shape[0]):
        for j in range(imageRectangles.shape[1]):
            # Decomposition of v into Patches Y
            Y = Patch_Extraction(f, imageRectangles[i,j,:,:])

            # PCA of Y and hard thresholding    
            Z = denoise(Y,eta)

            # Reconstruction of vdenoised
            vdenoised = Image_Reconstruction2(Z, imageRectangles.shape[2], imageRectangles.shape[3], f)

            x_pos, y_pos = i*step, j*step
        
            # Put on channel aux of tmp the image with int the correct position  
            tmp  [x_pos:length +x_pos, y_pos:length+y_pos, aux] = vdenoised 
        
            # Indicate on count the pixels contained in the image w. it will be necessary for calculate average
            count[x_pos:length +x_pos, y_pos:length+y_pos] += 1  
        
            aux += 1

    vdenoised = np.divide(np.sum(tmp,axis=2), count)
    return vdenoised
