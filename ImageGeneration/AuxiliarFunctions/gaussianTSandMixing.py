import matplotlib.pyplot as plt
import numpy as np

def loadImage(input_image_name):
    u = plt.imread(input_image_name)

    if u.shape[2]==4:
      #removing alpha chanel
      u = u[:,:,:3]

    if input_image_name[-3:]=="jpg"  or input_image_name[-4:]=="jpeg" or input_image_name[-3:]=="bmp":
        u = u/255

    return u

def printGrayTexton(texton, mean):
    texton_norm = np.minimum(texton, np.ones_like(texton))
    texton_norm += mean
    plt.imshow(np.clip(texton_norm,0,1), cmap='gray')
    plt.axis('off')


def printColorTexton(texton, mean):
    texton_color_norm = np.minimum(texton, np.ones_like(texton))
    texton_color_norm += mean
    plt.imshow(texton_color_norm)
    plt.axis('off')


def PlotTextures(Gaussian_texture, Gaussian_texture_2):
    nh, nw = Gaussian_texture.shape

    plt_0 = plt.figure(figsize=(5, 5))
    plt.imshow(np.clip(Gaussian_texture,0,1), cmap='gray')
    plt.title('Gaussian texture of size n x m')
    plt.axis('off')
    plt.show()

    plt_1 = plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(Gaussian_texture_2,0,1), cmap='gray')
    plt.title('Gaussian texture of size 2*n x 2*m') 
    plt.axis('off')
    plt.show()

def compare_images(imgs, n_p, gray):
    labels = ['image ' + str(i) for i in range(len(imgs))]
    tb = widgets.TabBar(labels, location='top')
    rhos = np.linspace(0,1,n_p +1)
    for i, img in enumerate(imgs):
        with tb.output_to(i, select=(i == 0)):
            if gray == True:
                plt.title('Rho = ' + str(rhos[i]))
                plt.axis('off')
                plt.imshow(np.clip(img,0,1), cmap='gray',vmin=0,vmax=1)
            else:
                plt.title('Rho = ' + str(rhos[i]))
                plt.axis('off')
                plt.imshow(np.clip(img,0,1))


def printImages(u, P, S, size1, size2):
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(size1, size1))

    # Plot orignal gray image, periodic and smooth component
    axes[0].imshow(u,cmap='gray',vmin=0,vmax=1)
    axes[0].set_title('Original gray image')
    axes[0].axis('off')
    axes[1].imshow(P,cmap='gray',vmin=0,vmax=1)
    axes[1].set_title('Periodic component')
    axes[1].axis('off')
    axes[2].imshow(S,cmap='gray')
    axes[2].set_title('Smooth component')
    axes[2].axis('off')

    fig.tight_layout()
    plt.figure()


def printFTImages(u, P, S, size1, size2):
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(size1, size1))

    # Comppute FT for orignal gray image, periodic and smooth component
    FT_u = np.log(np.abs(np.fft.fftshift(np.fft.fft2(u))) + 1)
    FT_periodic = np.log(np.abs(np.fft.fftshift(np.fft.fft2(P))) + 1)
    FT_S = np.log(np.abs(np.fft.fftshift(np.fft.fft2(S))) + 1)

    # Plot FT of orignal gray image, periodic and smooth component
    axes[0].set_title("Fourier transform image")
    axes[0].imshow(FT_u,cmap='gray')
    axes[1].set_title("Fourier transform periodic component")
    axes[1].imshow(FT_periodic,cmap='gray')
    axes[2].set_title("Fourier transform smooth component")
    axes[2].imshow(FT_S,cmap='gray')
    
    fig.tight_layout()
    plt.figure()