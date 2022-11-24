import numpy as np
import torch

### Helper function ####

# Denoiser function to be used by all the PnP implementations.
# Inspired by UCLA optimization group's code    
# https://github.com/uclaopt/Provable_Plug_and_Play
def denoise(xtilde, denoiser, m, n):

    # Scale xtilde to be in range of [0,1]
    mintmp = np.min(xtilde)
    maxtmp = np.max(xtilde)
    xtilde = (xtilde - mintmp) / (maxtmp - mintmp)

    # Denoise
    xtilde_torch = np.reshape(xtilde, (1,1,m,n))
    xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.cuda.FloatTensor)
    r = denoiser(xtilde_torch).cpu().numpy()
    r = np.reshape(r, -1)
    x = xtilde - r

    # Rescale the denoised x back to original scale
    x = x * (maxtmp - mintmp) + mintmp
    return x

### PnP functions ####

def pnp_admm(noisy, denoiser, proximal_step, **opts):
    """
    Parameters:
        :noisy - the noisy observation.
        :denoiser - the Gaussian denoiser used in Plug-and-Play ADMM.
        :proximal_step - the function which implements the proximal step of the ADMM algorithm.
        :opts - the kwargs for hyperparameters in Plug-and-Play ADMM.
    """

    # Process parameters
    maxitr = opts.get('maxitr', 100)

    # Initialization
    m, n = noisy.shape
    noisy_flat = np.reshape(noisy, -1)
    x = np.copy(noisy_flat)
    y = np.copy(noisy_flat)
    u = np.zeros_like(noisy_flat, dtype=np.float64)

    for i in range(maxitr):

        # Denoising step
        xtilde = np.copy(y - u)
        x = proximal_step(xtilde, noisy_flat, **opts)
        #x = denoise(xtilde, denoiser, m, n)

        # Proximal step
        y = denoise(x+u, denoiser, m, n)
        #y = proximal_step(x+u, noisy_flat, **opts)

        # Dual update      
        u = np.copy(u + x - y)

    # Get restored image
    x = np.reshape((x) , (m, n))
    return x


def pnp_fbs(noisy, denoiser, gradient_step, **opts):
    """
    Parameters:
        :noisy - the noisy observation.
        :denoiser - the Gaussian denoiser used in Plug-and-Play FBS.
        :gradient_step - the function which implements the gradient step: x- alpha*grad(f)
        :opts - the kwargs for hyperparameters in Plug-and-Play FBS.
    """

    # Process parameters
    maxitr = opts.get('maxitr', 100)

    # Initialization
    m, n = noisy.shape
    noisy_flat = np.reshape(noisy, -1)
    x = np.copy(noisy_flat)

    for i in range(maxitr):

        # FBS step
        xtilde = np.copy(gradient_step(x, noisy_flat, **opts))
        x = denoise(xtilde, denoiser, m, n)

    # Get restored image
    x = np.reshape((x) , (m, n))
    return x

def pnp_bbs(noisy, denoiser, proximal_step, **opts):
    """
    Parameters:
        :noisy - the noisy observation.
        :denoiser - the Gaussian denoiser used in Plug-and-Play BBS.
        :proximal_step - the function which implements the proximal step of the ADMM algorithm.
        :opts - the kwargs for hyperparameters in Plug-and-Play BBS.
    """

    # Process parameters
    maxitr = opts.get('maxitr', 100)

    # Initialization
    m, n = noisy.shape
    noisy_flat = np.reshape(noisy, -1)
    x = np.copy(noisy_flat)

    for i in range(maxitr):

        # Proximal step
        xtilde = np.copy(x)
        x = proximal_step(xtilde, noisy_flat, **opts)

        # Denoising step     
        x = denoise(x, denoiser, m, n)

    # Get restored image
    x = np.reshape((x) , (m, n))
    return x