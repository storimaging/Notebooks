import numpy as np
import os
os.system("wget -nc https://raw.githubusercontent.com/storimaging/Notebooks/main/Restoration/AuxiliarFunctions/AuxiliarFunctions_PnP.py")
from AuxiliarFunctions_PnP import *

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
        :denoiser - the Gaussian denoiser used in Plug-and-Play ADMM.
        :gradient_step - the function which implements the gradient step: x- alpha*grad(f)
        :opts - the kwargs for hyperparameters in Plug-and-Play ADMM.
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

    for i in range(maxitr):

        # Proximal step
        xtilde = np.copy(x)
        x = proximal_step(xtilde, noisy_flat, **opts)

        # Denoising step     
        x = denoise(x, denoiser, m, n)

    # Get restored image
    x = np.reshape((x) , (m, n))
    return x