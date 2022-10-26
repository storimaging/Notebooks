
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torchvision.models as models
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#### Functions needed defined in Notebook

# Functions to manage images

MEAN = (0.485, 0.456, 0.406)

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def prep_img(image: str, size=None, mean=MEAN):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    4) substract mean and multipy by 255
    """
    im = Image.open(image)
    texture = resize(im, size)
    tensor = to_tensor(texture).unsqueeze(0)
    if tensor.shape[1]==4:
        print('removing alpha chanel')
        tensor = tensor[:,:3,:,:]
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.sub_(mean).mul_(255)
    return tensor

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def denormalize(tensor: torch.Tensor, mean=MEAN):

    tensor = tensor.clone().squeeze() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(1./255).add_(mean)
    return tensor

# Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def to_pil(tensor: torch.Tensor):
    """Converts tensor to PIL Image.
    Args: tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns: PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = to_pil_image(img)
    return img


def printResults(target, opt_img, iter, loss):
    """ Displays the intermediate results of the main iteration
    """
    print('Iteration: %d, loss: %f'%(iter, loss.item()))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
    axes[0].imshow(to_pil(target.squeeze(0)))
    axes[0].set_title('Original texture')
    axes[1].imshow(to_pil(opt_img.squeeze(0)))
    axes[1].set_title('Synthesis')
    fig.tight_layout()
    plt.pause(0.05)

# for computing the spatial average of the feature responses of each selected layer
def mean_Spatial (input: torch.Tensor):  

    mean_input = torch.mean(input.squeeze(), axis=(1,2))  
    return mean_input

def feature_statistics(outputs: torch.Tensor, layers):

    meansOutputs = [mean_Spatial(outputs[key]) for key in layers] 
    return torch.cat(meansOutputs)


#######
#Excercise A
#######


def textureSynthesisA (n_iters, log_every, delta, gamma, epsilon, cnn, target, outputs, layers, x, beta):

    # Compute just once means spatial of activations of the target image. 
    cnn(target)
    featureTargetStatistics = feature_statistics(outputs, layers)

    # initialize weights (Theta)
    theta = torch.zeros_like(featureTargetStatistics)

    # Initialize list of intermediary images
    xpil_list = []

    # Forward pass using x. Get activations of selected layers for image x (outputs).
    x.requires_grad=True
    cnn(x)

    for iter in range(n_iters):
    
            if x.grad is not None:
                x.grad.zero_()
        
            # Compute V and its gradient with respect to x:

            # TODO
            #################
            # solution
            J = (epsilon /2) * x.pow(2).sum()
            V = torch.dot(theta, beta*(feature_statistics(outputs, layers) - featureTargetStatistics)) + J
            # gradient with respect to x in x.grad
            V.backward()
            #################

            # update image
            with torch.no_grad():
                
                # TODO
                #################
                # solution
                z_rand = torch.randn_like(x)
                x += - gamma * x.grad + np.sqrt(2 * gamma) * z_rand
                #################

            # Forward pass using x. Get activations of selected layers for image x (outputs).
            cnn(x)

            # update weights thetas:
            with torch.no_grad():
                # TODO
                #################
                # solution
                theta = theta + delta * beta *(feature_statistics(outputs, layers) - featureTargetStatistics)
                #################

            # Display results: print Loss value and show image
            if (iter==0 or iter % log_every == log_every-1):
                print('Iteration: ', iter)
                display(to_pil(torch.cat((target, x), axis=3)))
                # Store for comparison:
                xpil_list.append(to_pil(x.clone().detach()))

    return xpil_list

#######
#Excercise B
#######

# returns the mean spatial vector and the values of upper triangular part of covariance matrix
def MeanAndVectorizedUpperTriangularCovMatrix(input):

    input_den = denormalize(input)
    # Compute mean for input
    m_input = mean_Spatial(input_den.unsqueeze(0))

    # Compute covariance matrix for input
    c,h,w = input_den.size() 
    C_input = torch.cov(input_den.squeeze().view(c,h*w))

    # get indices of upper triangular matrix
    triu_indices = torch.triu_indices(3,3)

    # Get list of values un upper triangular matrix
    vectorized_upper_triangular_matrix = C_input[triu_indices[0], triu_indices[1]]

    return m_input, vectorized_upper_triangular_matrix

# We modify the $\theta$ list to add the two new theta tensors: one for spatial mean and one for covariance. 
# We modify $V$ to add the new statistics. We are also adding a section to update the new 𝜃.
def new_feature_statistics(x, outputs: torch.Tensor, layers):
    # Calculate f (old statistics)
    f = feature_statistics(outputs, layers)
    # Calculate statistics of mean color and covariance for x
    m_x, vectorized_cov_x = MeanAndVectorizedUpperTriangularCovMatrix(x)
    return torch.cat((f, m_x, vectorized_cov_x))


def textureSynthesisB (n_iters, log_every, delta, gamma, epsilon, cnn, target, outputs, layers, x, beta):

    ##### MODIFIED for using new_feature_statistics #####
    # Compute just once means spatial of activations of the target image. 
    cnn(target)
    featureTargetStatistics = new_feature_statistics(target, outputs, layers)

    # initialize weights (Theta)
    theta = torch.zeros_like(featureTargetStatistics)

    xpil_list = []

    # Forward pass using x. Get activations of selected layers for image x (outputs).
    x.requires_grad=True
    cnn(x)

    for iter in range(n_iters):
        
        if x.grad is not None:
            x.grad.zeros_()
    
        # Compute V
        J = (epsilon /2) * x.pow(2).sum()
        ##### MODIFIED for using new_feature_statistics #####
        V = torch.dot(theta, beta*(new_feature_statistics(x, outputs, layers) - featureTargetStatistics)) + J

        # gradient with respect to x in x.grad
        V.backward()

        # update image
        with torch.no_grad():
            z_rand = torch.randn_like(x) 
            x = x - gamma * x.grad + np.sqrt(2 * gamma) * z_rand
            x.requires_grad=True

        # Forward pass using x. Get activations of selected layers for image x (outputs).
        cnn(x)

        # Update weights thetas
        with torch.no_grad():   
            ##### MODIFIED for using new_feature_statistics #####
            theta = theta + delta * beta *(new_feature_statistics(x,outputs, layers) - featureTargetStatistics)
            
        # Display results: print Loss value and show image
        if (iter==0 or iter % log_every == log_every-1):
            xpil_list.append(to_pil(x.clone().detach()))
            print('Iteration: ', iter)
            display(to_pil(torch.cat((target, x), axis=3)))
            plt.pause(0.05)

    return xpil_list