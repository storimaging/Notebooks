import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import mse_loss
import torchvision.models as models
from torchvision.transforms.functional import resize, to_tensor, to_pil_image

from PIL import Image
import matplotlib.pyplot as plt

#########
# Functions to manage images
#########


MEAN = (0.485, 0.456, 0.406)
iter_ = 0

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


########
#Loss
########

# Computes Gram matrix for the input batch tensor.
#    Args: tnsr (torch.Tensor): input tensor of the Size([B, C, H, W]).
#    Returns:  G (torch.Tensor): output tensor of the Size([B, C, C]).
def gramm(tnsr: torch.Tensor) -> torch.Tensor: 

    b,c,h,w = tnsr.size() 
    F = tnsr.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1,2)) 
    G.div_(h*w)
    return G

# Computes MSE Loss for 2 Gram matrices 
def gram_loss(input: torch.Tensor, gramm_target: torch.Tensor, weight: float = 1.0):
  
    loss = weight * mse_loss(gramm(input), gramm_target)
    return loss


#######
# Exercise A
#######

# Computes Mean and Covariance matrix of each channel
def moyCov(input: torch.Tensor):
    
    input_redim = input.squeeze(0)
    c,h,w = input_redim.size() 

    mY = torch.mean(input_redim, axis=(1,2))
    CovTensor = gramm(input - mY.view(1,-1,1,1))
      
    return mY,CovTensor

# Computes loss we have to add to gramm Loss
def A_loss(input: torch.Tensor, m_target: torch.Tensor, C_target: torch.Tensor, lambda1, lambda2):
    
    m_input, C_input = moyCov(input)
    loss = lambda1 * mse_loss(m_input, m_target) +  lambda2 * mse_loss(C_input, C_target)
    return loss

def textureSynthesisA (n_iters, log_every, synth, cnn, target, gramm_targets, outputs, layers, layers_weights, optimizer):

    ##### Section Added #########

    # Compute mean and covariance for target (Just once)
    m_target, C_target = moyCov(target)
    print(m_target.shape) # should be 3
    print(C_target.shape) # should be 3x3

    # Define lambdas
    lambda1 = 1
    lambda2 = 1

    ##### End Section Added ######

    while iter_ <= n_iters:

        def closure():
            global iter_

            optimizer.zero_grad()

            # Forward pass using synth. Get activations of selected layers for image synth (outputs).
            cnn(synth)
            synth_outputs = [outputs[key] for key in layers] 
            
            # Compute loss for each activation
            losses = []
            for activations in zip(synth_outputs, gramm_targets, layers_weights):
                losses.append(gram_loss(*activations).unsqueeze(0))

            ##### Section Added #########
            losses.append(A_loss(synth, m_target, C_target, lambda1, lambda2).unsqueeze(0))
            ##### End Section Added ######
            
            total_loss = torch.cat(losses).sum()
            total_loss.backward()

            # Display results: print Loss value and show images
            if iter_ % log_every == 0:
                printResults(target, synth, iter_, total_loss)

            iter_ += 1

            return total_loss

        optimizer.step(closure)
    
    return synth

#######
# Exercise B
#######

# Computes loss we have to add to gramm Loss
def B_loss(input: torch.Tensor, f_transform_target_modulus: torch.Tensor, lambda3):
    
    # Computes fourier transform for last two dimensions of tensor
    f_transform_input_modulus = torch.abs(torch.fft.fft2(input))

    loss = lambda3 * mse_loss(f_transform_input_modulus, f_transform_target_modulus) 

    return loss

def textureSynthesisB (n_iters, log_every, synth, cnn, target, gramm_targets, outputs, layers, layers_weights, optimizer):

    # Compute Fourier transform for target (Just once)
    f_transform_target_module = torch.abs(torch.fft.fft2(target))

    # Define lambda
    lambda3 = 0.1

    while iter_ <= n_iters:

        def closure():
            global iter_

            optimizer.zero_grad()

            # Forward pass using synth. Get activations of selected layers for image synth (outputs).
            cnn(synth)
            synth_outputs = [outputs[key] for key in layers] 
            
            # Compute loss for each activation
            losses = []
            for activations in zip(synth_outputs, gramm_targets, layers_weights):
                losses.append(gram_loss(*activations).unsqueeze(0))

            ##### Section Added #########
            losses.append(B_loss(synth, f_transform_target_module, lambda3).unsqueeze(0))
            ##### End Section Added ######
            
            total_loss = torch.cat(losses).sum()
            total_loss.backward()

            # Display results: print Loss value and show images
            if iter_ % log_every == 0:
                printResults(target, synth, iter_, total_loss)

            iter_ += 1

            return total_loss

        optimizer.step(closure)
    
    return synth


#######
# Exercise C
#######

# mean spatial dimension tensor
def mean_Spatial (input: torch.Tensor):

    input_redim = input.squeeze(0)
    c,h,w = input_redim.size() 
    input_redim_view = input_redim.view(c, h*w)
    mean_input = torch.mean(input_redim_view, axis=1) 
    return mean_input

# Computes MSE Loss for 2 tensors, with weight
def C_loss(input: torch.Tensor, mean_spatial_target: torch.Tensor, weight: float = 1.0):

    loss = weight * mse_loss(mean_Spatial(input), mean_spatial_target)
    return loss

def textureSynthesisC (n_iters, log_every, synth, cnn, target, gramm_targets, outputs, layers, layers_weights, optimizer):

    ##### Section Added #########

    # Define weights for layers
    layers_weights = [1e3/n**2 for n in [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 
                                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]]

    # Computes mean of target (Just once)
    mean_target_outputs = [mean_Spatial(target) for target in target_outputs]

    ##### End Section Added ######

    while iter_ <= n_iters:

        def closure():
            global iter_

            optimizer.zero_grad()

            # Forward pass using synth. Get activations of selected layers for image synth (outputs).
            cnn(synth)
            synth_outputs = [outputs[key] for key in layers] 
            
            ##### Section Changed #########
            # Compute loss for each activation
            losses = []
            for activations in zip(synth_outputs, mean_target_outputs, layers_weights):
                losses.append(C_loss(*activations).unsqueeze(0))
            ##### End Section Changed ######
            
            total_loss = torch.cat(losses).sum()
            total_loss.backward()

            # Display results: print Loss value and show images
            if iter_ % log_every == 0:
                printResults(target, synth, iter_, total_loss)

            iter_ += 1

            return total_loss

        optimizer.step(closure)
    
    return synth