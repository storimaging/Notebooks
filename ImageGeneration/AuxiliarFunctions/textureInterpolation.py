import torch
from torchvision.transforms.functional import resize, to_tensor, to_pil_image, normalize
from PIL import Image
import matplotlib.pyplot as plt

# Utilities
# Functions to manage images

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def prep_img(image: str, size=None, mean=MEAN, std=STD):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    4) normalize
    """
    im = Image.open(image)
    texture = resize(im, size)
    texture_tensor = to_tensor(texture).unsqueeze(0)
    if texture_tensor.shape[1]==4:
        print('removing alpha chanel')
        texture_tensor = texture_tensor[:,:3,:,:]
    texture_tensor = normalize(texture_tensor, mean=mean, std=std)
    return texture_tensor

# Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD, inplace: bool = False):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone().squeeze() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
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


def printResults(target1, target2, opt_img, iter, loss, lambda1):
    """ Displays the intermediate results of the main iteration
    """ 
    print('Iteration: %d, loss: %f'%(iter, loss.item()))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    axes[0].imshow(to_pil(target1))
    axes[0].set_title('Texture 1. Lambda: ' + str(lambda1))
    axes[1].imshow(to_pil(target2))
    axes[1].set_title('Texture 2. Lambda: ' + str(1-lambda1))
    axes[2].imshow(to_pil(opt_img.squeeze(0)))
    axes[2].set_title('Interpolation')
    fig.tight_layout()
    plt.pause(0.05)