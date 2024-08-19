import torch
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y 

def normalize_img(image):
    """
    Normalize the input image to be within the range [0, 1].

    Parameters:
    image (numpy.ndarray): Input image array.

    Returns:
    numpy.ndarray: Normalized image array.
    """
    # Convert image to a numpy array if it isn't already
    image = np.array(image, dtype=np.float32)
    
    # Normalize the image to be within the range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    
    if min_val != max_val:  # Avoid division by zero
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)  # If all values are the same, return a zero array
    
    return image

 
def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model.
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)
