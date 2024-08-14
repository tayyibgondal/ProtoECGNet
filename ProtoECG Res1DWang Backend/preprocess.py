import torch
import numpy as np


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def ecg_to_image_tensor(ecg_waveform):
    # Convert the ECG waveform (1D) to an image
    batch_size, channels, length = ecg_waveform.shape
    image_tensors = []

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(ecg_waveform[i].transpose(0, 1).cpu().numpy())
        ax.axis('off')

        # Convert the plot to a numpy array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)

        # Convert numpy array to tensor and resize to (3, 224, 224)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image)
        image_tensors.append(image_tensor)
    
    # Stack the batch of image tensors
    return torch.stack(image_tensors).to(ecg_waveform.device)

def preprocess_input_function(x):
    '''
    Convert ECG waveform to image, resize it, and apply the normalization used in the pretrained model.
    '''
    image_tensor = ecg_to_image_tensor(x)
    return preprocess(image_tensor, mean=mean, std=std)
    
def preprocess(x, mean, std):
    assert x.size(1) == 3  # Ensure it's a 3-channel image
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
