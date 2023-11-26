# In your misc.py file
import cv2
from skimage import measure
import numpy as np
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap

def load_image(path):
    return cv2.imread(path)[:,:,::-1]

def load_mask(path):
    return cv2.imread(path)[:,:,0].astype(bool)

def load_pair(image_path, mask_path):
    image = load_image(image_path)
    mask = load_mask(mask_path)
    return mask, cv2.resize(image, mask.shape[::-1], interpolation=cv2.INTER_CUBIC)

def normalize(image):
    image = image[:,:,1]
    image = image.astype(float) / 255
    image = image - np.median(image)
    image[image > 0] = 0
    return np.abs(image)

def subtract_mask(raw, mask):
    image = raw.copy()
    image[mask == False] = 0
    return image

def get_circle_mask(image, sm=0.2):
    th = np.median(image) + sm * np.std(image)
    cmask = np.zeros_like(image, dtype=bool)
    cmask[image > th] = True
    return cmask

def plot_side_by_side(image, image_sub):
    ims = [image, image_sub]
    plt.figure(figsize=(12,5))
    for i in range(len(ims)):
        plt.subplot(1, 2, 1+i)
        plt.imshow(ims[i])
        plt.colorbar(shrink=0.5)
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()

def plot_line_profile(image_norm, yc=None, x_start=None, x_end=None, sm=None):
    line_profiles = []
    for channel in range(image_norm.shape[2]):
        line_profile = np.median(image_norm[yc-1:yc+2, x_start:x_end, channel], axis=0)
        line_profiles.append(line_profile)
    x_vals = np.arange(x_start, x_end)
    return x_vals, line_profiles
  
# Define a Gaussian function to fit
def gaussian(x, A, mu, sigma, baseline):
    return -A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + baseline





























