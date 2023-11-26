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

def get_area(labels):
    AREAS = []
    for i in np.unique(labels)[1:]:
        m = np.zeros_like(labels)
        m[labels == i] = 1
        area = np.sum(m)
        AREAS.append(area)
    return np.asarray(sorted(AREAS))[::-1].astype(float)

def diameter_from_image(image, mask, SM=1):
    image_norm = normalize(image)
    image_sub = subtract_mask(image_norm, mask)
    cmask = get_circle_mask(image_sub, sm=SM)
    labels = measure.label(cmask)
    AREAS = get_area(labels)
    DIAMETER = 2*np.sqrt(AREAS / np.pi)
    return DIAMETER, labels

def plot_side_by_side(image, image_sub):
    ims = [image, image_sub]
    plt.figure(figsize=(12,5))
    for i in range(len(ims)):
        plt.subplot(1, 2, 1+i)
        plt.imshow(ims[i])
        plt.colorbar(shrink=0.5)
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()

def create_circle_colormap(num_labels):
    colors = plt.cm.get_cmap('tab20', num_labels)
    circle_colormap = ListedColormap(colors(np.arange(num_labels)))
    return circle_colormap

def process_and_display_image(image_path, mask_path):
    mask, image = load_pair(image_path, mask_path)
    DIAMETER, labels = diameter_from_image(image, mask, SM=1)

    # Create a colormap for the circle labels
    num_labels = len(np.unique(labels))
    circle_colormap = create_circle_colormap(num_labels)

    # Display the labeled image with the custom colormap
    plt.imshow(labels, cmap=circle_colormap)
    plt.colorbar()
    plt.show()






























