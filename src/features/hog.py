# Histogram of oriented gradients

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage as ndi
from skimage import measure, color, feature
from skimage import data
from skimage import transform as tf
from skimage.feature import hog
from skimage import data, exposure
#from skimage.filters import threshold_otsu

def demo_hog(image, resize=True, dimensions=[96, 96]):
    """
    A small demo for HOG that shows an image with the HOG next to it.
    """
    image = color.colorconv.rgb2grey(image)
    if resize:
        image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    threshold_img = threshold_otsu(image)
    binary = image > threshold_img
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)
    print(fd);
    print(len(fd));
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

def get_hog(image, resize=True, dimensions=[48, 48], equalize=True, otsu=False):
    """
    Calculate the histogram of gradients for an image.
    The image should be an RGB image.
    We return a feature vector (see http://scikit-image.org/docs/dev/auto_examples/plot_hog.html).
    If otsu is True, we threshold the image first to a binary image before calculating HOG.
    """
    image = color.colorconv.rgb2grey(image)
    result = image
    if resize:
        result = tf.resize(result, (dimensions[0], dimensions[1]), mode='nearest')
    if equalize:
        result = exposure.equalize_hist(result)
    if otsu:
        thresh = threshold_otsu(result)
        result = result > thresh
    fd, hog_image = hog(result, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)
    return fd
