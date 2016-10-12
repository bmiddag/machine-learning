import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from scipy import ndimage as ndi
from skimage import measure, color, feature, exposure
from skimage import transform as tf

def find_contours(image_path):
    """
    Demo for a contour finding method based on skimage.measure.
    """
    image = misc.imread(image_path)
    image = color.colorconv.rgb2grey(image)
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')

    contours = measure.find_contours(image, 0.8)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth = 2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def find_canny_contours(image, demo=False):
    """
    Demo for a canny contour finding method based on skimage.feature.canny,
    if demo is set to True.
    This function returns a vector with edges.
    """
    image = color.colorconv.rgb2grey(image)
    image = tf.resize(image, (40, 40), mode='nearest')
    image = exposure.equalize_hist(image)

    edges = feature.canny(image, sigma=3)

    if(demo):
        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

        ax1.imshow(image, cmap=plt.cm.jet)
        ax1.axis('off')
        ax1.set_title('original image', fontsize=20)

        ax2.imshow(edges, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title('Canny filter, $\sigma=3$', fontsize=20)

        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                                bottom=0.02, left=0.02, right=0.98)
    return edges
