# Simple grayscale values

from scipy import misc
from skimage import color, exposure
from skimage import transform as tf

def get_raw_grayscale(image, dimensions=[28, 28], equalize=True):
    """
    Convert an image to grayscale and resize.
    Returns a 2D matrix of pixel values.
    """
    image = color.colorconv.rgb2grey(image)
    if equalize:
        image = exposure.equalize_hist(image)
    image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    return [pixel for column in image for pixel in column]

def get_raw_rgb(image, dimensions=[28, 28]):
    """
    Resize RGB image.
    Returns a 2D matrix of pixel values.
    """
    #image = exposure.equalize_hist(image)
    image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    return [channel for column in image for pixel in column for channel in pixel]

def get_raw_rgb_image(image, dimensions=[28, 28]):
    """
    Resize RGB image.
    Returns a 2D matrix of pixel values.
    """
    #image = exposure.equalize_hist(image)
    image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    return image
    #return [channel for column in image for pixel in column for channel in pixel]

def get_raw_hsv(image, dimensions=[28, 28]):
    """
    Convert an image to HSV and resize.
    Returns a 2D matrix of pixel values.
    """
    image = color.colorconv.rgb2hsv(image)
    #image = exposure.equalize_hist(image)
    image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    return [channel for column in image for pixel in column for channel in pixel]

def get_raw_hue(image, dimensions=[28, 28]):
    """
    Convert an image to HSV, extract HUE and resize.
    Returns a 1D vector of pixel values.
    """
    image = color.colorconv.rgb2hsv(image)
    image = tf.resize(image, (dimensions[0], dimensions[1]), mode='nearest')
    return image[:,:,0]
