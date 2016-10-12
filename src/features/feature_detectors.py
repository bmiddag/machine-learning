import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage as ndi
from skimage import measure, color, feature, exposure
from skimage import data
from skimage import transform as tf
from skimage.feature import match_descriptors, corner_harris, corner_peaks, ORB, plot_matches, CENSURE

def get_orb(image):
    image = color.colorconv.rgb2grey(image)
    image = tf.resize(image, (128, 128), mode='nearest')
    image = exposure.equalize_hist(image)
    descriptor_extractor = ORB(n_keypoints=128)
    descriptor_extractor.detect_and_extract(image)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.scatter(keypoints[:, 1], keypoints[:, 0], facecolors='none', edgecolors='r')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return descriptors