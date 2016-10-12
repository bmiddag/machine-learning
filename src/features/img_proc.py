import array
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorsys
import numpy as np
from PIL import Image
from scipy import misc
from scipy import ndimage
from skimage import color
import scipy.misc

def img_hist(image, plot=False):
    """
    Create six histograms containing hue, saturation and value values,
    and hue saturation and lightness
    for a given image and a number of bins.

    If the plot parameter is true, we will plot these histograms.

    Returns:
    Three histograms (as lists)

    Example:
    hist_h, hist_s, hist_v = img_hist("x.png", 16)
    """
    array=np.asarray(image)
    arr_hsv=(array.astype(float))/255.0
    arr_hsl=(array.astype(float))/255.0
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            arr_hsv[i, j] = colorsys.rgb_to_hsv(arr_hsv[i, j, 0], arr_hsv[i, j, 1], arr_hsv[i, j, 2])
            arr_hsl[i, j] = colorsys.rgb_to_hls(arr_hsl[i, j, 0], arr_hsl[i, j, 1], arr_hsl[i, j, 2])

    img_hsv = arr_hsv
    img_hsl = arr_hsl

    hsv_h=img_hsv[...,0].flatten()
    hsl_h=img_hsl[...,0].flatten()

    if (plot):
        plt.subplot(2,3,1)
        plt.hist(hsv_h*360,bins=360,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
        plt.title("Hue")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2,3,4)
        plt.hist(hsl_h*360,bins=360,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
        plt.title("Hue")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

    hsv_s=img_hsv[...,1].flatten()
    hsl_s=img_hsl[...,2].flatten()

    if (plot):
        plt.subplot(2,3,2)
        plt.hist(hsv_s,bins=100,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
        plt.title("Saturation")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2,3,5)
        plt.hist(hsl_s,bins=100,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
        plt.title("Saturation")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

    hsv_v=img_hsv[...,2].flatten()
    hsl_l=img_hsl[...,1].flatten()

    if (plot):
        plt.subplot(2,3,3)
        plt.hist(hsv_v,bins=100,range=(0.0,1.0),histtype='stepfilled', color='b', label='Intesity')
        plt.title("Value")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2,3,6)
        plt.hist(hsl_l,bins=100,range=(0.0,1.0),histtype='stepfilled', color='b', label='Intesity')
        plt.title("Lightness")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

        plt.show()

    #returns double array with the y-values in [0] and bins in [1]
    hist_hsv_h = np.histogram(hsv_h*360, bins=360,range=(0.0,360.0))
    hist_hsl_h = np.histogram(hsl_h*360, bins=360,range=(0.0,360.0))

    hist_hsv_s = np.histogram(hsv_s, bins=100,range=(0.0,1.0))
    hist_hsl_s = np.histogram(hsl_s, bins=100,range=(0.0,1.0))

    hist_hsv_v = np.histogram(hsv_v, bins=100,range=(0.0,1.0))
    hist_hsl_l = np.histogram(hsl_l, bins=100,range=(0.0,1.0))


    return hist_hsv_h[0], hist_hsv_s[0], hist_hsv_v[0], hist_hsl_h[0], hist_hsl_s[0], hist_hsl_l[0]

def percentages(image):
    """
    Estimate the percentage of white, red, blue and black in an image.

    Returns: Four percentages (white, red, blue, black)

    Example:
    w, r, b, bl = percentages("x.png")
    """
    hist_hsv_h, hist_hsv_s, hist_hsv_v, hist_hsl_h, hist_hsl_s, hist_hsl_l = img_hist(image)

    total_v = sum(hist_hsv_v)
    total_h = sum(hist_hsv_h)
    total_l = sum(hist_hsl_l)
    # Estimate the percentage of white in this image
    perc_white = sum(hist_hsl_l[80:100]) / total_l
    # Estimate the percentage of red in this image
    perc_red = (sum(hist_hsv_h[0:60]) + sum(hist_hsv_h[320:360])) / total_h
    # Estimage the percentage of blue in this image
    perc_blue = sum(hist_hsv_h[175:250]) / total_h
    # Estimage the percentage of black in this image
    perc_black = sum(hist_hsl_l[0.0:20]) / total_l
    
    #normalize
    norm = perc_white + perc_black + perc_blue + perc_red

    return perc_white/norm, perc_red/norm, perc_blue/norm, perc_black/norm
