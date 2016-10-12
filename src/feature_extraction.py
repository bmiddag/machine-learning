# Feature extraction

import sys, string
import numpy as np

from metadata import *

from features.contours import find_contours, find_canny_contours
from features.img_proc import percentages
from features.hog import get_hog
from features.raw import get_raw_grayscale, get_raw_rgb, get_raw_hsv, get_raw_hue, get_raw_rgb_image

from sklearn import preprocessing
from skimage.transform import resize
from scipy import misc
import pickle

from settings import *

class Features:
    """
    Class that extracts and keeps features settings for later reuse.
    """
    def __init__(self, features, feature_scaling='Standard'):
        """
        Use a dict for features if you want to pass any parameters to the feature functions,
        but it might be more readable to modify the default values of the feature functions and use a list instead
        """
        if features is not None:
            if isinstance(features, dict):
                self.features = features # Parameters were passed
            elif isinstance(features, list):
                self.features = dict.fromkeys(features,{}) # No parameters were passed
        else:
            self.features = {}
        self.normalizer = None
        self.feature_scaling = feature_scaling

    def extract_features_train(self, allow_shuffling=True):
        if features_IO and os.path.exists(train_features_file) and allow_shuffling:
            if verbose:
                print("Training: Loading features from file")
            features, normalizer, samples, mdata, feature_scaling = self.load_features(train_features_file)
            if features == self.features and feature_scaling == self.feature_scaling:
                self.normalizer = normalizer
                return samples, mdata
            if verbose:
                print("Training: Loaded features do not match selected features. Proceeding with feature extraction.")
        elif verbose:
            print("Training: Feature extraction")
        mdata = Metadata()
        mdata.read_traindata()
        if allow_shuffling:
            mdata.randomize_traindata()
        samples, normalizer = extract_features(mdata.file_paths, self.features, feature_scaling=self.feature_scaling, normalizer=None)
        self.normalizer = normalizer
        if features_IO and allow_shuffling:
            if verbose:
                print("Training: Saving extracted features")
            self.save_features(train_features_file, samples, mdata)
        return samples, mdata

    def extract_features_test(self, allow_IO=True):
        if features_IO and os.path.exists(test_features_file) and allow_IO:
            if verbose:
                print("Testing: Loading features from file")
            features, _, samples, mdata, feature_scaling = self.load_features(test_features_file)
            if features == self.features and feature_scaling == self.feature_scaling:
                return samples, mdata
            if verbose:
                print("Testing: Loaded features do not match selected features. Proceeding with feature extraction.")
        elif verbose:
            print("Testing: Feature extraction")
        mdata = Metadata()
        mdata.read_testdata()
        samples, normalizer = extract_features(mdata.file_paths, self.features, feature_scaling=self.feature_scaling, normalizer=self.normalizer)
        if features_IO and allow_IO:
            if verbose:
                print("Testing: Saving extracted features")
            self.save_features(test_features_file, samples, mdata)
        return samples, mdata

    def load_features(self, path):
        with open(path, "rb") as f:
            features, normalizer, samples, mdata, feature_scaling = pickle.load(f)
        return features, normalizer, samples, mdata, feature_scaling

    def save_features(self, path, samples, mdata):
        if not os.path.exists(features_directory):
            os.makedirs(features_directory)
        with open(path, "wb") as f:
            pickle.dump([self.features, self.normalizer, samples, mdata, self.feature_scaling], f)

def extract_features(file_paths, feature_options, feature_scaling='Standard', normalizer=None):
    """
    Extract features from several images.

    file_paths should contain the paths to the images.

    feature_options should be a dictionary containing any of the following as keys:
    * 'color_histograms' -> we will use color percentages
    * 'hog' -> we will use the Histogram of Oriented Gradients feature
    * 'raw_rgb' -> we will use the raw image, using rgb values
    * 'raw_hsv' -> we will use the raw image, using hsv values
    * 'raw_grayscale' -> we will use the raw image, in grayscale
    The parameters should be specified as nested dicts and are the same (Don't add any more!!) as the OPTIONAL parameters in the feature functions.
    For example, the signature for raw_grayscale is: get_raw_grayscale(image, dimensions=[28, 28], equalize=True), so an example option dict is {'dimensions': [16 16]}.

    If the grayscale parameter is true, we will first set the image to grayscale values.

    If the feature_scaling parameter is true, we will scale features and normalize them so they look
    more or less like standard normally distributed data: Gaussian with zero mean and unit variance.
    """
    samples = []
    i = 1
    prev_digits = 0
    for file_path in file_paths:
        if verbose:
            procstr = "Processing image " + str(i) + " of " + str(len(file_paths))
            print("%s%s" % ("\b"*prev_digits, procstr), end="")
            i = i + 1;
            prev_digits = len(procstr)
            sys.stdout.flush()
        features = []
        image = misc.imread(file_path) # speeds up the process when multiple features are combined
        if 'color_histograms' in feature_options:
            pw, pr, pb, pbl = percentages(image, **feature_options['color_histograms'])
            features.extend([pw, pr, pb, pbl])
        if 'hog' in feature_options:
            fd = get_hog(image, **feature_options['hog'])
            features.extend(fd)
        if 'raw_grayscale' in feature_options:
            gs = get_raw_grayscale(image, **feature_options['raw_grayscale'])
            features.extend(gs)
        if 'raw_rgb_image' in feature_options:
            img = get_raw_rgb_image(image, **feature_options['raw_rgb_image'])
            features.extend(img)
        if 'raw_rgb' in feature_options:
            rgb = get_raw_rgb(image, **feature_options['raw_rgb'])
            features.extend(rgb)
        if 'raw_hsv' in feature_options:
            hsv = get_raw_hsv(image, **feature_options['raw_hsv'])
            features.extend(hsv)
        if 'raw_hue' in feature_options:
            hue = get_raw_hue(image, **feature_options['raw_hue'])
            features.extend(hue)
        samples.append(features)
    if verbose:
        print("")
    samples = np.array(samples)
    if feature_scaling != 'None':
        if verbose:
            print("Feature normalizing/scaling")
        if normalizer is not None:
            samples = normalizer.transform(samples)
        else:
            if feature_scaling == 'Standard':
                normalizer = preprocessing.StandardScaler()
            elif feature_scaling == 'MinMax':
                normalizer = preprocessing.MinMaxScaler()
            elif feature_scaling == 'MaxAbs':
                normalizer = preprocessing.MaxAbsScaler()
            elif feature_scaling == 'Robust':
                normalizer = preprocessing.RobustScaler()
            else:
                normalizer = preprocessing.StandardScaler()
            samples = normalizer.fit_transform(samples)
    else:
        normalizer = None
    return samples, normalizer
