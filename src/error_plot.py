#!/usr/bin/env python3

from sklearn import preprocessing
import numpy as np
import sys, string, os, csv
from model import get_model
from settings import *
import warnings
from ensemble import EnsembleClassifier
from collections import defaultdict
from numpy import cumsum
from scipy import misc
from features.hog import get_hog
from sknn import mlp
import matplotlib.pyplot as plt

def main():
    """
    Plot the validation and training errors
    for our MLP classifier with HOG features for increasing
    number of epochs.
    """
    # Read training data
    superclasses = next(os.walk(train_data_folder))[1]
    file_paths = []
    y = []
    pole_ids = []
    for superclass_name in superclasses:
        classes = next(os.walk(train_data_folder + '/' + superclass_name))[1]
        for class_name in classes:
            files = next(os.walk(train_data_folder + '/' + superclass_name + '/' + class_name + '/'))[2]
            for file_name in files:
                if file_name == '.DS_Store': # OS X........
                    continue
                # We don't really need to know whether a sign of a different class but with the same pole ID is from the same pole,
                # so we can just append the class name to make the labels unique when the class is different.
                pole_id = class_name + "_" + file_name.split('_')[0]
                image_id = int(file_name.split('_')[1].replace('.png',''))
                image_name = train_data_folder + '/' + superclass_name + "/" + class_name + "/" + file_name
                file_paths.append(image_name)
                y.append(class_name)
                pole_ids.append(pole_id)

    # Extract validation set
    # We do this by selecting a certain percentage of data from each
    # class set.
    # We use some zipping and sorting magic to easily select
    # percentages of data below
    zipped = zip(y, file_paths, pole_ids)
    zipped = sorted(zipped)
    rd = defaultdict(list)
    for i,_ in enumerate(zipped):
        rd[y[i]].append(file_paths[i])
    # Select percentages (we use more training data than validation data)
    fraction_train = 0.7
    fraction_validation = 0.3
    percentages = [fraction_train, fraction_validation]

    training_set = defaultdict(list) # {'class_name': ['path','path']}
    validation_set = defaultdict(list)
    for classname in rd.keys():
        results = list(percentage_split(rd[classname], percentages))
        training_set[classname].extend(results[0])
        validation_set[classname].extend(results[1])

    # Extract features from training set
    file_paths = []
    y_train = []
    y_val = []
    for key in training_set.keys():
        t = training_set[key]
        for el in t:
            y_train.append(key)
        file_paths.extend(t)
    samples_train, _ = extract_features_train(file_paths)
    file_paths = []
    for key in validation_set.keys():
        t = validation_set[key]
        for el in t:
            y_val.append(key)
        file_paths.extend(t)
    samples_val, _ = extract_features_train(file_paths)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Train and get validation and train error for different epoch values
    n_iters = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    train_error = []
    validation_error = []
    for i in n_iters:
        print("Iterations: %d" % i)
        classifier = mlp.Classifier( # Parameters obtained using GridSearchCV for HOG features.
                layers=[
                    mlp.Layer("Rectifier", units=426),
                    mlp.Layer("Softmax")],
                learning_rate=0.005,
                n_iter=i)
        classifier = classifier.fit(samples_train, y_train)
        train_error.append(1.0 - classifier.score(samples_train, y_train))
        validation_error.append(1.0 - classifier.score(samples_val, y_val))

    # Print and plot
    for i,e in enumerate(n_iters):
        print(e, train_error[i], validation_error[i])
    plt.subplot(2, 1, 1)
    plt.plot(n_iters, train_error, label='Train')
    plt.plot(n_iters, validation_error, label='Validation')
    plt.legend(loc='lower left')
    plt.ylim([0, 0.2])
    plt.xlim([15,105])
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.show()

def extract_features_train(file_paths):
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
        fd = get_hog(image)
        features.extend(fd)
        samples.append(features)
    if verbose:
        print("")
    samples = np.array(samples)
    if verbose:
        print("Feature normalizing/scaling")
    normalizer = preprocessing.StandardScaler()
    samples = normalizer.fit_transform(samples)
    return samples, normalizer

def percentage_split(seq, percentages):
    assert sum(percentages) == 1.0
    prv = 0
    size = len(seq)
    cum_percentage = 0
    for p in percentages:
        cum_percentage += p
        nxt = int(cum_percentage * size)
        yield seq[prv:nxt]
        prv = nxt

if __name__ == "__main__":
    main()
