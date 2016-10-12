# This is a list of models we have used since the beginning of the project. Here we can specify options for features and classifiers which would make command-line arguments too complicated.
# These models are accessible using the -m command line option and specifying the name of the function you want to use (ex. -m model_hog_svm).
from feature_extraction import Features
from classifier import *

def model_example_blending():
    # For blending, just specify a tuple of main classifier and a list of its subclassifiers (ex. (main,[sub, sub]) ). Specify as many features as subclassifiers.
    return (KNeighborsClassifier(), [SVMClassifier(), LDAClassifier()]), [Features(['hog']), Features(['raw_rgb'])]

def model_ensemble_hog_mlp_rgb_mlp():
    return [(MLPClassifier(optimize_parameters=False), 0.3), (MLPClassifier(optimize_parameters=False), 0.7)], [Features({'raw_rgb': {'dimensions': [24, 24]}}, feature_scaling='None'), Features(['hog'])]

def model_ensemble_hog_mlp_hue_mlp():
    return [(MLPClassifier(optimize_parameters=False), 0.7), (MLPClassifier(optimize_parameters=False), 0.3)], [Features(['hog']), Features({'raw_hue': {'dimensions': [24, 24]}}, feature_scaling='None')]

def model_rgb_mlp():
    return MLPClassifier(), Features({'raw_rgb': {'dimensions': [24, 24]}}, feature_scaling='None') # You can use a dict if you want to pass parameters to feature functions

def model_hog_mlp():
    return MLPClassifier(optimize_parameters=False), Features(['hog'])

def model_hog_rgb_mlp():
    return MLPClassifier(optimize_parameters=False), Features(['hog', 'color_histograms'])

def model_grayscale_adaboost():
    return AdaBoostClassifier(), Features({'raw_grayscale': {'dimensions': [24, 24]}})

def model_hog_grayscale_svm():
    return SVMClassifier(C=0.0059948425031894088), Features(['raw_grayscale','hog']) #parameter was obtained through GridSearchCV. CV Score=0.963579353594

def model_hog_svm():
    return SVMClassifier(), Features(['hog'])

def model_hog_color_svm():
    return SVMClassifier(), Features(['hog', 'color_histograms']) #We got C=0.01 through GridSearchCV, but this is slightly worse on Kaggle. CV Score = 0.960926193922.

def model_color_knn():
    return KNeighborsClassifier(), Features(['color_histograms'])

def model_color_logreg():
    return LogisticRegressionClassifier(), Features(['color_histograms'])

def model_hog_lda():
    return LDAClassifier(), Features(['hog'])

def model_hog_grayscale_lda():
    return LDAClassifier(), Features(['raw_grayscale','hog'])

def model_hog_logreg():
    return LogisticRegressionClassifier(), Features(['hog'])

def model_hog_pca():
    return PCALogisticRegressionClassifier(), Features(['hog'])

def model_color_hog_pca():
    return PCALogisticRegressionClassifier(), Features(['hog', 'color_histograms'])

def model_hog_color_gd():
    return SGDClassifier(), Features(['hog', 'color_histograms'])
