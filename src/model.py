# Models
from feature_extraction import Features
import default_models
from classifier import *

def get_classifier(name):
    """
    Returns a classifier based on a name.
    Uses default options that were found useful by us.
    """
    if name == 'MLP':
        return MLPClassifier(optimize_parameters=False)
    elif name == 'Adaboost':
        return AdaboostClassifier()
    elif name == 'SVM':
        return SVMClassifier(C=0.0059948425031894088) # obtained using GridSearchCV
    elif name == 'KNN':
        return KNeighborsClassifier()
    elif name == 'LogReg':
        return LogisticRegressionClassifier()
    elif name == 'LDA':
        return LDAClassifier()
    elif name == 'PCA':
        return PCALogisticRegressionClassifier()
    elif name == 'SGD':
        return SGDClassifier()
    elif name == 'ConvNet':
        return ConvNetClassifier(optimize_parameters=False)

def get_features(name):
    """
    Returns a Features object based on a name.
    Uses default options that were found useful by us.
    """
    if name == 'HOG':
        return Features(['hog'])
    if name == 'RGB':
        return Features({'raw_rgb': {'dimensions': [24, 24]}}, feature_scaling='None')
    if name == 'RGB_CN':
        return Features({'raw_rgb_image': {'dimensions': [24, 24]}}, feature_scaling='None')
    if name == 'HSV':
        return Features({'raw_hsv': {'dimensions': [24, 24]}}, feature_scaling='None')
    if name == 'histogram':
        return Features(['color_histograms'])

def get_model(arguments):
    """
    Get a model: classifier and features.
    If classifier and features are lists, we're using an ensemble classifier.
    """
    if arguments['m'] is not None:
        if hasattr(default_models, arguments['m']):
            method = getattr(default_models, arguments['m'])
            return method()
        else:
            raise InputError('This is not an existing default model.')
    elif arguments['c'] == 'Blended':
        subclassifiers = []
        features = []
        main_classifier = arguments['e'][0]
        for i,feature in enumerate(arguments['f']):
            classifier = arguments['e'][i+1]
            subclassifiers.append(get_classifier(classifier))
            features.append(get_features(feature))
        return (get_classifier(main_classifier),subclassifiers), features
    elif arguments['c'] == 'Ensemble':
        # Ensemble model: extract multiple classifiers and features
        classifiers = []
        features = []
        for i,classifier in enumerate(arguments['e']):
            feature = arguments['f'][i]
            classifiers.append((get_classifier(classifier), arguments['w'][i]))
            features.append(get_features(feature))
        return classifiers, features
    else:
        return get_classifier(arguments['c']), get_features(arguments['f'][0])
