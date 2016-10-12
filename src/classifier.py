import sys, string, math
import numpy as np

from settings import *
from cross_validation import cross_validate, get_cross_validation_iterator

from sklearn import grid_search, neighbors, linear_model, decomposition, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sknn import mlp

class BaseClassifier:
    """
    Generic definition of classifier. Other classifiers should either work like this or extend this.
    Methods any classifier needs to support are:
    [*] __init__
    [*] train(train_data, train_metadata)
    [*] cross_validate(train_data, train_metadata, use_labels=False)
    [*] test(test_data)
    Optional arguments to each of those methods are allowed, but a default value should always be supplied by the method itself.
    """

    def __init__(self, classifier, reductor=None):
        self.classifier = classifier
        self.reductor = reductor
        self.classes = []
        if verbose:
            print("Selected classifier: %s" % type(self).__name__)

    def train(self, train_data, train_metadata):
        y = train_metadata.y
        self.classes = sorted(set(y))
        if self.reductor is not None:
            features_before = train_data.shape[1]
            train_data = self.reductor.fit_transform(train_data, y)
            features_after = train_data.shape[1]
            if verbose:
                print("Training: Dimensionality reduction (Before: " + str(features_before) + " - After: " + str(features_after) + ")")
        if verbose:
            print("Training: Fitting")
        self.classifier.fit(train_data, y)
        if verbose:
            print("Training: Done.")

    def cross_validate(self, train_data, train_metadata, use_labels=True, collectPredictions=False):
        y = train_metadata.y
        labels = train_metadata.pole_ids
        if verbose:
            print("Cross-validation")
        if use_labels:
            return cross_validate(train_data, y, self.classifier, reductor=self.reductor, labels=labels, collectPredictions=collectPredictions) # test on 4*2 random pole IDs
        else:
            return cross_validate(train_data, y, self.classifier, reductor=self.reductor, collectPredictions=collectPredictions) # stratified k-fold

    def test(self, test_data):
        if self.reductor is not None:
            if verbose:
                print("Training: Dimensionality reduction (Before: ", end="")
                print(str(test_data.shape[1]) + " - After: ", end="")
            test_data = self.reductor.transform(test_data)
            if verbose:
                print(str(test_data.shape[1]) + ")")
        if verbose:
            print("Testing: Classifying")
        results = self.classifier.predict_proba(test_data)
        if verbose:
            print("Testing: Done.")
        return results

    def getClasses(self):
        if hasattr(self.classifier, 'classes_'):
            return self.classifier.classes_
        else:
            return self.classes

# Below this line are implementations of different classifiers

class SVMClassifier(BaseClassifier):
    def __init__(self, C=1):
        classifier = svm.SVC(probability=True, kernel='linear', C=C)
        #classifier = CalibratedClassifierCV(classifier, method="isotonic", cv=5)
        reductor = LDA()
        BaseClassifier.__init__(self, classifier, reductor=reductor)

class AdaBoostClassifier(BaseClassifier):
    def __init__(self):
        classifier = ensemble.AdaBoostClassifier()
        reductor = LDA()
        BaseClassifier.__init__(self, classifier, reductor=reductor)

class KNeighborsClassifier(BaseClassifier):
    def __init__(self, K=10, reductor=None):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=K)
        BaseClassifier.__init__(self, classifier, reductor=reductor)

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, reductor=None):
        classifier = linear_model.LogisticRegression()
        BaseClassifier.__init__(self, classifier, reductor=reductor)

class PCALogisticRegressionClassifier(LogisticRegressionClassifier):
    def __init__(self):
        pca = decomposition.RandomizedPCA(n_components=24)
        LogisticRegressionClassifier.__init(reductor=pca)

class LDAClassifier(BaseClassifier):
    def __init__(self):
        classifier = LDA()
        BaseClassifier.__init__(self, classifier)

class SGDClassifier(BaseClassifier):
    def __init__(self):
        classifier = linear_model.SGDClassifier(loss='modified_huber')
        reductor = LDA()
        BaseClassifier.__init__(self, classifier, reductor=reductor)

class MLPClassifier(BaseClassifier):
    def __init__(self, optimize_parameters=True):
        self.optimize_parameters = optimize_parameters
        classifier = mlp.Classifier( # Parameters obtained using GridSearchCV for HOG features.
            layers=[
                mlp.Layer("Rectifier", units=426),
                mlp.Layer("Softmax")],
            learning_rate=0.005,
            n_iter=25)
        BaseClassifier.__init__(self, classifier, reductor=None)

    def train(self, train_data, train_metadata):
        if self.optimize_parameters:
            y = train_metadata.y
            self.classifier = self.find_good_classifier(train_data, y)
        super().train(train_data, train_metadata)

    def cross_validate(self, train_data, train_metadata, use_labels=False, collectPredictions=False):
        if collectPredictions:
            return super().cross_validate(train_data, train_metadata, use_labels=use_labels, collectPredictions=collectPredictions)
        # TODO: Enable cross-validation for MLPclassifier when it's actually fast enough...
        # pass

    def find_good_classifier(self, train_data, y):
        num_features = len(train_data[0])
        num_labels = len(set(y))
        num_hidden_units = int((num_features + num_labels) / 2)
        learning_rate = [0.05, 0.01, 0.005, 0.001]
        hidden_layer_units = [min(num_features, num_labels) + 1, num_hidden_units, max(num_features, num_labels) - 1]
        hidden_layer_types = ["Rectifier", "Sigmoid", "Tanh"]

        if verbose:
            print("Finding good classifier parameters...")
            print("Learning rate: %s" % ', '.join([str(f) for f in learning_rate]))
            print("Hidden layer units: %s" % ', '.join([str(i) for i in hidden_layer_units]))
            print("Hidden layer types: %s" % ', '.join(hidden_layer_types))

        nn = mlp.Classifier(
            layers=[
                mlp.Layer("Tanh", units=num_hidden_units),
                mlp.Layer("Softmax")],
                learning_rate=0.001,
                n_iter=25)

        gs = GridSearchCV(nn, param_grid={
            'learning_rate': learning_rate,
            'hidden0__units': hidden_layer_units,
            'hidden0__type': hidden_layer_types}, verbose=1)

        gs.fit(train_data, y)

        if verbose:
            print("Best parameters:")
            print(gs.best_params_)
            print("Score: %f" % gs.score(train_data, y))

        return mlp.Classifier(
                layers=[
                    mlp.Layer(gs.best_params_['hidden0__type'], units=gs.best_params_['hidden0__units']),
                    mlp.Layer("Softmax")],
                learning_rate=gs.best_params_['learning_rate'],
                n_iter=25)

class ConvNetClassifier(BaseClassifier):
    def __init__(self, optimize_parameters=True):
        self.optimize_parameters = optimize_parameters
        classifier = mlp.Classifier( # Parameters obtained using GridSearchCV for HOG features.
            layers=[
                mlp.Convolution("Rectifier", channels=150, kernel_shape=(4,4), pool_shape=(2,2), pool_type='max'),
                mlp.Layer("Softmax")],
            learning_rate=0.005,
            n_iter=25)
        BaseClassifier.__init__(self, classifier, reductor=None)

    def train(self, train_data, train_metadata):
        if self.optimize_parameters:
            y = train_metadata.y
            self.classifier = self.find_good_classifier(train_data, y)
        super().train(train_data, train_metadata)

    def cross_validate(self, train_data, train_metadata, use_labels=False, collectPredictions=False):
        if collectPredictions:
            return super().cross_validate(train_data, train_metadata, use_labels=use_labels, collectPredictions=collectPredictions)

    def find_good_classifier(self, train_data, y):
        num_features = len(train_data[0])
        num_labels = len(set(y))
        num_hidden_units = int((num_features + num_labels) / 2)
        learning_rate = [0.05, 0.01, 0.005, 0.001]
        hidden_layer_units = [min(num_features, num_labels) + 1, num_hidden_units, max(num_features, num_labels) - 1]
        hidden_layer_types = ["Rectifier", "Sigmoid", "Tanh"]

        if verbose:
            print("Finding good classifier parameters...")
            print("Learning rate: %s" % ', '.join([str(f) for f in learning_rate]))
            print("Hidden layer units: %s" % ', '.join([str(i) for i in hidden_layer_units]))
            print("Hidden layer types: %s" % ', '.join(hidden_layer_types))

        nn = mlp.Classifier(
            layers=[
                mlp.Layer("Tanh", units=num_hidden_units),
                mlp.Layer("Softmax")],
                learning_rate=0.001,
                n_iter=25)

        gs = GridSearchCV(nn, param_grid={
            'learning_rate': learning_rate,
            'hidden0__units': hidden_layer_units,
            'hidden0__type': hidden_layer_types}, verbose=1)

        gs.fit(train_data, y)

        if verbose:
            print("Best parameters:")
            print(gs.best_params_)
            print("Score: %f" % gs.score(train_data, y))

        return mlp.Classifier(
                layers=[
                    mlp.Layer(gs.best_params_['hidden0__type'], units=gs.best_params_['hidden0__units']),
                    mlp.Layer("Softmax")],
                learning_rate=gs.best_params_['learning_rate'],
                n_iter=25)
