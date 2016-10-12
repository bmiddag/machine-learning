from settings import *
import numpy as np

class EnsembleClassifier:
    """
    The ensemble classifier takes a list of classifiers and features,
    and combines the results of these using a normalized weighted sum.
    Make sure that the weights sum to 1.
    """
    def __init__(self, classifiers, features):
        self.classifiers = [x[0] for x in classifiers]
        self.weights = [x[1] for x in classifiers]
        if not sum(self.weights) == 1.0:
            print("Warning: classifier weights should sum to 1!")
        self.features = features
        self.results = []
        self.train_data = []
        self.test_data = []
        self.train_metadata = []
        self.test_metadata = []

        for i, feature_list in enumerate(features):
            if verbose:
                print("Loading features for classifier: %s" % type(self.classifiers[i]).__name__)
            train_data, train_metadata = feature_list.extract_features_train()
            self.train_data.append(train_data)
            self.train_metadata.append(train_metadata)

            test_data, test_metadata = feature_list.extract_features_test()
            self.test_data.append(test_data)
            self.test_metadata.append(test_metadata)

    def train(self):
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Training classifier: %s" % type(self.classifiers[i]).__name__)
            classifier.train(self.train_data[i], self.train_metadata[i])

    def cross_validate(self, use_labels=True):
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Cross-validating classifier: %s" % type(self.classifiers[i]).__name__)
            classifier.cross_validate(self.train_data[i], self.train_metadata[i], use_labels)

    def test(self):
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Testing with classifier: %s" % type(self.classifiers[i]).__name__)
            self.results.append(np.array(classifier.test(self.test_data[i])))
        # This is where the ensemble classifier actually works:
        # We combine the results of all classifiers by using their weights
        if verbose:
            print("Ensembling.")
        sum_vector = np.array(self.results[0]) * self.weights[0]
        for i, result_list in enumerate(self.results[1:]):
            sum_vector += result_list * self.weights[i + 1]
        # Normalize
        sum_vector = sum_vector / np.amax(sum_vector)
        return sum_vector

    def getClasses(self):
        # Classes are identical in subclassifiers
        return self.classifiers[0].getClasses()

    def getIds(self):
        # Ids are identical in subclassifiers
        return self.test_metadata[0].ids

class BlendedClassifier:
    def __init__(self, main_classifier, classifiers, features):
        self.main_classifier = main_classifier
        self.classifiers = classifiers
        self.features = features
        self.results = []
        self.train_data = []
        self.test_data = []
        self.train_metadata = []
        self.test_metadata = []
        for i, feature_list in enumerate(features):
            if verbose:
                print("Loading features for classifier: %s" % type(self.classifiers[i]).__name__)
            train_data, train_metadata = feature_list.extract_features_train(allow_shuffling=False)
            self.train_data.append(train_data)
            self.train_metadata.append(train_metadata)

            test_data, test_metadata = feature_list.extract_features_test(allow_IO=False)
            self.test_data.append(test_data)
            self.test_metadata.append(test_metadata)
    
    def train(self, use_labels=True):
        num_classes = np.unique(self.train_metadata[0].y).shape[0]
        classifier_predictions = np.zeros((self.train_data[0].shape[0],len(self.classifiers)*num_classes))
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Cross-validating classifier: %s" % type(self.classifiers[i]).__name__)
            pred = classifier.cross_validate(self.train_data[i], self.train_metadata[i], use_labels=use_labels, collectPredictions=True)
            classifier_predictions[:,range(num_classes*i,num_classes*(i+1))] = pred
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Training classifier: %s" % type(self.classifiers[i]).__name__)
            classifier.train(self.train_data[i], self.train_metadata[i])
        if verbose:
            print("Training main classifier: %s" % type(self.main_classifier).__name__)
        self.main_classifier.train(classifier_predictions, self.train_metadata[0])
            
    def cross_validate(self, use_labels=True):
        pass # Cross-validation is a necessary part of training for blended classifier, so it's not necessary as a separate method

    def test(self):
        num_classes = np.unique(self.train_metadata[0].y).shape[0]
        classifier_predictions = np.zeros((self.test_data[0].shape[0],len(self.classifiers)*num_classes))
        for i, classifier in enumerate(self.classifiers):
            if verbose:
                print("Testing with classifier: %s" % type(self.classifiers[i]).__name__)
            classifier_predictions[:,range(num_classes*i,num_classes*(i+1))] = classifier.test(self.test_data[i])
        if verbose:
            print("Testing with main classifier: %s" % type(self.main_classifier).__name__)
        return self.main_classifier.test(classifier_predictions)

    def getClasses(self):
        # Classes are identical in subclassifiers
        return self.main_classifier.getClasses()

    def getIds(self):
        # Ids are identical in subclassifiers
        return self.test_metadata[0].ids
    