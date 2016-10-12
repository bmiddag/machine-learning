import sys, string, math
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.base import clone
from settings import *

def get_cross_validation_iterator(y, k=5, labels=[], foldLabels=True):
    """
    Returns cross-validation iterator dependent on:
    * y: The classes
    * k: amount of folds, or amount of labels that are left out in leave-p-out cross-validation
    * Labels: If labels are supplied, samples with the same label will never be split across folds.
    * foldLabels: Whether K-fold is applied using these labels or not (if not, then leave-p-out cross-validation is applied)
    """
    if len(labels) > 0:
        if foldLabels:
            cv = ClassLabelKFold(y, labels, n_folds=k)
        else:
            if k == 1:
                cv = cross_validation.LeaveOneLabelOut(labels)
            else:
                cv = cross_validation.LeavePLabelOut(labels, p=k)
    else:
        cv = cross_validation.StratifiedKFold(y, n_folds=k)
    return cv

def cross_validate(samples, y, classifier, reductor=None, k=5, labels=[], iterations=-1, foldLabels=True, collectPredictions=False):
    """
    Perform cross-validation on the samples.
    Parameters should speak for themselves.
    collectPredictions is used to collect cross-validation probabilities into a [n_samples, n_classes] matrix
    """
    logloss = []
    acc = []
    #classes_y = set(y)
    #print(len(classes_y))
    cv = get_cross_validation_iterator(y, k=k, labels=labels, foldLabels=foldLabels)

    if collectPredictions:
        collected_predictions = np.zeros((samples.shape[0], np.unique(y).shape[0]))
    current_fold = 1
    for train_index, test_index in cv:
        if iterations == 0:
            break
        elif iterations > 0:
            iterations = iterations - 1
        print("Current fold: %d" % current_fold)
        clf = clone(classifier)
        if reductor is not None:
            red = clone(reductor)
        X_train, X_test = samples[train_index], samples[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if reductor is not None:
            X_train = red.fit_transform(X_train, y_train)
        clf.fit(X_train, y_train) # y_train should have all classes
        if hasattr(clf, 'classes_'):
            classes = clf.classes_
        else:
            classes = np.unique(y)
        if reductor is not None:
            X_test = red.transform(X_test)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        # Construct "true values" matrix to be able to calculate logloss
        y_true = np.zeros_like(y_pred_proba)
        y_true_indices = []
        for i in range(0, len(y_test)):
            #print(y_test[i])
            #print(np.where(clf.classes_ == y_test[i]))
            try:
                class_index = np.where(classes == y_test[i])[0][0]
                y_true_indices.append(class_index)
                y_true[i,class_index] = 1
            except IndexError:
                print(np.where(y_test == y_test[i]))
                print(np.where(y_train == y_test[i]))
                print(y_test[i])
                raise

        #logloss.append(metrics.log_loss(y_test, y_pred_proba)) # Cannot use logloss thanks to a bug in sklearn :( It happens when y_true and y_pred have different number of classes.
        #logloss.append(calculate_logloss(y_true, y_pred_proba)) # We can use Kaggle's version of logloss though.
        logloss.append(calculate_multiclass_logloss(np.array(y_true_indices), y_pred_proba)) # We can use Kaggle's version of multiclass logloss though.
        acc.append(metrics.accuracy_score(y_test, y_pred))
        if collectPredictions:
            collected_predictions[test_index,:] = y_pred_proba
        current_fold = current_fold + 1

    #scores = cross_validation.cross_val_score(clf, samples, y, cv=k)
    logloss_avg = math.fsum(logloss) / len(logloss)
    acc_avg = math.fsum(acc) / len(acc)
    if verbose:
        #print("Accuracy: %0.5f" % (acc_avg))
        print("Log Loss: %0.5f - Accuracy: %0.5f" % (logloss_avg, acc_avg))
        # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    if collectPredictions:
        return collected_predictions
    else:
        return acc_avg

# Kaggle's logloss function
def calculate_logloss(act, pred):
    epsilon = 1e-15
    pred = np.maximum(epsilon, pred)
    pred = np.minimum(1-epsilon, pred)
    ll = sum(act*np.log(pred) + np.subtract(1,act)*np.log(np.subtract(1,pred)))
    ll = sum(ll)
    ll = ll * -1.0/len(act)
    return ll

# Kaggle's multiclass logloss function
def calculate_multiclass_logloss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

class ClassLabelKFold(cross_validation.LabelKFold):
    """K-fold iterator variant with non-overlapping labels but with
    overlapping classes.
    The same label will not appear in two different folds (the number of
    distinct labels has to be at least equal to the number of folds) AND
    one fold will never contain all labels of one class.
    The folds are approximately balanced in the sense that the number of
    distinct labels is approximately the same in each fold.
    .. versionadded:: 0.17
    Parameters
    ----------
    labels : array-like with shape (n_samples, )
        Contains a label for each sample.
        The folds are built so that the same label does not appear in two
        different folds.
    y : array-like with shape (n_samples, )
        Contains the class for each sample.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    """
    def __init__(self, y, labels, n_folds=3):
        cross_validation.LabelKFold.__init__(self, labels, n_folds=n_folds)

        unique_labels, labels = np.unique(labels, return_inverse=True)
        n_labels = len(unique_labels)

        if n_folds > n_labels:
            raise ValueError(
                    ("Cannot have number of folds n_folds={0} greater"
                     " than the number of labels: {1}.").format(n_folds,
                                                                n_labels))

        # Weight labels by their number of occurences
        n_samples_per_label = np.bincount(labels)

        # Distribute the most frequent labels first
        indices = np.argsort(n_samples_per_label)[::-1]
        n_samples_per_label = n_samples_per_label[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(n_folds)

        # Mapping from class to amount of labels distributed per fold
        n_classlabels_per_fold = dict()
        keys = set(y.tolist())
        for key in keys:
            n_classlabels_per_fold[key] = np.zeros(n_folds)

        # Mapping from label index to fold index
        label_to_fold = np.zeros(len(unique_labels))

        # Distribute samples by adding the largest weight to the lightest fold
        for label_index, weight in enumerate(n_samples_per_label):
            class_of_label = y[np.where(labels == indices[label_index])[0][0]]
            #if unique_labels[indices[label_index]] == 1753:
            #    print(class_of_label)
            #if class_of_label == "C29":
            #    print(unique_labels[indices[label_index]])
            #    print(np.where(labels == indices[label_index]))
            sorted_cpf = np.argsort(n_classlabels_per_fold[class_of_label])
            least_cpf = n_classlabels_per_fold[class_of_label][sorted_cpf[0]]
            least_samples = float("inf")
            lightest_fold = None
            for i in range(0, len(sorted_cpf)):
                if n_classlabels_per_fold[class_of_label][sorted_cpf[i]] == least_cpf:
                    if n_samples_per_fold[sorted_cpf[i]] < least_samples:
                        least_samples = n_samples_per_fold[sorted_cpf[i]]
                        lightest_fold = sorted_cpf[i]
                else:
                    break
            n_samples_per_fold[lightest_fold] += weight
            label_to_fold[indices[label_index]] = lightest_fold
            n_classlabels_per_fold[class_of_label][lightest_fold] = n_classlabels_per_fold[class_of_label][lightest_fold] + 1
        self.idxs = label_to_fold[labels]
        #np.set_printoptions(threshold=np.nan)
        #print(n_classlabels_per_fold["C29"])
