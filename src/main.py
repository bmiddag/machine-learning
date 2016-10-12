#!/usr/bin/env python3

import sys, string, os, csv
from model import get_model
import numpy as np
import default_models
from settings import *
import warnings
from ensemble import EnsembleClassifier, BlendedClassifier
import argparse

def parse_args():
    """
    Parse command line arguments.
    """
    methods     = ['MLP', 'Adaboost', 'SVM', 'KNN', 'LogReg', 'LDA', 'PCA', 'SGD', 'ConvNet']
    methods_all = list(methods)
    methods_all.extend(['Ensemble', 'Blended'])
    features    = ['HOG', 'RGB', 'RGB_CN', 'HSV', 'histogram']
    parser = argparse.ArgumentParser(
            description='Traffic Sign Recognition by Mathieu De Coster, Bart Middag and Daan Spitael.',
            epilog='Example of an Ensemble classifier: `-c Ensemble -e SVM MLP -f HOG RGB -w 0.2 0.8`. When using a ConvNet, please use the RGB_CN feature.')
    parser.add_argument('-c', help='<Required (unless -m is specified)> The name of the used classifier', choices=methods_all)
    parser.add_argument('-f', help='<Required (unless -m is specified)> The used features', choices=features, nargs='+')
    parser.add_argument('-e', help='The used classifiers: only required when using Ensemble or Blended classifier; if using the Blended classifier, the first will be the main one', choices=methods, nargs='+')
    parser.add_argument('-w', help='The used weights: only used when using Ensemble classifier; default: uniform', nargs='+', type=float)
    parser.add_argument('-m', help='The name of the default model you want to use, if any. If specified, classifier and features are ignored.')

    args = parser.parse_args()
    if args.m != None:
        if not hasattr(default_models, args.m):
            print("When using a default model, please specify the name of an existing model from model.py.")
            parser.print_help()
            sys.exit(1)
    else:
        if(args.c == None or args.f == None):
            print("Please specify the classifier and features you want to use.")
            parser.print_help()
            sys.exit(1)
        if(args.c == 'Ensemble'):
            if(args.e == None):
                print("Please supply the -e argument when using an Ensemble classifier.")
                parser.print_help()
                sys.exit(1)
            if(len(args.e) < 2 and len(args.f) < 2):
                print("Please provide at least 2 classifiers or feature types when using an Ensemble classifier.")
                parser.print_help()
                sys.exit(1)
            if(len(args.e) != len(args.f)):
                print("Please provide as many classifiers as you do features and vice versa.")
                parser.print_help()
                sys.exit(1)
            if(args.w != None and sum(args.w) != 1.0):
                print("Please make sure that the sum of the weights is 1.")
                parser.print_help()
                sys.exit(1)
        elif(args.c == 'Blended'):
            if(args.e == None):
                print("Please supply the -e argument when using a Blended classifier.")
                parser.print_help()
                sys.exit(1)
            if(len(args.e) < 3 or len(args.f) < 2):
                print("Please provide at least 3 classifiers and 2 feature types when using a Blended classifier.")
                parser.print_help()
                sys.exit(1)
            if(len(args.e) != len(args.f)+1):
                print("Please provide one more classifier (the first will be the main classifier) than you have features.")
                parser.print_help()
                sys.exit(1)
    return vars(args)

def main():
    # Process command line arguments
    arguments = parse_args()

    # Set custom warning format
    overwrite_warnings()

    # Get classifier and features
    classifier, features = get_model(arguments)
    if isinstance(classifier, tuple):
        # Perform blended learning
        print("Performing blended learning.")
        blended_classifier = BlendedClassifier(classifier[0], classifier[1], features)

        # Train the classifier
        blended_classifier.train()

        # Classify test data and export results
        results = blended_classifier.test()
        classes = blended_classifier.getClasses()
        ids = blended_classifier.getIds()
        export_results(ids, results, classes)
    elif isinstance(classifier, list):
        # Perform ensemble learning
        print("Performing ensemble learning.")
        ensemble_classifier = EnsembleClassifier(classifier, features)

        # Cross-validate and train the classifier
        ensemble_classifier.cross_validate()
        ensemble_classifier.train()

        # Classify test data and export results
        results = ensemble_classifier.test()
        classes = ensemble_classifier.getClasses()
        ids = ensemble_classifier.getIds()
        export_results(ids, results, classes)
    else:
        # Perform regular learning
        print("Performing regular learning.")

        # Load training data and perform feature extraction
        train_data, train_metadata = features.extract_features_train()

        # Cross-validate and train the classifier
        classifier.cross_validate(train_data, train_metadata)
        classifier.train(train_data, train_metadata)

        # Load test data and perform feature extraction
        test_data, test_metadata = features.extract_features_test()

        # Classify test data and export results
        test_results = classifier.test(test_data)
        classes = classifier.getClasses()
        export_results(test_metadata.ids, test_results, classes)

def export_results(ids, test_results, classes):
    """
    Export the results to a CSV file.
    """
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    with open(results_file, 'w', newline='') as csvfile:
        resultwriter = csv.writer(csvfile)
        header = ['Id']
        for c in classes:
            header.append(c)
        resultwriter.writerow(header)
        index = 0
        for r in test_results:
            row = [str(int(ids[index]))]
            for p in r:
                row.append(str(p))
            resultwriter.writerow(row)
            index = index + 1

def custom_formatwarning(msg, *a):
    # ignore everything except the message
    return '-- Warning: ' + str(msg) + '\n'

def no_formatwarning():
    return ''

def overwrite_warnings():
    if warnings_level <= 0:
        warnings.formatwarning = no_formatwarning
    elif warnings_level == 1:
        warnings.formatwarning = custom_formatwarning

if __name__ == "__main__":
    main()
