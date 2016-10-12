# This file takes care of the samples' metadata for our Machine Learning project

import sys, string, os
import numpy as np
from sklearn.utils import shuffle
from operator import itemgetter
from settings import *

class Metadata:
    """
    Class that saves the following metadata for each sample:
    * File path
    * Classification label (if train data)
    * Pole ID (if train data)
    * ID (if test data)
    It is able to collect this metadata from the directories specified in the settings.
    """
    def read_traindata(self):
        """
        Read training data. Saves file paths, a list of training labels,
        and a list of pole ids.
        """
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
        # Sort train data, so that it is always in the same order if it isn't randomized.
        file_paths, y, pole_ids = (list(t) for t in zip(*sorted(zip(file_paths, y, pole_ids))))
        self.set_traindata(file_paths, y, pole_ids)
        
    def randomize_traindata(self):
        self.file_paths, self.y, self.pole_ids = shuffle(self.file_paths, self.y, self.pole_ids)      
        
    def set_traindata(self, file_paths, y, pole_ids):
        self.file_paths = file_paths
        self.y = np.array(y)
        self.pole_ids = pole_ids
        
    def read_testdata(self):
        """
        Read test data. Saves a list of file paths and a list of ids.
        """
        file_paths = []
        ids = []
        for test_img in os.listdir(test_data_folder):
            if test_img == '.DS_Store': # OS X....
                continue
            ids.append(test_img.replace('.png',''))
            file_paths.append(test_data_folder + '/' + test_img)
        ids, file_paths = (list(t) for t in zip(*sorted(zip(ids, file_paths))))
        self.set_testdata(file_paths, ids)
    
    def set_testdata(self, file_paths, ids):
        self.file_paths = file_paths
        self.ids = ids