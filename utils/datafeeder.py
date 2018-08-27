import logging
import os
import numpy as np
from glob import glob

logger = logging.getLogger(__name__)


class SplitFeeder(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train

        train_size = len(X_train)
        assert (train_size == len(y_train))

        self.X_test = X_test
        self.y_test = y_test

        test_size = len(X_test)
        assert (test_size == len(y_test))

        self.orig_train_keys = np.arange(train_size)
        self.orig_test_keys = np.arange(test_size)

    def __call__(self, keys, train=True):
        if train:
            X = self.X_train
            y = self.y_train
        else:
            X = self.X_test
            y = self.y_test

        for key in keys:
            yield X[key].copy(), y[key].copy()


class DataFeeder(SplitFeeder):
    def __init__(self, datasets_path):
        X_train, y_train, X_test, y_test = self.load_train_data(datasets_path)
        super(DataFeeder, self).__init__(X_train, y_train, X_test, y_test)


    def load_train_data(self, datasets_path):
            datasets_files_path = glob(os.path.join(datasets_path, '*'))

            for path in datasets_files_path:
                if   'X_train' in path:
                    X_train = np.load(path)
                elif 'y_train' in path:
                    y_train = np.load(path)
                elif 'X_test'  in path:
                    X_test = np.load(path)
                elif 'y_test'  in path:
                    y_test = np.load(path)
                    
            return X_train, y_train, X_test, y_test
