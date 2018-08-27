import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

"""
Usage for the CIFAR-10 dataset
  The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
  There are 50000 training images and 10000 test images. 
  The dataset is divided into five training batches and one test batch, each with 10000 images. 
  The test batch contains exactly 1000 randomly-selected images from each class. 
  The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. 
  Between them, the training batches contain exactly 5000 images from each class.   

  -shape of each image   :  32h, 32w, 3ch
  -range of each channel :  0.0 - 255.0
  -num of images (total) :  60,000
  -num of images (train) :  50,000 
  -num of images (test)  :  10,000
  -num of class label    :  10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  
  web-site : http://www.cs.toronto.edu/~kriz/cifar.html
"""

nb_classes = 10
argmax_ch  = 255.0

if __name__=='__main__':
    # load CIFAR-10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert pixel value to 0.0-1.0
    X_train = X_train.astype('float32') / 128. #argmax_ch
    X_test  = X_test.astype('float32')  / 128. #argmax_ch

    ####
    # add
    X_train -= 1.
    X_test  -= 1.
    ####

    # convert class label (0-9) to one-hot encoding format
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)

    # save datasets as np.ndarray class format files
    np.save('X_train', X_train)
    np.save('y_train', y_train)
    np.save('X_test' , X_test)
    np.save('y_test' , y_test)
