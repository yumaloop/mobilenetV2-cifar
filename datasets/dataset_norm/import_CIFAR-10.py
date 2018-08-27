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

    # set data type as 'float32'
    X_train = X_train.astype('float32') #argmax_ch
    X_test  = X_test.astype('float32')  #argmax_ch

    def ch_wise_normalization(X_type, ch):
        mean_ch = X_type[:, :, :, ch].mean()
        std_ch = X_type[:, :, :, ch].std()
        X_type[:, :, :, ch] = (X_type[:, :, :, ch] - mean_ch) / std_ch
        return X_type[:, :, :, ch]

    # normalize data for each R-G-B(0, 1, 2) channel 
    X_train[:, :, :, 0] = ch_wise_normalization(X_train, 0)
    X_train[:, :, :, 1] = ch_wise_normalization(X_train, 1)
    X_train[:, :, :, 2] = ch_wise_normalization(X_train, 2)

    X_test[:, :, :, 0]  = ch_wise_normalization(X_test, 0)
    X_test[:, :, :, 1]  = ch_wise_normalization(X_test, 1)
    X_test[:, :, :, 2]  = ch_wise_normalization(X_test, 2)

    # convert class label (0-9) to one-hot encoding format
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)

    # save datasets as np.ndarray class format files
    np.save('X_train', X_train)
    np.save('y_train', y_train)
    np.save('X_test' , X_test)
    np.save('y_test' , y_test)



'''
### IPython debug code ###

import numpy as np
X_train=np.load('X_train.npy')
X_test=np.load('X_test.npy')

[In]  X_train.mean(axis=(0, 1, 2))
[Out] array([-1.3623739e-05, -5.9634608e-06,  1.0563416e-05], dtype=float32)

[In]  X_train.std(axis=(0, 1, 2))
[Out] array([0.9330368 , 0.9294675 , 0.92152774], dtype=float32)

[In]  X_test.mean(axis=(0, 1, 2))
[Out] array([-7.8059866e-06, -4.9333439e-06, -2.0499765e-06], dtype=float32)

[In]  X_test.std(axis=(0, 1, 2))
[Out] array([0.99158114, 0.99174786, 0.9931346 ], dtype=float32)

'''




