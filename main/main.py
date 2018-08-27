import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import Sequential, Model
from keras.applications.mobilenetv2 import MobileNetV2
from keras.engine.topology import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
from datetime import datetime

from utils.plot_log import save_plot_log 
from utils.datafeeder import DataFeeder
from models.mobilenet_v2 import MobileNetV2


# set meta params
batch_size = 128
nb_classes = 10
nb_epoch   = 300
nb_data    = 32*32

# set meta info
log_dir         = '../train_log/mobilenet_v2-like_log'
dataset_dir     = '..//datasets/dataset_norm'
model_name      = 'mobilenet_v2-like__' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_arch_path = os.path.join(log_dir, (model_name + '_arch.png'))
model_cp_path   = os.path.join(log_dir, (model_name + '_checkpoint.h5'))
model_csv_path  = os.path.join(log_dir, (model_name + '_csv.csv'))


# load data
DF = DataFeeder(dataset_dir)
X_train, y_train, X_test, y_test = DF.X_train, DF.y_train, DF.X_test, DF.y_test

# data augumatation
datagen = ImageDataGenerator(
        featurewise_center=False, 
        featurewise_std_normalization=False, 
        rotation_range=0.0,
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        vertical_flip=False,
        horizontal_flip=True)
datagen.fit(X_train)


# build model
model = MobileNetV2(input_shape=X_train.shape[1:], include_top=True, alpha=1.0)
model.summary()
print('Model Name: ', model_name)

# save model architechture plot (.png)
from keras.utils import plot_model
plot_model(model, to_file=model_arch_path, show_shapes=True)

'''
# check the keras class of each layer in model
for i, layer in enumerate(model.layers):
    print(i, layer)

# freeze model params
for layer in model.layers[:150]:
    layer.trainable = OB
'''

# set learning rate
learning_rates=[]
for i in range(5):
    learning_rates.append(2e-2)
for i in range(50-5):
    learning_rates.append(1e-2)
for i in range(100-50):
    learning_rates.append(8e-3)
for i in range(150-100):
    learning_rates.append(4e-3)
for i in range(200-150):
    learning_rates.append(2e-3)
for i in range(300-200):
    learning_rates.append(1e-3)

# set callbacks
callbacks = []
#callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
callbacks.append(ModelCheckpoint(model_cp_path, monitor='val_loss', save_best_only=True))
callbacks.append(LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))
callbacks.append(CSVLogger(model_csv_path)) 

# compile & learning model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])
history = model.fit_generator(
              datagen.flow(X_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(X_train) / batch_size,
              epochs=nb_epoch,
              verbose=1,
              callbacks=callbacks,
              validation_data=(X_test, y_test))
   
# validation
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('Model Name: ', model_name)
print('Test loss     : {:.5f}'.format(val_loss))
print('Test accuracy : {:.5f}'.format(val_acc))



# save model "INSTANCE"
f1_name = model_name + '_instance'
f1_path = os.path.join(log_dir, f1_name) + '.h5'
model.save(f1_path)

# save model "WEIGHTs"
f2_name = model_name + '_weights'
f2_path = os.path.join(log_dir, f2_name) + '.h5'
model.save_weights(f2_path)

# save model "ARCHITECHTURE"
f3_name = model_name + '_architechture'
f3_path = os.path.join(log_dir, f3_name) + '.json'
json_string = model.to_json()
with open(f3_path, 'w') as f:
    f.write(json_string)

# save plot figure
save_plot_log(log_dir, model_csv_path, index='acc')
