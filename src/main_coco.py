from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


#early_stopping = EarlyStopping(patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=3e-6, patience=5, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('coco.hdf5', monitor='loss',verbose=1, save_best_only=True)
tensorboard_callback = TensorBoard(log_dir='../logs', histogram_freq=1)


x_size = 256; y_size = 256 # Input internal image size, external images are resized to this (x,y) format.
batchSize = 2
rate = 3e-4

model = unet(input_size=(x_size, y_size, 3), learningRate=rate)

# Training and validation generators
trainGen = trainGenerator(train_path='../data/coco/train',
                          image_folder='image',
                          mask_folder='mask',
                          aug_dict=data_gen_args,
                          save_to_dir = None,
                          image_color_mode="rgb",
                          mask_color_mode="grayscale",
                          target_size=(x_size, y_size),
                          batch_size=batchSize)

validateGen = validateGenerator(val_path='../data/coco/validate',
                          image_folder='image',
                          mask_folder='mask',
                          image_color_mode="rgb",
                          mask_color_mode="grayscale",
                          target_size=(x_size, y_size),
                          batch_size=batchSize)

# Testing generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '/home/helge/dev/unet/data/coco/test',
        target_size=(x_size, y_size),
        color_mode="rgb",
        shuffle = False,
        class_mode='binary',
        batch_size=1)

# Model training
trainSamples = 1000
valSamples = 700
numEpochs = 100
model.fit(trainGen,
          steps_per_epoch=trainSamples,
          validation_data=validateGen,
          validation_steps=valSamples,
          validation_freq=2,
          epochs=numEpochs,
          batch_size=batchSize,
          callbacks=[model_checkpoint, reduce_lr, tensorboard_callback])

# Load model from previous training
# model.load_weights('coco.hdf5')


# Test phase
results = model.predict(test_generator,verbose=1)
saveResult("../data/coco/result",results)

# This did not work...
#testGene = testGenerator('../data/coco/validate',as_gray=False, num_image=14, target_size=(320, 240), as_jpg=True)

# Code for looking at the filenames in the generators.
# filenames = test_generator.filenames
# for file in filenames :
#     print(file)