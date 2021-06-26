from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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

myGene = trainGenerator(2,'../data/Harnverhalt/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_rib-segments.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(myGene,steps_per_epoch=30,epochs=600,callbacks=[model_checkpoint,reduce_lr]) # Originally just 1 epochs, but 300 steps_per_epoch May add more callbacks.
# model.load_weights('unet_rib-segments.hdf5')

testGene = testGenerator("../data/Harnverhalt/test",4)
results = model.predict(testGene,30,verbose=1)
saveResult("../data/Harnverhalt/test",results)
