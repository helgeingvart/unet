from __future__ import print_function

import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_uint


Unlabelled = [0, 0, 0]
TennisRacket = [255,0,0]
Ball = [0, 255, 0]
Person = [0,0,255]
thresholds = [0.9, 0.15, 0.005, 0.5] # Threshold on the probability for labelling pixel that item.

COLOR_DICT = np.array([Unlabelled, TennisRacket, Ball, Person])

def adjustData(img, mask, flag_multi_class, num_class):
    if (np.max(img) > 1) :
        img = img/255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)

def trainGenerator(train_path, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale",flag_multi_class=False, num_class=2,
                   save_to_dir=None, target_size=(256, 256), batch_size=32,
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        os.path.join(train_path, "image"),
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="image",
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        os.path.join(train_path, "mask"),
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="mask",
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def validateGenerator(val_path, image_color_mode="grayscale",
                      mask_color_mode="grayscale",
                      flag_multi_class=False, num_class=2, target_size=(256, 256), batch_size=32):
    '''
    Generator for validation images. No augmentation here of course.
    '''
    count=0
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)

    image_generator = image_datagen.flow_from_directory(
        os.path.join(val_path, "image"),
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(
        os.path.join(val_path, "mask"),
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size)
    val_generator = zip(image_generator, mask_generator)
    for (img, mask) in val_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


# This failed to work on color images, but OK for bw :-?
def testGenerator(test_path, num_image=30, target_size=(256, 256), as_gray=True, as_jpg=False):
    for i in range(num_image):
        if (as_jpg) :
            img = io.imread(os.path.join(test_path, "%d.jpg" % i), as_gray=as_gray)
        else :
            img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = trans.resize(img, target_size)
        yield img

def labelVisualize(num_class: int, color_dict, img) :

    img_out = np.zeros((numpy.shape(img)[0], numpy.shape(img)[1], 3))  # Create standard rgb image
    for i in range(num_class):
        img_out[img[:, :, i] >= thresholds[i], :] = color_dict[i]  # Put the right label color in the output image based on probability threshold
    return img_out

def saveResult(save_path, npyfile, flag_multi_class=True, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)  # A conversion to uint image could be attempted using img_as_uint()











# This seems to be a test method
def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr
