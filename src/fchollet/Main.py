import os
from Model import *
from Generator import *

# input_dir = "../../data/oxford-pets/images/"
input_dir = "/home/helge/dev/unet/data/coco/fchollet-test/image/"
# target_dir = "../../data/oxford-pets/annotations/trimaps/"
target_dir = "/home/helge/dev/unet/data/coco/fchollet-test/label/"

# img_size = (160, 160)
img_size = (256, 256)
num_classes = 3    # Minimum is 3. However, we may omit the last label in training data to trick it into a binary classifier
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        # if fname.endswith(".png") and not fname.startswith(".")
        if fname.endswith(".jpg") and not fname.startswith(".")
    ]
)

from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps, Image

# Display input image #19

im = Image.open(input_img_paths[19])
# im.show()

# Display auto-contrast version of corresponding target (per-pixel categories)
arr = np.array(load_img(target_img_paths[19]))
print(f"Max {np.max(arr)}")

img = PIL.ImageOps.autocontrast(load_img(target_img_paths[19]))
# img.show()

model = get_model(img_size, num_classes)
# model = get_old(img_size, num_classes)
model.summary()

import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
# train_gen = Generator(batch_size, img_size, train_input_img_paths, train_target_img_paths, adjustLabel=False)
# val_gen = Generator(batch_size, img_size, val_input_img_paths, val_target_img_paths, adjustLabel=False)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_gen = trainGenerator(train_path='../../data/coco/train',
                          aug_dict=data_gen_args,
                          image_color_mode="rgb",
                          mask_color_mode="grayscale",
                          target_size=img_size,
                          batch_size=batch_size) #, flag_multi_class=True, num_class=4)

val_gen = validateGenerator(val_path='../../data/coco/validate',
                                image_color_mode="rgb",
                                mask_color_mode="grayscale",
                                target_size=img_size,
                                batch_size=batch_size) #, flag_multi_class=True, num_class=4)


"""## Train the model"""

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.

# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
model.compile(optimizer=Adam(lr=1e-4), loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("segmentation.h5", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='../../logs', histogram_freq=1)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50
train_samples = 1000  # Should generate this automatically by considering content of image directories.
val_samples = 400  # Should generate this automatically by considering content of image directories.
# model.fit(train_gen,
#           epochs=epochs,
#           validation_data=val_gen,
#           callbacks=callbacks,
#           steps_per_epoch=train_samples/batch_size,
#           validation_steps=val_samples,
#           validation_freq=4)
model.load_weights('segmentation.h5')

"""## Visualize predictions"""

# Generate predictions for all images in the validation set

test_gen = Generator(batch_size, img_size, val_input_img_paths, val_target_img_paths, adjustLabel=False)
val_preds = model.predict(test_gen)

import matplotlib.pyplot as plt

def display_mask(i, originalIm: Image):
    """Quick utility to display a model's prediction."""
    # mask = np.argmax(val_preds[i], axis=-1)
    # mask = np.expand_dims(mask, axis=-1)
    # img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    # img = img.resize(size)
    # img.show()
    firstLabel = val_preds[i,:,:,0]*255
    secondLabel = val_preds[i,:,:,1]*255
    mask = Image.fromarray(secondLabel)
    mask = mask.resize(originalIm.size, Image.BICUBIC)
    # thirdLabel = val_preds[i,:,:,2]*255
    plt.subplot(2, 2, 1)
    plt.imshow(np.array(originalIm))
    plt.subplot(2,2,2)
    plt.imshow(np.array(PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))))
    # plt.imshow(secondLabel, 'gray', interpolation='none')
    plt.subplot(2,2,3)
    plt.imshow(mask, 'gray', interpolation='none')
    # plt.imshow(thirdLabel, 'gray', interpolation='none')
    plt.subplot(2,2,4)

    # original
    plt.imshow(np.array(originalIm))
    plt.imshow(np.array(mask), 'jet', interpolation='none', alpha=0.8)
    plt.show()




# im.show()

# Display ground-truth target mask
# img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
# img.show()

# Display mask predicted by our model
plt.ion()
for i in range(99) :
    im = Image.open(val_input_img_paths[i])
    display_mask(i, im)  # Note that the model only sees inputs at 150x150.
    plt.waitforbuttonpress()
    plt.clf()

