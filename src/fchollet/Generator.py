from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator


class Generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, adjustLabel=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.adjustLabel = adjustLabel

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)  # This just means that we are adding another
            if self.adjustLabel:
                # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
                y[j] -= 1
        # y[y==2] = 1  # Just a test to see if we can remove the border labelling on the pets and simply operate as a binary classification
        return x, y


def trainGenerator(train_path, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", flag_multi_class=False, num_class=2,
                   save_to_dir=None, target_size=(256, 256), batch_size=32,
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    # image_datagen = ImageDataGenerator(**aug_dict, rescale=1. / 255.)
    # mask_datagen = ImageDataGenerator(**aug_dict)  # Handle possible rescale intelligently later on
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator()

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
        img, mask = adjustData(img, mask)
        yield (img, mask)


def validateGenerator(val_path, image_color_mode="grayscale",
                      mask_color_mode="grayscale",
                      flag_multi_class=False, num_class=2, target_size=(256, 256), batch_size=32):
    '''
    Generator for validation images. No augmentation here of course.
    '''
    count = 0
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
    image_datagen
    for (img, mask) in val_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def adjustData(img, mask):
    if np.max(mask) > 1:
        # print(f'Did adjust, mask max were: {np.max(mask)}')
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask
