import tensorflow as tf
import numpy as np
import cv2

# todo now using this only for runner 1, move to class


def load_source_img(path_to_img):
    img = cv2.imread(path_to_img)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img


# Function to pre-process by resizing an central cropping it.
def preprocess_source_image(image, target_dim=512):
    # Resize the image so that the shorter dimension becomes the target dim.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image
