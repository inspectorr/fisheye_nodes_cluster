import tensorflow as tf


def resize_and_central_crop(img, target_dim):
    # Resize the image so that the shorter dimension becomes the target dim.
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    # Central crop the image.
    img = tf.image.resize_with_crop_or_pad(img, target_dim, target_dim)
    return img
