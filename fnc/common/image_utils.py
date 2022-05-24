import numpy as np
import tensorflow as tf
from PIL import Image


def squarize_image(img, target_dim):
    img = resize_image_by_min_side(img, target_dim)
    img = tf.image.resize_with_pad(img, target_dim, target_dim)
    return img


def resize_image_by_min_side(img, target_dim):
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    return tf.image.resize(img, new_shape)


def restore_image(np_img, orig_image_path):
    orig_size = Image.open(orig_image_path).size
    orig_width, orig_height = orig_size
    max_dim, min_dim = max(orig_size), min(orig_size)
    diff = max_dim - min_dim
    semi_diff = diff // 2

    img = Image.fromarray(np_img)
    img = img.resize((max_dim, max_dim), Image.ANTIALIAS)

    if orig_width != orig_height:
        if orig_width > orig_height:
            img = img.crop((0, semi_diff, orig_width, orig_height + semi_diff))
        else:
            img = img.crop((semi_diff, 0, orig_width + semi_diff, orig_height))
    return np.array(img)




#
# def expand_to_square(pil_img, background_color=(0, 0, 0)):
#     width, height = pil_img.size
#     if width == height:
#         return pil_img, 0, 0
#     elif width > height:
#         result = Image.new(pil_img.mode, (width, width), background_color)
#         diff = (width - height) // 2
#         result.paste(pil_img, (0, diff))
#         return result, 0, diff / height
#     else:
#         result = Image.new(pil_img.mode, (height, height), background_color)
#         diff = (height - width) // 2
#         result.paste(pil_img, (diff, 0))
#         return result, diff / height, 0
#
#
# def crop_to_original(pil_img, x, y):
#
