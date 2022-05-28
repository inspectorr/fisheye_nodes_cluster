import os
import uuid

import numpy as np
import requests
import tensorflow as tf
from PIL import Image

from fnc.common.exceptions import RemoteImageException
from settings import UPLOAD_FOLDER, ALLOWED_EXTENSIONS


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


def get_remote_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code > 400:
        raise RemoteImageException(response.status_code)
    return response.content


def save_image_locally(image_url):
    content = get_remote_image_content(image_url)
    return write_image_file(content)


def generate_image_filepath():
    return os.path.join(UPLOAD_FOLDER,  f'{uuid.uuid4()}.png')


def write_image_file(content, image_path=None):
    image_path = image_path or generate_image_filepath()
    with open(image_path, 'wb') as f:
        f.write(content)
    return image_path


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

