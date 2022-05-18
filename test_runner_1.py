import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from backend import ImageToImageMLBackend

test_image_path = 'test_images/roman.jpg'


def preprocess_image(source_image_path, target_dim=512):
    source_image = cv2.imread(source_image_path)
    img = source_image.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    # Resize the image so that the shorter dimension becomes the target dim.
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    # Central crop the image.
    img = tf.image.resize_with_crop_or_pad(img, target_dim, target_dim)
    return img


def postprocess_image(output_image_data):
    output = (np.squeeze(output_image_data) + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return np.squeeze(output)


backend = ImageToImageMLBackend(
    model_path='models/lite-model_cartoongan_dr_1.tflite',
    readme_url='https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1',
    preprocess_image=preprocess_image,
    postprocess_image=postprocess_image,
)

output_image = backend.predict(test_image_path)

ImageToImageMLBackend.visualize_for_test((
    ('Source image', plt.imread(test_image_path)),
    ('Cartoonized image', output_image)
))
