import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from fnc.common.backend import ImageToImageMLBackend
from fnc.common.image_utils import resize_and_central_crop


def preprocess_image(source_image_path):
    source_image = cv2.imread(source_image_path)
    img = source_image.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return resize_and_central_crop(img, target_dim=512)


def postprocess_image(output_image_data):
    output = (np.squeeze(output_image_data) + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return np.squeeze(output)


backend = ImageToImageMLBackend(
    model_path='static/models/lite-model_cartoongan_dr_1.tflite',
    readme_url='https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1',
    preprocess_image=preprocess_image,
    postprocess_image=postprocess_image,
)


if __name__ == '__main__':
    test_image_path = 'static/test_images/roman.jpg'

    output_image = backend.predict(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Cartoonized image', output_image)
    ))
