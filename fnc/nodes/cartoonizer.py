import cv2
import numpy as np
import tensorflow as tf

from fnc.common import NodeRunner, ImageToImageMLBackend, squarize_image


def preprocess_image(source_image_path):
    source_image = cv2.imread(source_image_path)
    img = source_image.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return squarize_image(img, target_dim=512)


def postprocess_image(output_image_data):
    output = (np.squeeze(output_image_data) + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return np.squeeze(output)


backend = ImageToImageMLBackend(
    model_path='models/lite-model_cartoongan_int8_1.tflite',
    readme_url='https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1',
    preprocess_image=preprocess_image,
    postprocess_image=postprocess_image,
)


class Runner(NodeRunner):
    def run_backend(self, image_path, params=None):
        return backend.predict(image_path)


runner = Runner()

__all__ = ['runner']

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_image_path = 'test_images/witch.jpg'

    output_image = runner.run(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Cartoonized image', output_image)
    ))
