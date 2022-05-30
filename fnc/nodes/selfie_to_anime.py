import cv2
import numpy as np
import tensorflow as tf

from fnc.common import NodeRunner, ImageToImageMLBackend, squarize_image, restore_image


def preprocess_image(source_image_path):
    source_image = cv2.imread(source_image_path)
    img = source_image.astype(np.uint8)
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return squarize_image(img, target_dim=256)


def postprocess_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0).numpy()
    image *= 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


backend = ImageToImageMLBackend(
    model_path_tflite='models/tflite/selfie2anime.tflite',
    readme_url='https://github.com/margaretmz/Selfie2Anime-with-TFLite/',
    preprocess_image=preprocess_image,
    postprocess_image=postprocess_image,
)


class Runner(NodeRunner):
    def run_backend(self, image_path, params=None):
        output_np_image = backend.predict(image_path)
        return restore_image(output_np_image, image_path)


runner = Runner()

__all__ = ['runner']

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_image_path = 'test_images/roman.jpg'

    output_image = runner.run(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Anime image', output_image)
    ))
