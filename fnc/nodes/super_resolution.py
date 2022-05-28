import numpy as np
import tensorflow as tf

from fnc.common import NodeRunner, ImageToImageMLBackend, squarize_image


def preprocess_image(image_path):
    img = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4

    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, 0)

    img = squarize_image(img, target_dim=120)
    return img


def postprocess_image(image):
    image = tf.clip_by_value(image, 0, 255)
    image = tf.squeeze(image).numpy()
    return image.astype(np.uint8)


backend = ImageToImageMLBackend(
    model_path_pb='models/pb/esrgan-tf2_1',
    # model_path_tflite='models/tflite/lite-model_esrgan-tf2_1.tflite',
    readme_url='https://tfhub.dev/captain-pool/esrgan-tf2/1',
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

    # test_image_path = 'test_images/babuin.png'
    test_image_path = 'test_images/van_gogh_output.png'

    output_image = runner.run(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Super-resolution image', output_image)
    ))
