import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import Resampling

from fnc.common import NodeRunner, ImageToImageMLBackend


TARGET_HEIGHT = 512


def preprocess_image(image_path):
    img = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4

    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)

    return downscale_image(img)
    #
    # img = tf.expand_dims(img, 0)
    # return img.numpy()


def downscale_image(image):
    """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
    """
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))

    ratio = image_size[0] / image_size[1]
    target_height = TARGET_HEIGHT
    target_width = round(target_height * ratio)

    lr_image = np.asarray(Image.fromarray(image.numpy()).resize(
        [target_width, target_height],
        Resampling.BICUBIC
    ))

    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image.numpy()


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

    test_image_path = 'test_images/babuin.png'

    output_image = runner.run(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Super-resolution image', output_image)
    ))
