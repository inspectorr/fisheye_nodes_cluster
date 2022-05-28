import numpy as np
import tensorflow as tf

from fnc.common import NodeRunner, ImageToImageMLBackend, squarize_image, save_image_locally, restore_image

prediction_model_path = 'models/tflite/magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite'
transfer_model_path = 'models/tflite/magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite'


def postprocess_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0).numpy()
    image *= 255
    return image.astype(np.uint8)


backend_prediction = ImageToImageMLBackend(
    model_path_tflite=prediction_model_path,
    readme_url='https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1',
    preprocess_image=lambda img_path: squarize_image(load_img(img_path), target_dim=256)
)

backend_transfer = ImageToImageMLBackend(
    model_path_tflite=transfer_model_path,
    readme_url='https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1',
    preprocess_image=lambda img_path: squarize_image(load_img(img_path), target_dim=384),
    postprocess_image=postprocess_image
)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


class Runner(NodeRunner):
    def run_backend(self, image_path, params=None):
        style_image = params.get('style_image') if params else None
        if not style_image:
            raise Exception('No style image provided.')

        style_image_local = save_image_locally(style_image) if not params.get('PROTECTED_is_local_file') else style_image

        style_bottleneck = backend_prediction.predict(style_image_local)

        output_np_image = backend_transfer.predict(image_path, style_bottleneck)

        return restore_image(output_np_image, image_path)


runner = Runner()

__all__ = ['runner']

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    content_image_path = 'test_images/ozerki.jpg'

    stylized_image = runner.run(
        content_image_path,
        params={
            'style_image': 'test_images/abandoned_city.jpeg',
            'PROTECTED_is_local_file': True,
        }
    )

    ImageToImageMLBackend.visualize_for_test([
        ('Original image', plt.imread(content_image_path)),
        ('Stylized Image', stylized_image)
    ])
