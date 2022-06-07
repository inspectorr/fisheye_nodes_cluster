from fnc.common import (
    NodeRunner, MLBackend,
    squarize_image, save_image_locally, restore_image, tf_to_np, load_img_to_tf
)

prediction_model_path = 'models/tflite/magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite'
transfer_model_path = 'models/tflite/magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite'


backend_prediction = MLBackend(
    model_path_tflite=prediction_model_path,
    readme_url='https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1',
    preprocess=lambda img_path: squarize_image(load_img_to_tf(img_path), target_dim=256)
)

backend_transfer = MLBackend(
    model_path_tflite=transfer_model_path,
    readme_url='https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1',
    preprocess=lambda img_path: squarize_image(load_img_to_tf(img_path), target_dim=384),
    postprocess=lambda image: tf_to_np(image)
)


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

    MLBackend.visualize_for_test([
        ('Original image', plt.imread(content_image_path)),
        ('Stylized Image', stylized_image)
    ])
