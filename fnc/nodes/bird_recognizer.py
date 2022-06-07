import csv

import tensorflow as tf

from fnc.common import MLBackend, squarize_image, load_img_to_tf, NodeRunner
from fnc.common.google import google

backend = MLBackend(
    readme_url='https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1',
    model_path_tflite='models/tflite/lite-model_aiy_vision_classifier_birds_V1_3.tflite',
    # model_path_pb='models/pb/aiy_vision_classifier_birds_V1_1',
    preprocess=lambda img_path: tf.image.convert_image_dtype(squarize_image(load_img_to_tf(img_path, tf.uint8), 224), tf.uint8),
    postprocess=lambda tensor: tensor[0][len(tensor[0]) - 1]
)


def get_bird_raw_name(code):
    with open('fnc/nodes/data/aiy_birds_V1_labelmap.csv', 'r') as file:
        table = csv.reader(file, delimiter=',')
        for row in table:
            if row[0] == str(code):
                return row[1]
        return 'Unknown'


class Runner(NodeRunner):
    def run_backend(self, image_path, params=None):
        out = backend.predict(image_path)
        bird_name = get_bird_raw_name(out)
        googled = google(bird_name)
        return googled[0]


runner = Runner()

__all__ = ['runner']

if __name__ == '__main__':
    # test_image_path = 'test_images/who.png'
    test_image_path = 'test_images/golub.png'

    runner.run(test_image_path)
