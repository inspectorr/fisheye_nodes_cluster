import cv2
import numpy as np
import tensorflow as tf

from fnc.common import NodeRunner, ImageToImageMLBackend, squarize_image, restore_image
from fnc.nodes.stylization import load_img


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
          is the color indexed by the corresponding element in the input label
          to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
          map maximum entry.
      """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def preprocess_image(source_image_path):
    source_image = cv2.imread(source_image_path)
    img = source_image.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return squarize_image(img, target_dim=513)


def postprocess_image(image):
    seg_map = tf.argmax(tf.image.resize(image, (513, 513)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.uint8)
    seg_map = label_to_color_image(seg_map).astype(np.uint8)
    new_seg_image = np.where(seg_map == 0, seg_map, 255)
    return new_seg_image


backend = ImageToImageMLBackend(
    model_path_tflite='models/tflite/mobilenetv2_coco_voctrainval.tflite',
    readme_url='https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/Semantic_Segmentation_%2B_Background_Removal_%2B_Style_Transfer.ipynb',
    preprocess_image=preprocess_image,
    postprocess_image=postprocess_image,
)


class Runner(NodeRunner):
    def run_backend(self, image_path, params=None):
        output_np_image = backend.predict(image_path)

        # output_np_image = np.where(output_np_image == 15, output_np_image, 0)  # Person index is 15
        # cropped_image_np = tf.squeeze(squarize_image(load_img(image_path), target_dim=513), axis=0).numpy()
        cropped_image_np = tf.squeeze(squarize_image(load_img(image_path), target_dim=513), axis=0).numpy()
        # import pdb; pdb.set_trace()
        new_seg_image_gray = cv2.cvtColor(output_np_image, cv2.COLOR_RGB2GRAY)  # Convert the mask to grayscale
        masked_out = cv2.bitwise_and(cropped_image_np, cropped_image_np, mask=new_seg_image_gray)  # Blend the mask
        masked_out_new = np.where(masked_out != 0, masked_out, 255)  # Remove the background
        # return masked_out_new
        # return cropped_image_np

        if len(masked_out_new.shape) > 3:
            masked_out_new = tf.squeeze(masked_out_new, axis=0).numpy()
        masked_out_new *= 255
        masked_out_new = np.clip(masked_out_new, 0, 255).astype(np.uint8)

        return restore_image(masked_out_new, image_path)

        # return restore_image(output_np_image, image_path)
        # return restore_image(cropped_image_np, image_path)

        # source_image = cv2.imread(image_path)
        # img = source_image.astype(np.float32) / 127.5 - 1
        # img = np.expand_dims(source_image.astype(np.float32), 0)
        # img = tf.convert_to_tensor(img)
        # img = squarize_image(img, target_dim=513)
        # img = squarize_image(load_img(image_path), target_dim=513)
        #
        # return tf.squeeze(img, axis=0).numpy()


runner = Runner()

__all__ = ['runner']

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_image_path = 'test_images/roman.jpg'

    output_image = runner.run(test_image_path)

    ImageToImageMLBackend.visualize_for_test((
        ('Source image', plt.imread(test_image_path)),
        ('Masked image', output_image)
    ))
