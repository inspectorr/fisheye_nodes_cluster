import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

image_path = 'test_images/cat.jpeg'

# https://tfhub.dev/sayakpaul/lite-model/boundless-quarter/dr/1
model_path = 'models/lite-model_boundless-quarter_dr_1.tflite'


def preprocess_image(image_path):
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    # crop to make the image square
    pil_image = pil_image.crop((0, 0, height, height))
    pil_image = pil_image.resize((257, 257), Image.ANTIALIAS)
    image_unscaled = np.array(pil_image)
    image_np = np.expand_dims(
        image_unscaled.astype(np.float32) / 255., axis=0)
    return image_np


preprocessed_image = preprocess_image(image_path)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

print('input_details', input_details)
print('output_details', output_details)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
masked_data = interpreter.get_tensor(output_details[1]['index'])

output = np.squeeze(output_data)
masked = np.squeeze(masked_data)

plt.subplot(1, 3, 1)
plt.imshow(plt.imread(image_path))
plt.title('Source image')
plt.subplot(1, 3, 2)
plt.imshow(output)
plt.title('Extrapolated image')
plt.subplot(1, 3, 3)
plt.imshow(masked)
plt.title('Masked image')
plt.show()
