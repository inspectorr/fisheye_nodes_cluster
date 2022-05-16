import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from image import load_source_img, preprocess_source_image

image_path = 'test_images/roman.jpg'

# https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1
model_path = 'models/lite-model_cartoongan_dr_1.tflite'

image = load_source_img(image_path)

preprocessed_image = preprocess_source_image(image)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

output = (np.squeeze(output_data)+1.0)*127.5
output = np.clip(output, 0, 255).astype(np.uint8)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

output_image = np.squeeze(output)

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(image_path))
plt.title('Source image')
plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Cartoonized image')
plt.show()
