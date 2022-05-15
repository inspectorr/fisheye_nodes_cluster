import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from image import load_img, preprocess_image

image_path = 'roman.jpg'
model_path = 'lite-model_cartoongan_dr_1.tflite'

image = load_img(image_path)

preprocessed_image = preprocess_image(image)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

output = (np.squeeze(output_data)+1.0)*127.5
output = np.clip(output, 0, 255).astype(np.uint8)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(image_path))
plt.title('Source image')
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title('Cartoonized image')
plt.show()
