import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


image_path = '../../../static/test_images/roman.jpg'

# https://tfhub.dev/sayakpaul/lite-model/mirnet-fixed/dr/1
model_path = '../../../static/models/lite-model_mirnet-fixed_dr_1.tflite'


def infer_tflite(image):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)
    output_image = raw_prediction()

    output_image = output_image.squeeze() * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image


IMG_SIZE = 400


def preprocess_image(image_path):
    original_image = Image.open(image_path)
    width, height = original_image.size
    preprocessed_image = original_image.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        Image.ANTIALIAS)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return original_image, preprocessed_image


original_image, preprocessed_image = preprocess_image(image_path)
output_image = infer_tflite(preprocessed_image)


plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Source image')
plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Enhanced image')
plt.show()
