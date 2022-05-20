import tensorflow as tf

from fnc.common.backend import ImageToImageMLBackend
from fnc.common.image_utils import resize_and_central_crop

# https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/prediction/1
predict_model_path = 'models/magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite'
transfer_model_path = 'models/magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite'

content_image_path = 'test_images/hermitage.jpg'
style_image_path = 'test_images/drawn_city.jpg'


# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    return resize_and_central_crop(image, target_dim=target_dim)


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=predict_model_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=transfer_model_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return stylized_image


preprocessed_style_image = preprocess_image(load_img(style_image_path), target_dim=256)
preprocessed_content_image = preprocess_image(load_img(content_image_path), target_dim=384)

# Calculate style bottleneck for the preprocessed style image.
style_bottleneck = run_style_predict(preprocessed_style_image)
print('Style Bottleneck Shape:', style_bottleneck.shape)

# Stylize the content image using the style bottleneck.
stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)


def postprocess_image(image):
    if len(image.shape) > 3:
        return tf.squeeze(image, axis=0)


ImageToImageMLBackend.visualize_for_test(
    [('Stylized Image', postprocess_image(stylized_image))]
)
