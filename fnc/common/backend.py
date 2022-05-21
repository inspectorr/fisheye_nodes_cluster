from abc import ABC
import tensorflow as tf
import matplotlib.pyplot as plt


def postprocess_image_default(output_image_data):
    return output_image_data


class ImageToImageMLBackend(ABC):
    def __init__(
            self,
            model_path,
            readme_url,
            preprocess_image,
            postprocess_image=postprocess_image_default,
    ):
        self.model_path = model_path
        self.readme_url = readme_url
        self.preprocess_image = preprocess_image
        self.postprocess_image = postprocess_image

    @staticmethod
    def visualize_for_test(images):
        """
        Visualize source and output image using matplotlib
        """
        for i, (title, image) in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(image)
            plt.title(title)
        plt.show()

    def predict(self, image_path, *args):
        preprocessed_image = self.preprocess_image(image_path)
        output_data = self.invoke_interpreter(preprocessed_image, *args)
        return self.postprocess_image(output_data)

    def invoke_interpreter(self, *tensors):
        """
        Do the main work with TensorFlow
        :param tensors:
        :return:
        """
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for i, tensor in enumerate(tensors):
            interpreter.set_tensor(input_details[i]['index'], tensor)

        # todo time measurement
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index'])
