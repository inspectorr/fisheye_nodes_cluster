from abc import ABC
import tensorflow as tf
import matplotlib.pyplot as plt


def postprocess_image_default(output_image_data):
    return output_image_data


class ImageToImageMLBackend(ABC):
    def __init__(
            self,
            readme_url,
            preprocess_image,
            model_path_pb=None,
            model_path_tflite=None,
            postprocess_image=postprocess_image_default,
    ):
        self.model_path_pb = model_path_pb
        self.model_path_tflite = model_path_tflite
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
        output_data = None
        if self.model_path_tflite:
            output_data = self.invoke_tflite_interpreter(preprocessed_image, *args)
        if self.model_path_pb:
            output_data = self.invoke_pb_interpreter(preprocessed_image)
        return self.postprocess_image(output_data)

    def invoke_pb_interpreter(self, *args):
        # import pdb; pdb.set_trace()
        model = tf.saved_model.load(self.model_path_pb)
        result = model(args[0])
        return result

    def invoke_tflite_interpreter(self, *tensors):
        """
        Do the main work with TensorFlow
        :param tensors:
        :return:
        """
        interpreter = tf.lite.Interpreter(model_path=self.model_path_tflite)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for i, tensor in enumerate(tensors):
            interpreter.set_tensor(input_details[i]['index'], tensor)

        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index'])
