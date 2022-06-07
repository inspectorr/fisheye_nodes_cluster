from abc import ABC
import tensorflow as tf
import matplotlib.pyplot as plt


class MLBackend(ABC):
    def __init__(
            self,
            readme_url,
            model_path_pb=None,
            model_path_tflite=None,
            preprocess=lambda x: x,
            postprocess=lambda x: x,
    ):
        self.readme_url = readme_url
        self.preprocess_func = preprocess
        self.postprocess_func = postprocess
        self.tflite_interpreter = None
        if model_path_tflite:
            self.tflite_interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
            print(f'TFLite loaded: {model_path_tflite}')
        self.pb_interpreter = None
        if model_path_pb:
            self.pb_interpreter = tf.saved_model.load(model_path_pb)
            print(f'TB loaded: {model_path_pb}')

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

    def predict(self, data, *args):
        preprocessed_data = self.preprocess_func(data)
        output_data = None
        if self.tflite_interpreter:
            output_data = self.invoke_tflite_interpreter(preprocessed_data, *args)
        if self.pb_interpreter:
            output_data = self.invoke_pb_interpreter(preprocessed_data)
        return self.postprocess_func(output_data)

    def invoke_pb_interpreter(self, data):
        return self.pb_interpreter(data)

    def invoke_tflite_interpreter(self, *tensors):
        interpreter = self.tflite_interpreter
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for i, tensor in enumerate(tensors):
            interpreter.set_tensor(input_details[i]['index'], tensor)

        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]['index'])
