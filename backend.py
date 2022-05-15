from abc import ABC, abstractmethod

from PIL import Image


class MLBackend(ABC):
    @property
    @abstractmethod
    def model_path(self) -> str:
        """
        Path to .tflite model
        """

    @property
    @abstractmethod
    def model_original_readme_url(self) -> str:
        """
        Readme URL for model where it's downloadable
        """

    # @property
    # @abstractmethod
    # def url_name(self) -> str:
    #     """
    #     Name be used for api access ?
    #     """

    def preprocess_image(self, source_image):
        """
        Converts an image to input data for tf model
        """

    def set_params_to_tensor(self, params_dict):
        ...

    @abstractmethod
    def visualize_for_test(self, source_image, output_image):
        """
        Visualize source and output image using matplotlib
        todo default ?
        """

    def predict(self, source_image) -> Image:
        """
        Do the main work with TensorFlow
        :param source_image:
        :return:
        """
        ...


# class TestBackend(MLBackend):
#     @property
#     def model_path(self) -> str:
#         return 'models/...'
