from abc import ABC, abstractmethod

from fnc.common import restore_image


class NodeRunner(ABC):
    @abstractmethod
    def run_backend(self, image_path):
        raise NotImplementedError

    def run(self, image_path):
        output_np_image = self.run_backend(image_path)
        return restore_image(output_np_image, image_path)
