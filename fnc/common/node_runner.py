from abc import ABC, abstractmethod


class NodeRunner(ABC):
    @abstractmethod
    def run_backend(self, image_path, params=None):
        raise NotImplementedError

    def run(self, image_path, params=None):
        return self.run_backend(image_path, params)
