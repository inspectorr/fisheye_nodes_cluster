from abc import ABC, abstractmethod

from PIL import Image


class NodeRunner(ABC):
    @abstractmethod
    def run(self, *args) -> Image:
        ...
