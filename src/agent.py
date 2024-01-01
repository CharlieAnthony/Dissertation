from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def move(self, dist):
        pass

    @abstractmethod
    def rotate(self, deg):
        pass