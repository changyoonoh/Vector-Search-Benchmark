from abc import ABC
from abc import abstractmethod

class AbstractVectorIndex(ABC):

    def train(self, data): #not pass because sometimes it is not used? 
        return

    @abstractmethod
    def add(self, data):
        pass

    @abstractmethod
    def search(self, queries, k):
        pass
