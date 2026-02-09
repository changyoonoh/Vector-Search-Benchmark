import faiss
from abstract_vector_index import AbstractVectorIndex

class FaissScalarQuantizerL2Index(AbstractVectorIndex):

    def __init__(self, d):
        self.index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)

    def train(self, data):
        self.index.train(data)

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        return self.index.search(queries, k)
    