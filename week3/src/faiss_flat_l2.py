import faiss
from abstract_vector_index import AbstractVectorIndex

class FaissFlatL2Index(AbstractVectorIndex):

    def __init__(self, d):
        self.index = faiss.IndexFlatL2(d)

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        return self.index.search(queries, k)