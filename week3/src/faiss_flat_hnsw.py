import faiss
from src.abstract_vector_index import AbstractVectorIndex

class FaissFlatHNSWIndex(AbstractVectorIndex):

    def __init__(self, d):
        self.index = faiss.IndexHNSWFlat(d, 32)

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        return self.index.search(queries, k)
    
