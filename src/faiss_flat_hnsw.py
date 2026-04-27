import faiss
from src.abstract_vector_index import AbstractVectorIndex

class FaissFlatHNSWIndex(AbstractVectorIndex):

    def __init__(self, d, M=32, efConstruction=200, efSearch=16):
        self.index = faiss.IndexHNSWFlat(d, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch

    def train(self, data):
        return

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        return self.index.search(queries, k)

    def set_query_params(self, params):
        if "efSearch" in params:
            self.index.hnsw.efSearch = params["efSearch"]
