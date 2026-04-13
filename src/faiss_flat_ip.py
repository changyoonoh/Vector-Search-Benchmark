import faiss
import numpy as np
from src.abstract_vector_index import AbstractVectorIndex

class FaissFlatIPIndex(AbstractVectorIndex):

    def __init__(self, d):
        self.index = faiss.IndexFlatIP(d)

    def add(self, data):
        data = data.copy()
        faiss.normalize_L2(data)
        self.index.add(data)

    def search(self, queries, k):
        queries = queries.copy()
        faiss.normalize_L2(queries)
        return self.index.search(queries, k)