import numpy as np
import hnswlib
from src.abstract_vector_index import AbstractVectorIndex

class HNSWLibIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2", ef_construction=200, M=16):
        self.d = d
        self.metric = "l2" if metric_type == "l2" else "cosine"
        self.ef_construction = ef_construction
        self.M = M
        self.index = hnswlib.Index(space=self.metric, dim=d)

    def train(self, data):
        return

    def add(self, data):
        self.index.init_index(
            max_elements=len(data),
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.index.add_items(data)

    def search(self, queries, k):
        labels, distances = self.index.knn_query(queries, k=k)
        return distances.astype(np.float32), labels.astype(np.int64)