import numpy as np
from annoy import AnnoyIndex
from src.abstract_vector_index import AbstractVectorIndex

class AnnoyIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="euclidean", n_trees=10):
        self.d = d
        self.n_trees = n_trees
        self.metric = "euclidean" if metric_type == "l2" else "angular"
        self.index = AnnoyIndex(d, self.metric)

    def train(self, data):
        return

    def add(self, data):
        for i, vec in enumerate(data):
            self.index.add_item(i, vec.tolist())
        self.index.build(self.n_trees)

    def search(self, queries, k):
        all_D, all_I = [], []

        for q in queries:
            ids, distances = self.index.get_nns_by_vector(
                q.tolist(), k, include_distances=True
            )
            all_I.append(ids)
            all_D.append(distances)

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)