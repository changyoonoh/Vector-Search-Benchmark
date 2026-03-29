import numpy as np
import lancedb
from src.abstract_vector_index import AbstractVectorIndex

class LanceDBIVFIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2", db_path="./lancedb_ivf_bench"):
        self.d = d
        self.metric_type = metric_type
        self.db = lancedb.connect(db_path)
        self.table = None

    def train(self, data):
        return

    def add(self, data):
        docs = [
            {"id": i, "vector": vec.tolist()}
            for i, vec in enumerate(data)
        ]

        self.table = self.db.create_table("vectors", data=docs, mode="overwrite")

        self.table.create_index(
            metric=self.metric_type,
            num_partitions=128,
            num_sub_vectors=8,
        )

    def search(self, queries, k):
        all_D = []
        all_I = []

        for q in queries:
            results = (
                self.table.search(q.tolist())
                .metric(self.metric_type)
                .nprobes(16)
                .limit(k)
                .to_list()
            )

            D_row = [row["_distance"] for row in results]
            I_row = [row["id"] for row in results]

            all_D.append(D_row)
            all_I.append(I_row)

        D = np.array(all_D, dtype=np.float32)
        I = np.array(all_I, dtype=np.int64)
        return D, I