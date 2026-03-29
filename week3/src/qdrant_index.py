import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.abstract_vector_index import AbstractVectorIndex

class QdrantIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = metric_type
        self.collection_name = "bench_vectors"
        self.client = QdrantClient(":memory:")  # in-memory for now

        distance = Distance.EUCLID if metric_type == "l2" else Distance.COSINE

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=d, distance=distance)
        )

    def train(self, data):
        return

    def add(self, data):
        points = [
            PointStruct(id=i, vector=vec.tolist())
            for i, vec in enumerate(data)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, queries, k):
        all_D, all_I = [], []

        for q in queries:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=q.tolist(),
                limit=k
            )
            all_D.append([r.score for r in results])
            all_I.append([r.id for r in results])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)