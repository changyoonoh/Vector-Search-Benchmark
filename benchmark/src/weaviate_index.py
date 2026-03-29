import numpy as np
import weaviate
import weaviate.classes as wvc
from src.abstract_vector_index import AbstractVectorIndex

class WeaviateIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = metric_type
        self.collection_name = "BenchVectors"
        self.client = weaviate.connect_to_embedded()

        distance = wvc.config.VectorDistances.L2_SQUARED if metric_type == "l2" else wvc.config.VectorDistances.COSINE

        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        self.collection = self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.flat(
                distance_metric=distance
            )
        )

    def train(self, data):
        return

    def add(self, data):
        objects = [
            wvc.data.DataObject(properties={"doc_id": i}, vector=vec.tolist())
            for i, vec in enumerate(data)
        ]
        self.collection.data.insert_many(objects)

    def search(self, queries, k):
        all_D, all_I = [], []

        for q in queries:
            results = self.collection.query.near_vector(
                near_vector=q.tolist(),
                limit=k,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            all_D.append([o.metadata.distance for o in results.objects])
            all_I.append([o.properties["doc_id"] for o in results.objects])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)
    
