import numpy as np
from elasticsearch import Elasticsearch
from src.abstract_vector_index import AbstractVectorIndex

class ElasticsearchIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = metric_type
        self.index_name = "bench_vectors"
        self.client = Elasticsearch("http://localhost:9200", request_timeout=300)

        similarity = "l2_norm" if metric_type == "l2" else "cosine"

        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        self.client.indices.create(index=self.index_name, body={
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": d,
                        "index": True,
                        "similarity": similarity
                    }
                }
            }
        })

    def train(self, data):
        return

    def add(self, data):
        # Insert in 5k-doc chunks to stay under the 100MB HTTP payload limit
        chunk_size = 5000
        for start in range(0, len(data), chunk_size):
            ops = []
            for i in range(start, min(start + chunk_size, len(data))):
                ops.append({"index": {"_index": self.index_name}})
                ops.append({"id": i, "vector": data[i].tolist()})
            self.client.bulk(operations=ops)
        self.client.indices.refresh(index=self.index_name)

    def search(self, queries, k):
        all_D, all_I = [], []

        for q in queries:
            results = self.client.search(index=self.index_name, body={
                "knn": {
                    "field": "vector",
                    "query_vector": q.tolist(),
                    "k": k,
                    "num_candidates": k * 10
                }
            })
            hits = results["hits"]["hits"]
            all_D.append([h["_score"] for h in hits])
            all_I.append([h["_source"]["id"] for h in hits])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)