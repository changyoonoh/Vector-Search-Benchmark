import numpy as np
import meilisearch

from src.abstract_vector_index import AbstractVectorIndex


class MeilisearchIndex(AbstractVectorIndex):

    def __init__(self, d, host="http://127.0.0.1:7700"):
        self.d = int(d)
        self.embedder = "manual"
        self.next_id = 0

        self.client = meilisearch.Client(host)
        self.index = self.client.index("bench_vectors_meili")

        self.index.update_settings({
            "embedders": {
                self.embedder: {
                    "source": "userProvided",
                    "dimensions": self.d
                }
            }
        })

    def train(self, data):
        return

    def add(self, data):
        docs = [
            {"id": i, "_vectors": {self.embedder: vec.tolist()}}
            for i, vec in enumerate(data)
        ]
        self.index.add_documents(docs)

    def search(self, queries, k):
        all_ids = []

        for q in queries:
            results = self.index.search("", {
                "vector": q.tolist(),
                "hybrid": {"embedder": self.embedder, "semanticRatio": 1.0},
                "limit": int(k),
            })

            ids = [h["id"] for h in results["hits"]]

            while len(ids) < k:
                ids.append(-1)

            all_ids.append(ids[:k])

        I = np.array(all_ids, dtype=np.int64)
        D = np.zeros_like(I, dtype=np.float32)

        return D, I