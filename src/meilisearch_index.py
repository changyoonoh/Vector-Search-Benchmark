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

        task = self.index.update_settings({
            "embedders": {
                self.embedder: {
                    "source": "userProvided",
                    "dimensions": self.d
                }
            }
        })
        self.client.wait_for_task(task.task_uid)

    def train(self, data):
        return

    def add(self, data):
        # Insert in 5k-doc chunks to stay under the 100MB HTTP payload limit
        chunk_size = 5000
        for start in range(0, len(data), chunk_size):
            chunk_docs = [
                {"id": i, "_vectors": {self.embedder: data[i].tolist()}}
                for i in range(start, min(start + chunk_size, len(data)))
            ]
            task = self.index.add_documents(chunk_docs)
            self.client.wait_for_task(task.task_uid, timeout_in_ms=300000)

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