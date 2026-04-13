import numpy as np
import chromadb
from src.abstract_vector_index import AbstractVectorIndex

class ChromaIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = metric_type
        self.client = chromadb.Client()  # in-memory
        
        space = "l2" if metric_type == "l2" else "cosine"
        
        try:
            self.client.delete_collection("bench_vectors")
        except:
            pass

        self.collection = self.client.create_collection(
            name="bench_vectors",
            metadata={"hnsw:space": space}
        )

    def train(self, data):
        return

    def add(self, data):
        batch_size = 5000
        for start in range(0, len(data), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=[str(i) for i in range(start, min(end, len(data)))],
                embeddings=data[start:end].tolist()
            )

    def search(self, queries, k):
        all_D, all_I = [], []

        results = self.collection.query(
            query_embeddings=queries.tolist(),
            n_results=k
        )

        for ids, distances in zip(results["ids"], results["distances"]):
            all_I.append([int(i) for i in ids])
            all_D.append(distances)

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)