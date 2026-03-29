import numpy as np
import redis
from redis.commands.search.field import VectorField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import struct
from src.abstract_vector_index import AbstractVectorIndex

class RedisIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = "L2" if metric_type == "l2" else "COSINE"
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=False)
        self.index_name = "bench_vectors"

        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except:
            pass

        schema = [
            NumericField("id"),
            VectorField("vector", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": d,
                "DISTANCE_METRIC": self.metric_type
            })
        ]

        self.client.ft(self.index_name).create_index(
            schema,
            definition=IndexDefinition(prefix=["vec:"], index_type=IndexType.HASH)
        )

    def train(self, data):
        return

    def add(self, data):
        pipe = self.client.pipeline()
        for i, vec in enumerate(data):
            pipe.hset(f"vec:{i}", mapping={
                "id": i,
                "vector": vec.astype(np.float32).tobytes()
            })
        pipe.execute()

    def search(self, queries, k):
        all_D, all_I = [], []

        for q in queries:
            query_bytes = q.astype(np.float32).tobytes()
            q_obj = (
                Query(f"*=>[KNN {k} @vector $vec AS score]")
                .sort_by("score")
                .return_fields("id", "score")
                .dialect(2)
            )
            results = self.client.ft(self.index_name).search(
                q_obj, query_params={"vec": query_bytes}
            )
            all_D.append([float(r.score) for r in results.docs])
            all_I.append([int(r.id.split(":")[1]) for r in results.docs])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)