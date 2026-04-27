import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from src.abstract_vector_index import AbstractVectorIndex

class PgvectorIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2", index_type="flat", m=16, ef_construction=200, ef_search=200):
        self.d = d
        self.metric_type = metric_type
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.conn = psycopg2.connect(
            host="localhost", port=5432,
            dbname="postgres", user="postgres", password="password"
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.cur.execute("DROP TABLE IF EXISTS bench_vectors")
        self.cur.execute(f"CREATE TABLE bench_vectors (id integer, vector vector({d}))")

    def train(self, data):
        return

    def add(self, data):
        chunk_size = 5000
        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            execute_values(
                self.cur,
                "INSERT INTO bench_vectors (id, vector) VALUES %s",
                [(i, data[i].tolist()) for i in range(start, end)],
            )

        if self.index_type == "hnsw":
            ops = "vector_l2_ops" if self.metric_type == "l2" else "vector_cosine_ops"
            self.cur.execute(
                f"CREATE INDEX ON bench_vectors USING hnsw (vector {ops}) "
                f"WITH (m = {self.m}, ef_construction = {self.ef_construction})"
            )

    def search(self, queries, k):
        all_D, all_I = [], []

        op = "<->" if self.metric_type == "l2" else "<=>"

        if self.index_type == "hnsw":
            self.cur.execute(f"SET hnsw.ef_search = {self.ef_search}")

        for q in queries:
            self.cur.execute(
                f"SELECT id, vector {op} %s::vector AS distance FROM bench_vectors ORDER BY distance LIMIT %s",
                (q.tolist(), k)
            )
            results = self.cur.fetchall()
            all_I.append([r[0] for r in results])
            all_D.append([r[1] for r in results])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)

    def set_query_params(self, params):
        self.ef_search = params.get("ef_search", self.ef_search)
