import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from src.abstract_vector_index import AbstractVectorIndex

class PgvectorIndex(AbstractVectorIndex):

    def __init__(self, d, metric_type="l2"):
        self.d = d
        self.metric_type = metric_type

        self.conn = psycopg2.connect(
            host="localhost", port=5432,
            dbname="postgres", user="postgres", password="password"
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

        register_vector(self.conn)

        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.cur.execute("DROP TABLE IF EXISTS bench_vectors")
        self.cur.execute(f"CREATE TABLE bench_vectors (id integer, vector vector({d}))")

    def train(self, data):
        return

    def add(self, data):
        for i, vec in enumerate(data):
            self.cur.execute(
                "INSERT INTO bench_vectors (id, vector) VALUES (%s, %s)",
                (i, vec.tolist())
            )

    def search(self, queries, k):
        all_D, all_I = [], []

        op = "<->" if self.metric_type == "l2" else "<=>"

        for q in queries:
            self.cur.execute(
                f"SELECT id, vector {op} %s AS distance FROM bench_vectors ORDER BY distance LIMIT %s",
                (q.tolist(), k)
            )
            results = self.cur.fetchall()
            all_I.append([r[0] for r in results])
            all_D.append([r[1] for r in results])

        return np.array(all_D, dtype=np.float32), np.array(all_I, dtype=np.int64)