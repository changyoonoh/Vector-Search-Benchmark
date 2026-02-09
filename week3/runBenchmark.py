import time
import numpy as np

from src.faiss_flat_l2 import FaissFlatL2Index
from src.faiss_flat_ip import FaissFlatIPIndex
from src.faiss_flat_hnsw import FaissFlatHNSWIndex
from src.faiss_sq_l2 import FaissScalarQuantizerL2Index
from src.faiss_sq_ip import FaissScalarQuantizerIPIndex

def run_benchmark (index, data, queries, k=5):
    t0 = time.perf_counter()

    index.train(data)
    t1 = time.perf_counter()

    index.add(data)
    t2 = time.perf_counter()

    index.search(queries, k)
    t3 = time.perf_counter()

    train_time = t1 - t0
    build_time = t2 - t1
    search_time = t3 - t2
    return train_time, build_time, search_time

def main():
    d = 32
    nq = 20000
    queries = np.random.random((nq, d)).astype("float32")

    sizes = [500, 1000, 2000, 5000]

    index_classes = [
        ("FlatL2", FaissFlatL2Index),
        ("FlatIP", FaissFlatIPIndex),
        ("HNSW", FaissFlatHNSWIndex),
        ("SQ_L2", FaissScalarQuantizerL2Index),
        ("SQ_IP", FaissScalarQuantizerIPIndex),
    ]

    for n in sizes:
        data = np.random.random((n, d)).astype("float32")
        print("Data Size =", n)

        for IndexTypeName, IndexClass in index_classes:
            index = IndexClass(d)
            tt, bt, st = run_benchmark(index, data, queries, k=5)
            print(IndexTypeName, tt, bt, st)

if __name__ == "__main__":
    main()