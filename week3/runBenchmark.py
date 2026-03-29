import time
import numpy as np
import matplotlib.pyplot as plt

from src.faiss_flat_l2 import FaissFlatL2Index
from src.faiss_flat_ip import FaissFlatIPIndex
from src.faiss_flat_hnsw import FaissFlatHNSWIndex
from src.faiss_sq_l2 import FaissScalarQuantizerL2Index
from src.faiss_sq_ip import FaissScalarQuantizerIPIndex
from src.milvus_index import MilvusIndex
from src.meilisearch_index import MeilisearchIndex
from src.lancedb_flat_index import LanceDBFlatIndex
from src.lancedb_ivf_index import LanceDBIVFIndex
from src.qdrant_index import QdrantIndex
from src.weaviate_index import WeaviateIndex

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
    nq = 200
    queries = np.random.random((nq, d)).astype("float32")

    sizes = [1000, 5000, 10000, 20000, 50000]

    index_factories = [ #using factories to make loop simpler, factory is just assiging similar but different stuff to a function so that we call it easier later
    ("FlatL2", lambda d: FaissFlatL2Index(d)),
    ("FlatIP", lambda d: FaissFlatIPIndex(d)),
    ("HNSW", lambda d: FaissFlatHNSWIndex(d)),
    ("SQ_L2", lambda d: FaissScalarQuantizerL2Index(d)),
    ("SQ_IP", lambda d: FaissScalarQuantizerIPIndex(d)),
    ("Milvus_FLAT_L2", lambda d: MilvusIndex(d, metric_type="L2", index_type="FLAT")),
    ("Milvus_FLAT_IP", lambda d: MilvusIndex(d, metric_type="IP", index_type="FLAT")),
    ("Milvus_HNSW_L2", lambda d: MilvusIndex(d, metric_type="L2", index_type="HNSW")),
    ("Milvus_IVF_FLAT_L2", lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_FLAT")),
    ("Milvus_IVF_SQ8_L2",  lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_SQ8")),
    ("Milvus_IVF_PQ_L2",   lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_PQ")),
    ("Meili_vec", lambda d: MeilisearchIndex(d)),
    ("LanceDB_FLAT_L2", lambda d: LanceDBFlatIndex(d, metric_type="l2")),
    ("LanceDB_FLAT_DOT", lambda d: LanceDBFlatIndex(d, metric_type="dot")),
    ("LanceDB_IVF_L2", lambda d: LanceDBIVFIndex(d, metric_type="l2")),
    ("LanceDB_IVF_DOT", lambda d: LanceDBIVFIndex(d, metric_type="dot")),
    ("Qdrant_L2", lambda d: QdrantIndex(d, metric_type="l2")),
    ("Qdrant_Cosine", lambda d: QdrantIndex(d, metric_type="cosine")),
    ("Weaviate_L2", lambda d: WeaviateIndex(d, metric_type="l2")),
    ("Weaviate_Cosine", lambda d: WeaviateIndex(d, metric_type="cosine")),
    ]

    results = {name: {"train": [], "build": [], "search": []} for name, _ in index_factories}

    for n in sizes:
        data = np.random.random((n, d)).astype("float32")
        print("Data Size =", n)

        for IndexTypeName, make_index in index_factories:
            index = make_index(d)
            tt, bt, st = run_benchmark(index, data, queries, k=5)
            print(IndexTypeName, tt, bt, st)
            results[IndexTypeName]["train"].append(tt)
            results[IndexTypeName]["build"].append(bt)
            results[IndexTypeName]["search"].append(st)

    #For search time
    plt.figure()
    for name, _ in index_factories:
        plt.plot(sizes, results[name]["search"], label=name)
    plt.xlabel("Number of Vectors")
    plt.ylabel("Search time (seconds)")
    plt.title("Search Time (" + str(nq) + " queries)")
    plt.legend()
    plt.xticks(sizes, rotation=45)
    plt.tight_layout()

    #For build time
    plt.figure()
    for name, _ in index_factories:
        plt.plot(sizes, results[name]["build"], label=name)
    plt.xlabel("Number of Vectors")
    plt.ylabel("Build time (seconds)")
    plt.title("Build Time (" + str(nq) + " queries)")
    plt.legend()
    plt.xticks(sizes, rotation=45)
    plt.tight_layout()


    plt.show()

if __name__ == "__main__": # the __name__ is ___main__ here only within this file, so this makes this only runnable directly, not when imported, prevents accidential execution?
    main()