import os
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import h5py

def get_timeout(n):
    if n <= 10000:
        return 180    #  3 min
    elif n <= 50000:
        return 600    # 10 min
    elif n <= 100000:
        return 1200   # 20 min
    elif n <= 300000:
        return 2700   # 45 min
    elif n <= 500000:
        return 4500   # 75 min
    else:
        return 7200   #  2 hr (covers 1M, 1.18M, and anything larger)

from src import (
    FaissFlatL2Index, FaissFlatIPIndex, FaissFlatHNSWIndex,
    FaissScalarQuantizerL2Index, FaissScalarQuantizerIPIndex,
    MilvusIndex, MeilisearchIndex,
    LanceDBFlatIndex, LanceDBIVFIndex,
    WeaviateIndex, ChromaIndex,  # QdrantIndex commented out
    RedisIndex, ElasticsearchIndex,
    AnnoyIndex, PgvectorIndex, HNSWLibIndex,
)
from plot_results import plot_results

def load_dataset(path):
    with h5py.File(path, "r") as f:
        data      = f["train"][:].astype("float32")
        queries   = f["test"][:].astype("float32")
        neighbors = f["neighbors"][:].astype("int64") if "neighbors" in f else None
    return data, queries, neighbors

def run_benchmark(index, data, queries, k=10, latency_queries=100):
    index.train(data)

    t0 = time.perf_counter()
    index.add(data)
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    _, I = index.search(queries, k)
    search_time = time.perf_counter() - t1
    #TODO: Divide it by number of queries to get time per query? Remember 

    # Single-query latency: run each query individually, take median in ms
    latencies = []
    for q in queries[:latency_queries]:
        t_start = time.perf_counter()
        index.search(q[np.newaxis], k)
        latencies.append((time.perf_counter() - t_start) * 1000)
    latency_ms = float(np.median(latencies))

    if hasattr(index, "close"):
        index.close()

    return build_time, search_time, I, latency_ms

# Returns the largest m in [8,4,2,1] that evenly divides d, required by Milvus IVF_PQ
def good_pq_m(d):
    for m in [8, 4, 2, 1]:
        if d % m == 0:
            return m
    return 1

def compute_recall(I, neighbors, k):
    recall = 0
    for i in range(len(I)):
        true_set = set(int(x) for x in neighbors[i][:k])
        pred_set = set(int(x) for x in I[i][:k])
        recall += len(true_set & pred_set) / k
    return recall / len(I)

def get_family(name):
    prefix = name.split("_")[0]
    families = {
        "Faiss":    "FAISS",
        "Milvus":   "Milvus",
        "LanceDB":  "LanceDB",
        "Qdrant":   "Qdrant",
        "Weaviate": "Weaviate",
        "Chroma":   "Chroma",
        "Annoy":    "Annoy",
        "HNSWLib":  "HNSWLib",
        "Redis":    "Redis",
        "ES":       "Elasticsearch",
        "PgVector": "PgVector",
        "Meili":    "Meilisearch",
    }
    return families.get(prefix, prefix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to folder containing HDF5 dataset files")
    parser.add_argument("--lancedb-dir", default=os.path.join(os.path.expanduser("~"), "lancedb_cache"), help="Path to folder for LanceDB on-disk storage")
    args = parser.parse_args()
    data_dir = args.data_dir
    lancedb_dir = args.lancedb_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    datasets = [
        (os.path.join(data_dir, "sift-128-euclidean.hdf5"),         "sift-128-euclidean",         "l2",     [10000, 50000, 100000, 300000, 500000, 1000000]),
        (os.path.join(data_dir, "glove-100-angular.hdf5"),           "glove-100-angular",          "angular", [10000, 50000, 100000, 300000, 500000, 1183514]),
        (os.path.join(data_dir, "fashion-mnist-784-euclidean.hdf5"), "fashion-mnist-784-euclidean","l2",     [5000, 10000, 30000, 60000]),
    ]

    index_factories = [ #using factories to make loop simpler, factory is just assiging similar but different stuff to a function so that we call it easier later

    # --- EMBEDDED | IN-MEMORY | BATCHED ---
    # FAISS (HNSW is L2-only, no angular variant available in the current class)
    ("Faiss_Flat_L2",     lambda d: FaissFlatL2Index(d)),
    ("Faiss_Flat_Angular", lambda d: FaissFlatIPIndex(d)),
    ("Faiss_HNSW_L2",     lambda d: FaissFlatHNSWIndex(d)),
    ("Faiss_SQ_L2",       lambda d: FaissScalarQuantizerL2Index(d)),
    ("Faiss_SQ_Angular",   lambda d: FaissScalarQuantizerIPIndex(d)),
    # Qdrant (in-memory mode via :memory:) — commented out; too slow >20k, consider server-client version
    # ("Qdrant_L2",         lambda d: QdrantIndex(d, metric_type="l2")),
    # ("Qdrant_Angular",     lambda d: QdrantIndex(d, metric_type="cosine")),
    # Chroma (in-memory client)
    ("Chroma_L2",         lambda d: ChromaIndex(d, metric_type="l2")),
    ("Chroma_Angular",     lambda d: ChromaIndex(d, metric_type="cosine")),
    # Annoy
    ("Annoy_L2",          lambda d: AnnoyIndex(d, metric_type="l2")),
    ("Annoy_Angular",      lambda d: AnnoyIndex(d, metric_type="angular")),
    # HNSWLib
    ("HNSWLib_L2",        lambda d: HNSWLibIndex(d, metric_type="l2")),
    ("HNSWLib_Angular",    lambda d: HNSWLibIndex(d, metric_type="cosine")),

    # --- EMBEDDED | IN-MEMORY | BATCHED (spins up a local embedded server process) ---
    # Weaviate
    ("Weaviate_L2",       lambda d: WeaviateIndex(d, metric_type="l2")),
    ("Weaviate_Angular",   lambda d: WeaviateIndex(d, metric_type="cosine")),

    # --- EMBEDDED | ON-DISK | BATCHED ---
    # LanceDB
    ("LanceDB_FLAT_L2",     lambda d: LanceDBFlatIndex(d, metric_type="l2",     db_path=os.path.join(lancedb_dir, "lancedb_flat_bench"))),
    ("LanceDB_FLAT_Angular", lambda d: LanceDBFlatIndex(d, metric_type="cosine", db_path=os.path.join(lancedb_dir, "lancedb_flat_bench"))),
    ("LanceDB_IVF_L2",      lambda d: LanceDBIVFIndex(d, metric_type="l2",      db_path=os.path.join(lancedb_dir, "lancedb_ivf_bench"))),
    ("LanceDB_IVF_Angular",  lambda d: LanceDBIVFIndex(d, metric_type="cosine",  db_path=os.path.join(lancedb_dir, "lancedb_ivf_bench"))),

    # --- SERVER-CLIENT | IN-MEMORY | BATCHED --- (requires Docker)
    ("Milvus_FLAT_L2",         lambda d: MilvusIndex(d, metric_type="L2", index_type="FLAT")),
    ("Milvus_FLAT_Angular",     lambda d: MilvusIndex(d, metric_type="IP", index_type="FLAT")),
    ("Milvus_HNSW_L2",         lambda d: MilvusIndex(d, metric_type="L2", index_type="HNSW")),
    ("Milvus_HNSW_Angular",     lambda d: MilvusIndex(d, metric_type="IP", index_type="HNSW")),
    ("Milvus_IVF_FLAT_L2",     lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_FLAT")),
    ("Milvus_IVF_FLAT_Angular", lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_FLAT")),
    ("Milvus_IVF_SQ8_L2",      lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_SQ8")),
    ("Milvus_IVF_SQ8_Angular",  lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_SQ8")),
    ("Milvus_IVF_PQ_L2",       lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_PQ", index_params={"nlist": 128, "m": good_pq_m(d), "nbits": 8})),
    ("Milvus_IVF_PQ_Angular",   lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_PQ", index_params={"nlist": 128, "m": good_pq_m(d), "nbits": 8})),

    # --- SERVER-CLIENT | IN-MEMORY | PER-QUERY --- (requires Docker)
    ("Redis_L2",               lambda d: RedisIndex(d, metric_type="l2")),
    ("Redis_Angular",           lambda d: RedisIndex(d, metric_type="cosine")),

    # --- SERVER-CLIENT | ON-DISK | PER-QUERY --- (requires Docker)
    ("Meili_L2",               lambda d: MeilisearchIndex(d)),
    ("Meili_Angular",           lambda d: MeilisearchIndex(d)),
    ("ES_L2",                  lambda d: ElasticsearchIndex(d, metric_type="l2")),
    ("ES_Angular",              lambda d: ElasticsearchIndex(d, metric_type="cosine")),
    ("PgVector_L2",            lambda d: PgvectorIndex(d, metric_type="l2")),
    ("PgVector_Angular",        lambda d: PgvectorIndex(d, metric_type="cosine")),
    ]

    for ds_path, ds_name, ds_metric, sizes in datasets:
        print(f"\n=== Dataset: {ds_name} ({ds_metric}) ===")
        all_data, all_queries, neighbors = load_dataset(ds_path)
        d = all_data.shape[1]

        queries   = all_queries[:1000]
        neighbors = neighbors[:1000] if neighbors is not None else None
        nq        = len(queries)

        suffix = "_L2" if ds_metric == "l2" else "_Angular"
        active_factories = [(name, make) for name, make in index_factories if name.endswith(suffix)]

        results = {name: {"build": [], "search": [], "recall": [], "latency_ms": []} for name, _ in active_factories}

        for n in sizes:
            data = all_data[:n]
            print("Data Size =", n)

            timeout = get_timeout(n)

            for IndexTypeName, make_index in active_factories:
                def _run(mk=make_index):
                    idx = mk(d)
                    return run_benchmark(idx, data, queries, k=10)

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_run)
                        bt, st, I, lat = future.result(timeout=timeout)
                    recall = compute_recall(I, neighbors, k=10) if neighbors is not None else None
                    print(IndexTypeName, bt, st, f"recall={recall:.3f}" if recall is not None else "", f"latency={lat:.3f}ms")
                    results[IndexTypeName]["build"].append(bt)
                    results[IndexTypeName]["search"].append(st)
                    results[IndexTypeName]["recall"].append(recall)
                    results[IndexTypeName]["latency_ms"].append(lat)
                except FuturesTimeoutError:
                    print(f"TIMEOUT — {IndexTypeName} at n={n} (>{timeout}s)")
                    results[IndexTypeName]["build"].append(None)
                    results[IndexTypeName]["search"].append(None)
                    results[IndexTypeName]["recall"].append(None)
                    results[IndexTypeName]["latency_ms"].append(None)
                except Exception as e:
                    print(f"ERROR — {IndexTypeName} at n={n}: {e}")
                    results[IndexTypeName]["build"].append(None)
                    results[IndexTypeName]["search"].append(None)
                    results[IndexTypeName]["recall"].append(None)
                    results[IndexTypeName]["latency_ms"].append(None)

        size_labels = ["1M" if n == 1000000 else "1.18M" if n == 1183514 else f"{n//1000}k" if n >= 1000 else str(n) for n in sizes]

        out_dir = os.path.join("results", timestamp, ds_name)
        os.makedirs(out_dir, exist_ok=True)

        plot_results(results, active_factories, sizes, size_labels, ds_name, nq, out_dir)

if __name__ == "__main__": # the __name__ is ___main__ here only within this file, so this makes this only runnable directly, not when imported, prevents accidential execution?
    main()