import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

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
from src.chroma_index import ChromaIndex
from src.redis_index import RedisIndex
from src.elasticsearch_index import ElasticsearchIndex
from src.annoy_index import AnnoyIndex as AnnoyVecIndex
from src.pgvector_index import PgvectorIndex
from src.hnswlib_index import HNSWLibIndex

def load_dataset(path):
    with h5py.File(path, "r") as f:
        data      = f["train"][:].astype("float32")
        queries   = f["test"][:].astype("float32")
        neighbors = f["neighbors"][:].astype("int64") if "neighbors" in f else None
    return data, queries, neighbors

def run_benchmark(index, data, queries, k=5):
    t0 = time.perf_counter()

    index.train(data)
    t1 = time.perf_counter()

    index.add(data)
    t2 = time.perf_counter()

    _, I = index.search(queries, k)
    t3 = time.perf_counter()

    train_time = t1 - t0
    build_time = t2 - t1
    search_time = t3 - t2
    return train_time, build_time, search_time, I

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
    datasets = [
        ("/Users/yoonoh/Desktop/CS 492/data/sift-128-euclidean.hdf5",         "sift-128-euclidean",         "l2",     [10000, 50000, 100000, 300000, 500000]),
        ("/Users/yoonoh/Desktop/CS 492/data/glove-100-angular.hdf5",           "glove-100-angular",          "cosine", [10000, 50000, 100000, 300000, 500000]),
        ("/Users/yoonoh/Desktop/CS 492/data/fashion-mnist-784-euclidean.hdf5", "fashion-mnist-784-euclidean","l2",     [1000,  5000,  10000,  30000,  50000]),
    ]

    index_factories = [ #using factories to make loop simpler, factory is just assiging similar but different stuff to a function so that we call it easier later

    # --- EMBEDDED | IN-MEMORY | BATCHED ---
    # FAISS (HNSW is L2-only, no cosine variant available in the current class)
    ("Faiss_Flat_L2",     lambda d: FaissFlatL2Index(d)),
    ("Faiss_Flat_Cosine", lambda d: FaissFlatIPIndex(d)),
    ("Faiss_HNSW_L2",     lambda d: FaissFlatHNSWIndex(d)),
    ("Faiss_SQ_L2",       lambda d: FaissScalarQuantizerL2Index(d)),
    ("Faiss_SQ_Cosine",   lambda d: FaissScalarQuantizerIPIndex(d)),
    # Qdrant (in-memory mode via :memory:)
    ("Qdrant_L2",         lambda d: QdrantIndex(d, metric_type="l2")),
    ("Qdrant_Cosine",     lambda d: QdrantIndex(d, metric_type="cosine")),
    # Chroma (in-memory client)
    ("Chroma_L2",         lambda d: ChromaIndex(d, metric_type="l2")),
    ("Chroma_Cosine",     lambda d: ChromaIndex(d, metric_type="cosine")),
    # Annoy
    ("Annoy_L2",          lambda d: AnnoyVecIndex(d, metric_type="l2")),
    ("Annoy_Cosine",      lambda d: AnnoyVecIndex(d, metric_type="angular")),
    # HNSWLib
    ("HNSWLib_L2",        lambda d: HNSWLibIndex(d, metric_type="l2")),
    ("HNSWLib_Cosine",    lambda d: HNSWLibIndex(d, metric_type="cosine")),

    # --- EMBEDDED | IN-MEMORY | BATCHED (spins up a local embedded server process) ---
    # Weaviate
    ("Weaviate_L2",       lambda d: WeaviateIndex(d, metric_type="l2")),
    ("Weaviate_Cosine",   lambda d: WeaviateIndex(d, metric_type="cosine")),

    # --- EMBEDDED | ON-DISK | BATCHED ---
    # LanceDB
    ("LanceDB_FLAT_L2",     lambda d: LanceDBFlatIndex(d, metric_type="l2")),
    ("LanceDB_FLAT_Cosine", lambda d: LanceDBFlatIndex(d, metric_type="cosine")),
    ("LanceDB_IVF_L2",      lambda d: LanceDBIVFIndex(d, metric_type="l2")),
    ("LanceDB_IVF_Cosine",  lambda d: LanceDBIVFIndex(d, metric_type="cosine")),

    # --- SERVER-CLIENT | IN-MEMORY | BATCHED --- (requires Docker, commented out for quick runs)
    # ("Milvus_FLAT_L2",         lambda d: MilvusIndex(d, metric_type="L2", index_type="FLAT")),
    # ("Milvus_FLAT_Cosine",     lambda d: MilvusIndex(d, metric_type="IP", index_type="FLAT")),
    # ("Milvus_HNSW_L2",         lambda d: MilvusIndex(d, metric_type="L2", index_type="HNSW")),
    # ("Milvus_HNSW_Cosine",     lambda d: MilvusIndex(d, metric_type="IP", index_type="HNSW")),
    # ("Milvus_IVF_FLAT_L2",     lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_FLAT")),
    # ("Milvus_IVF_FLAT_Cosine", lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_FLAT")),
    # ("Milvus_IVF_SQ8_L2",      lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_SQ8")),
    # ("Milvus_IVF_SQ8_Cosine",  lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_SQ8")),
    # ("Milvus_IVF_PQ_L2",       lambda d: MilvusIndex(d, metric_type="L2", index_type="IVF_PQ", index_params={"nlist": 128, "m": good_pq_m(d), "nbits": 8})),
    # ("Milvus_IVF_PQ_Cosine",   lambda d: MilvusIndex(d, metric_type="IP", index_type="IVF_PQ", index_params={"nlist": 128, "m": good_pq_m(d), "nbits": 8})),

    # --- SERVER-CLIENT | IN-MEMORY | PER-QUERY --- (requires Docker, commented out for quick runs)
    # ("Redis_L2",               lambda d: RedisIndex(d, metric_type="l2")),
    # ("Redis_Cosine",           lambda d: RedisIndex(d, metric_type="cosine")),

    # --- SERVER-CLIENT | ON-DISK | PER-QUERY --- (requires Docker, commented out for quick runs)
    # ("Meili_L2",               lambda d: MeilisearchIndex(d)),
    # ("Meili_Cosine",           lambda d: MeilisearchIndex(d)),
    # ("ES_L2",                  lambda d: ElasticsearchIndex(d, metric_type="l2")),
    # ("ES_Cosine",              lambda d: ElasticsearchIndex(d, metric_type="cosine")),
    # ("PgVector_L2",            lambda d: PgvectorIndex(d, metric_type="l2")),
    # ("PgVector_Cosine",        lambda d: PgvectorIndex(d, metric_type="cosine")),
    ]

    for ds_path, ds_name, ds_metric, sizes in datasets:
        print(f"\n=== Dataset: {ds_name} ({ds_metric}) ===")
        all_data, all_queries, neighbors = load_dataset(ds_path)
        d = all_data.shape[1]

        queries   = all_queries[:1000]
        neighbors = neighbors[:1000] if neighbors is not None else None
        nq        = len(queries)

        suffix = "_L2" if ds_metric == "l2" else "_Cosine"
        active_factories = [(name, make) for name, make in index_factories if name.endswith(suffix)]

        results = {name: {"train": [], "build": [], "search": [], "recall": []} for name, _ in active_factories}

        for n in sizes:
            data = all_data[:n]
            print("Data Size =", n)

            for IndexTypeName, make_index in active_factories:
                index = make_index(d)
                tt, bt, st, I = run_benchmark(index, data, queries, k=5)
                recall = compute_recall(I, neighbors, k=5) if neighbors is not None else None
                print(IndexTypeName, tt, bt, st, f"recall={recall:.3f}" if recall is not None else "")
                results[IndexTypeName]["train"].append(tt)
                results[IndexTypeName]["build"].append(bt)
                results[IndexTypeName]["search"].append(st)
                results[IndexTypeName]["recall"].append(recall)

        size_labels = [f"{n//1000}k" if n >= 1000 else str(n) for n in sizes]

        # Assign a distinct color to each index (tab20 supports up to 20)
        cmap = plt.colormaps["tab20"]
        color_map = {name: cmap(i / max(len(active_factories) - 1, 1)) for i, (name, _) in enumerate(active_factories)}

        # Recall — bar chart at the largest size only
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(f"Recall@5 at n={sizes[-1]:,} — {ds_name}")
        names  = [n for n, _ in active_factories if results[n]["recall"][-1] is not None]
        values = [results[n]["recall"][-1] for n in names]
        colors = [color_map[n] for n in names]
        ax.bar(names, values, color=colors)
        ax.set_ylabel("Recall@5")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Time-based plots — all indexes on one graph per metric
        for metric_key, ylabel, title_prefix in [
            ("search", "Search time (s)", "Search Time"),
            ("build",  "Build time (s)",  "Build Time"),
            ("train",  "Train time (s)",  "Train Time"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(f"{title_prefix} — {ds_name} ({nq} queries)")
            for name, _ in active_factories:
                vals = results[name][metric_key]
                if any(v is not None for v in vals):
                    ax.plot(sizes, vals, label=name, marker="o", color=color_map[name])
            ax.set_xlabel("Number of Vectors")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.set_xticks(sizes)
            ax.set_xticklabels(size_labels)
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.show()

if __name__ == "__main__": # the __name__ is ___main__ here only within this file, so this makes this only runnable directly, not when imported, prevents accidential execution?
    main()