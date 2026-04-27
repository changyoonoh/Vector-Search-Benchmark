import os
import time
import argparse
import itertools
from datetime import datetime

import numpy as np
import h5py
import yaml

from src import (
    FaissFlatL2Index, FaissFlatIPIndex, FaissFlatHNSWIndex,
    FaissScalarQuantizerL2Index, FaissScalarQuantizerIPIndex,
    MilvusIndex, MeilisearchIndex,
    LanceDBFlatIndex, LanceDBIVFIndex,
    WeaviateIndex, ChromaIndex,
    RedisIndex, ElasticsearchIndex,
    AnnoyIndex, PgvectorIndex, HNSWLibIndex,
)
from src.faiss_ivf_l2 import FaissIVFFlatL2Index, FaissIVFSQ8L2Index
from plot_speed_recall import plot_speed_recall


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(path):
    with h5py.File(path, "r") as f:
        data      = f["train"][:].astype("float32")
        queries   = f["test"][:].astype("float32")
        neighbors = f["neighbors"][:].astype("int64") if "neighbors" in f else None
    return data, queries, neighbors


def compute_recall(I, neighbors, k):
    recall = 0
    for i in range(len(I)):
        true_set = set(int(x) for x in neighbors[i][:k])
        pred_set = set(int(x) for x in I[i][:k])
        recall += len(true_set & pred_set) / k
    return recall / len(I)


def good_pq_m(d):
    for m in [8, 4, 2, 1]:
        if d % m == 0:
            return m
    return 1


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_query_args(query_args_dict):
    """Yield one param-dict per point in the sweep (cartesian product of all lists)."""
    keys = list(query_args_dict.keys())
    for combo in itertools.product(*[query_args_dict[k] for k in keys]):
        yield dict(zip(keys, combo))


# ── Index sweep/single-run ────────────────────────────────────────────────────

def run_sweep(make_index, data, queries, neighbors, query_args_list, k=10):
    """Build once, then sweep query_args. Returns (build_time, list of point dicts)."""
    idx = make_index()
    idx.train(data)
    t0 = time.perf_counter()
    idx.add(data)
    build_time = time.perf_counter() - t0

    points = []
    for qa in query_args_list:
        if hasattr(idx, "set_query_params"):
            idx.set_query_params(qa)
        t = time.perf_counter()
        _, I = idx.search(queries, k)
        elapsed = time.perf_counter() - t
        recall = compute_recall(I, neighbors, k) if neighbors is not None else None
        points.append({"recall": recall, "qps": len(queries) / elapsed, **qa})

    if hasattr(idx, "close"):
        idx.close()
    return build_time, points


def run_single(make_index, data, queries, neighbors, k=10):
    """Build and run once. Returns (recall, qps)."""
    idx = make_index()
    idx.train(data)
    idx.add(data)
    t = time.perf_counter()
    _, I = idx.search(queries, k)
    elapsed = time.perf_counter() - t
    recall = compute_recall(I, neighbors, k) if neighbors is not None else None
    qps = len(queries) / elapsed
    if hasattr(idx, "close"):
        idx.close()
    return recall, qps


# ── Index specifications ──────────────────────────────────────────────────────

def make_tunable_specs(d, lancedb_dir):
    """
    Returns a list of tunable index specs. Each spec:
      config_file : path to YAML
      variants    : list of (name, metric_filter, factory_fn)
                    metric_filter: "L2" | "Angular"
      category    : category key for plot grouping
    """
    return [
        {
            "config_file": "src/configs/annoy.yml",
            "category": "Embedded_In_Memory",
            "variants": [
                ("Annoy_L2",
                 "L2",
                 lambda d=d, **kw: AnnoyIndex(d, metric_type="l2", **kw)),
                ("Annoy_Angular",
                 "Angular",
                 lambda d=d, **kw: AnnoyIndex(d, metric_type="angular", **kw)),
            ],
        },
        {
            "config_file": "src/configs/hnswlib.yml",
            "category": "Embedded_In_Memory",
            "variants": [
                ("HNSWLib_L2",
                 "L2",
                 lambda d=d, **kw: HNSWLibIndex(d, metric_type="l2", **kw)),
                ("HNSWLib_Angular",
                 "Angular",
                 lambda d=d, **kw: HNSWLibIndex(d, metric_type="cosine", **kw)),
            ],
        },
        {
            "config_file": "src/configs/faiss_hnsw.yml",
            "category": "Embedded_In_Memory",
            "variants": [
                ("Faiss_HNSW_L2",
                 "L2",
                 lambda d=d, **kw: FaissFlatHNSWIndex(d, **kw)),
            ],
        },
        {
            "config_file": "src/configs/faiss_ivf.yml",
            "category": "Embedded_In_Memory",
            "variants": [
                ("Faiss_IVF_FLAT_L2",
                 "L2",
                 lambda d=d, **kw: FaissIVFFlatL2Index(d, **kw)),
                ("Faiss_IVF_SQ8_L2",
                 "L2",
                 lambda d=d, **kw: FaissIVFSQ8L2Index(d, **kw)),
            ],
        },
        {
            "config_file": "src/configs/milvus_hnsw.yml",
            "category": "Server_Batched",
            "variants": [
                ("Milvus_HNSW_L2",
                 "L2",
                 lambda d=d, M=16, efConstruction=200, **kw: MilvusIndex(
                     d, metric_type="L2", index_type="HNSW",
                     index_params={"M": M, "efConstruction": efConstruction})),
                ("Milvus_HNSW_Angular",
                 "Angular",
                 lambda d=d, M=16, efConstruction=200, **kw: MilvusIndex(
                     d, metric_type="IP", index_type="HNSW",
                     index_params={"M": M, "efConstruction": efConstruction})),
            ],
        },
        {
            "config_file": "src/configs/milvus_ivf.yml",
            "category": "Server_Batched",
            "variants": [
                ("Milvus_IVF_FLAT_L2",
                 "L2",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="L2", index_type="IVF_FLAT",
                     index_params={"nlist": nlist})),
                ("Milvus_IVF_FLAT_Angular",
                 "Angular",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="IP", index_type="IVF_FLAT",
                     index_params={"nlist": nlist})),
                ("Milvus_IVF_SQ8_L2",
                 "L2",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="L2", index_type="IVF_SQ8",
                     index_params={"nlist": nlist})),
                ("Milvus_IVF_SQ8_Angular",
                 "Angular",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="IP", index_type="IVF_SQ8",
                     index_params={"nlist": nlist})),
                ("Milvus_IVF_PQ_L2",
                 "L2",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="L2", index_type="IVF_PQ",
                     index_params={"nlist": nlist, "m": good_pq_m(d), "nbits": 8})),
                ("Milvus_IVF_PQ_Angular",
                 "Angular",
                 lambda d=d, nlist=128, **kw: MilvusIndex(
                     d, metric_type="IP", index_type="IVF_PQ",
                     index_params={"nlist": nlist, "m": good_pq_m(d), "nbits": 8})),
            ],
        },
        {
            "config_file": "src/configs/pgvector.yml",
            "category": "Server_Per_Query",
            "variants": [
                ("PgVector_HNSW_L2",
                 "L2",
                 lambda d=d, m=16, ef_construction=200, **kw: PgvectorIndex(
                     d, metric_type="l2", index_type="hnsw",
                     m=m, ef_construction=ef_construction)),
                ("PgVector_HNSW_Angular",
                 "Angular",
                 lambda d=d, m=16, ef_construction=200, **kw: PgvectorIndex(
                     d, metric_type="cosine", index_type="hnsw",
                     m=m, ef_construction=ef_construction)),
            ],
        },
    ]


def make_non_tunable_specs(d, lancedb_dir):
    """
    Returns list of (name, metric_filter, factory_fn) for non-tunable indexes.
    """
    ldb_flat_path = os.path.join(lancedb_dir, "lancedb_flat_bench")
    ldb_ivf_path  = os.path.join(lancedb_dir, "lancedb_ivf_bench")

    return [
        # FAISS flat / SQ (embedded, in-memory)
        ("Faiss_Flat_L2",      "L2",      lambda d=d: FaissFlatL2Index(d)),
        ("Faiss_Flat_Angular", "Angular", lambda d=d: FaissFlatIPIndex(d)),
        ("Faiss_SQ_L2",        "L2",      lambda d=d: FaissScalarQuantizerL2Index(d)),
        ("Faiss_SQ_Angular",   "Angular", lambda d=d: FaissScalarQuantizerIPIndex(d)),
        # Chroma (embedded, in-memory)
        ("Chroma_L2",          "L2",      lambda d=d: ChromaIndex(d, metric_type="l2")),
        ("Chroma_Angular",     "Angular", lambda d=d: ChromaIndex(d, metric_type="cosine")),
        # Weaviate (embedded, in-memory server process)
        ("Weaviate_L2",        "L2",      lambda d=d: WeaviateIndex(d, metric_type="l2")),
        ("Weaviate_Angular",   "Angular", lambda d=d: WeaviateIndex(d, metric_type="cosine")),
        # LanceDB (embedded, on-disk)
        ("LanceDB_FLAT_L2",      "L2",      lambda d=d: LanceDBFlatIndex(d, metric_type="l2",     db_path=ldb_flat_path)),
        ("LanceDB_FLAT_Angular", "Angular", lambda d=d: LanceDBFlatIndex(d, metric_type="cosine", db_path=ldb_flat_path)),
        ("LanceDB_IVF_L2",       "L2",      lambda d=d: LanceDBIVFIndex(d, metric_type="l2",      db_path=ldb_ivf_path)),
        ("LanceDB_IVF_Angular",  "Angular", lambda d=d: LanceDBIVFIndex(d, metric_type="cosine",  db_path=ldb_ivf_path)),
        # Milvus Flat (server, batched) — brute force, no tuning
        ("Milvus_FLAT_L2",      "L2",      lambda d=d: MilvusIndex(d, metric_type="L2", index_type="FLAT")),
        ("Milvus_FLAT_Angular", "Angular", lambda d=d: MilvusIndex(d, metric_type="IP", index_type="FLAT")),
        # Redis (server, per-query)
        ("Redis_L2",            "L2",      lambda d=d: RedisIndex(d, metric_type="l2")),
        ("Redis_Angular",       "Angular", lambda d=d: RedisIndex(d, metric_type="cosine")),
        # Meilisearch (server, on-disk)
        ("Meili_L2",            "L2",      lambda d=d: MeilisearchIndex(d)),
        ("Meili_Angular",       "Angular", lambda d=d: MeilisearchIndex(d)),
        # Elasticsearch (server, on-disk)
        ("ES_L2",               "L2",      lambda d=d: ElasticsearchIndex(d, metric_type="l2")),
        ("ES_Angular",          "Angular", lambda d=d: ElasticsearchIndex(d, metric_type="cosine")),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True,
                        help="Path to folder containing HDF5 dataset files")
    parser.add_argument("--lancedb-dir", default=os.path.join(os.path.expanduser("~"), "lancedb_cache"),
                        help="Path to folder for LanceDB on-disk storage")
    args = parser.parse_args()

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir   = args.data_dir
    lancedb_dir = args.lancedb_dir

    datasets = [
        (os.path.join(data_dir, "sift-128-euclidean.hdf5"),
         "sift-128-euclidean", "L2", 1000000),
        (os.path.join(data_dir, "glove-100-angular.hdf5"),
         "glove-100-angular", "Angular", 1183514),
        (os.path.join(data_dir, "fashion-mnist-784-euclidean.hdf5"),
         "fashion-mnist-784-euclidean", "L2", 60000),
    ]

    for ds_path, ds_name, ds_metric, full_n in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_metric}) — full size n={full_n:,}")
        print(f"{'='*60}")

        all_data, all_queries, neighbors = load_dataset(ds_path)
        d         = all_data.shape[1]
        data      = all_data[:full_n]
        queries   = all_queries[:1000]
        neighbors = neighbors[:1000] if neighbors is not None else None
        k         = 10

        out_dir = os.path.join("results", timestamp, "speed_recall", ds_name)
        os.makedirs(out_dir, exist_ok=True)

        tunable_results     = {}   # name -> list of {recall, qps, ...params}
        non_tunable_results = {}   # name -> (recall, qps)

        # ── Tunable indexes ───────────────────────────────────────────────
        for spec in make_tunable_specs(d, lancedb_dir):
            cfg       = load_config(spec["config_file"])
            build_kw  = cfg.get("args", {})
            qa_dict   = cfg.get("query_args", {})
            qa_list   = list(expand_query_args(qa_dict))

            for name, metric_filter, factory in spec["variants"]:
                if metric_filter != ds_metric:
                    continue
                print(f"\n[TUNABLE] {name}  build_args={build_kw}  ({len(qa_list)} sweep points)")
                try:
                    make_fn = lambda f=factory, kw=build_kw: f(**kw)
                    bt, points = run_sweep(make_fn, data, queries, neighbors, qa_list, k=k)
                    tunable_results[name] = points
                    print(f"  built in {bt:.1f}s")
                    for p in points:
                        param_str = "  ".join(f"{k}={v}" for k, v in p.items()
                                              if k not in ("recall", "qps"))
                        print(f"  {param_str}  recall={p['recall']:.3f}  qps={p['qps']:.1f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    tunable_results[name] = []

        # ── Non-tunable indexes ───────────────────────────────────────────
        for name, metric_filter, factory in make_non_tunable_specs(d, lancedb_dir):
            if metric_filter != ds_metric:
                continue
            print(f"\n[SINGLE]  {name}")
            try:
                recall, qps = run_single(factory, data, queries, neighbors, k=k)
                non_tunable_results[name] = (recall, qps)
                print(f"  recall={recall:.3f}  qps={qps:.1f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                non_tunable_results[name] = (None, None)

        # ── Plot ──────────────────────────────────────────────────────────
        plot_speed_recall(tunable_results, non_tunable_results, ds_name, out_dir)
        print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
