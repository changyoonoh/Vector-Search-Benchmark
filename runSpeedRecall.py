import os
import time
import argparse
import itertools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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


# ── Timeout constants ─────────────────────────────────────────────────────────

# Per-sweep-point search timeout: 1000 queries at any reasonable ef/nprobe
# should never take longer than this.
SEARCH_TIMEOUT = 300  # 5 minutes

# Non-tunable indexes that are likely too slow to build at large dataset sizes.
# At 1M vectors these can take hours, making them poor single-dot candidates.
SLOW_NON_TUNABLE_PREFIXES = frozenset({"Chroma", "Weaviate", "Meili"})
SLOW_SKIP_THRESHOLD = 500_000


def get_timeout(n):
    """Size-based build timeout, mirroring runBenchmark.py."""
    if n <= 10_000:   return 180
    if n <= 50_000:   return 600
    if n <= 100_000:  return 1_200
    if n <= 300_000:  return 2_700
    if n <= 500_000:  return 4_500
    return 7_200


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


def expand_query_args(qa_dict):
    """Yield one param-dict per point in the sweep (cartesian product of all lists)."""
    keys = list(qa_dict.keys())
    for combo in itertools.product(*[qa_dict[k] for k in keys]):
        yield dict(zip(keys, combo))


def adapt_for_dataset(raw_build_kw, qa_dict, n):
    """
    Return (adapted_build_kw, qa_list) tuned for a dataset of size n.

    Build-time:
      nlist — capped at 64 for n <= 60k to avoid over-partitioning.

    Query-time:
      search_k — filtered to <= n // 2 (probing half the dataset is thorough;
                 the YAML cap of 100k handles the 1M case automatically).
      ef / efSearch / ef_search — filtered to <= 500 for n <= 60k; recall
                                  saturates near 1.0 well below that.
      nprobe — capped at the (possibly adapted) nlist to remove redundant points.
    """
    adapted_bkw = dict(raw_build_kw)
    if "nlist" in adapted_bkw and n <= 60000:
        adapted_bkw["nlist"] = min(adapted_bkw["nlist"], 64)

    adapted_qa = {}
    for key, values in qa_dict.items():
        if key == "search_k":
            cap = n // 2
            filtered = [v for v in values if v <= cap]
            adapted_qa[key] = filtered if filtered else [values[0]]
        elif key in ("ef", "efSearch", "ef_search") and n <= 60000:
            adapted_qa[key] = [v for v in values if v <= 500]
        elif key == "nprobe" and "nlist" in adapted_bkw:
            nlist = adapted_bkw["nlist"]
            adapted_qa[key] = [v for v in values if v <= nlist]
        else:
            adapted_qa[key] = values

    return adapted_bkw, list(expand_query_args(adapted_qa))


# ── Timed sweep / single-run ──────────────────────────────────────────────────

def run_sweep(make_index, data, queries, neighbors, query_args_list, k=10,
              build_timeout=3600, search_timeout=SEARCH_TIMEOUT):
    """
    Build once (with build_timeout), then sweep query_args.

    Each sweep point gets its own search_timeout so a slow high-ef point
    doesn't abort the whole curve — it's recorded as (None, None) instead.

    Raises FuturesTimeoutError if the build itself times out.
    Returns (build_time, points) where points is a list of dicts
    {recall, qps, <param>} — recall/qps are None for timed-out points.
    """
    def _build():
        idx = make_index()
        idx.train(data)
        t0 = time.perf_counter()
        idx.add(data)
        return idx, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_build)
        idx, build_time = fut.result(timeout=build_timeout)  # raises on timeout

    points = []
    for qa in query_args_list:
        if hasattr(idx, "set_query_params"):
            idx.set_query_params(qa)

        def _search(idx=idx):
            t = time.perf_counter()
            _, I = idx.search(queries, k)
            elapsed = time.perf_counter() - t
            recall = compute_recall(I, neighbors, k) if neighbors is not None else None
            return recall, len(queries) / elapsed

        recall, qps = None, None
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_search)
                recall, qps = fut.result(timeout=search_timeout)
        except FuturesTimeoutError:
            param_str = " ".join(f"{pk}={pv}" for pk, pv in qa.items())
            print(f"    TIMEOUT  {param_str}  (>{search_timeout}s)")
        except Exception as e:
            param_str = " ".join(f"{pk}={pv}" for pk, pv in qa.items())
            print(f"    ERROR    {param_str}  {e}")

        points.append({"recall": recall, "qps": qps, **qa})

    if hasattr(idx, "close"):
        idx.close()
    return build_time, points


def run_single(make_index, data, queries, neighbors, k=10, timeout=3600):
    """
    Build and search once, both wrapped in a single timeout.
    Raises FuturesTimeoutError on timeout.
    """
    def _run():
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

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        return fut.result(timeout=timeout)


# ── Index specifications ──────────────────────────────────────────────────────

def make_tunable_specs(d, lancedb_dir):
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
    ldb_flat_path = os.path.join(lancedb_dir, "lancedb_flat_bench")
    ldb_ivf_path  = os.path.join(lancedb_dir, "lancedb_ivf_bench")

    return [
        ("Faiss_Flat_L2",        "L2",      lambda d=d: FaissFlatL2Index(d)),
        ("Faiss_Flat_Angular",   "Angular", lambda d=d: FaissFlatIPIndex(d)),
        ("Faiss_SQ_L2",          "L2",      lambda d=d: FaissScalarQuantizerL2Index(d)),
        ("Faiss_SQ_Angular",     "Angular", lambda d=d: FaissScalarQuantizerIPIndex(d)),
        ("Chroma_L2",            "L2",      lambda d=d: ChromaIndex(d, metric_type="l2")),
        ("Chroma_Angular",       "Angular", lambda d=d: ChromaIndex(d, metric_type="cosine")),
        ("Weaviate_L2",          "L2",      lambda d=d: WeaviateIndex(d, metric_type="l2")),
        ("Weaviate_Angular",     "Angular", lambda d=d: WeaviateIndex(d, metric_type="cosine")),
        ("LanceDB_FLAT_L2",      "L2",      lambda d=d: LanceDBFlatIndex(d, metric_type="l2",     db_path=ldb_flat_path)),
        ("LanceDB_FLAT_Angular", "Angular", lambda d=d: LanceDBFlatIndex(d, metric_type="cosine", db_path=ldb_flat_path)),
        ("LanceDB_IVF_L2",       "L2",      lambda d=d: LanceDBIVFIndex(d, metric_type="l2",      db_path=ldb_ivf_path)),
        ("LanceDB_IVF_Angular",  "Angular", lambda d=d: LanceDBIVFIndex(d, metric_type="cosine",  db_path=ldb_ivf_path)),
        ("Milvus_FLAT_L2",       "L2",      lambda d=d: MilvusIndex(d, metric_type="L2", index_type="FLAT")),
        ("Milvus_FLAT_Angular",  "Angular", lambda d=d: MilvusIndex(d, metric_type="IP", index_type="FLAT")),
        ("Redis_L2",             "L2",      lambda d=d: RedisIndex(d, metric_type="l2")),
        ("Redis_Angular",        "Angular", lambda d=d: RedisIndex(d, metric_type="cosine")),
        ("Meili_L2",             "L2",      lambda d=d: MeilisearchIndex(d)),
        ("Meili_Angular",        "Angular", lambda d=d: MeilisearchIndex(d)),
        ("ES_L2",                "L2",      lambda d=d: ElasticsearchIndex(d, metric_type="l2")),
        ("ES_Angular",           "Angular", lambda d=d: ElasticsearchIndex(d, metric_type="cosine")),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True,
                        help="Path to folder containing HDF5 dataset files")
    parser.add_argument("--lancedb-dir", default=os.path.join(os.path.expanduser("~"), "lancedb_cache"),
                        help="Path to folder for LanceDB on-disk storage")
    parser.add_argument("--only", nargs="+", metavar="PREFIX",
                        help="Only run indexes whose name starts with these prefixes "
                             "(e.g. --only Faiss  or  --only Faiss Annoy HNSWLib)")
    args = parser.parse_args()

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data_dir    = args.data_dir
    lancedb_dir = args.lancedb_dir
    only        = [p.lower() for p in args.only] if args.only else None

    def _wanted(name):
        if only is None:
            return True
        return any(name.lower().startswith(p) for p in only)

    datasets = [
        (os.path.join(data_dir, "sift-128-euclidean.hdf5"),
         "sift-128-euclidean", "L2", 1_000_000),
        (os.path.join(data_dir, "glove-100-angular.hdf5"),
         "glove-100-angular", "Angular", 1_183_514),
        (os.path.join(data_dir, "fashion-mnist-784-euclidean.hdf5"),
         "fashion-mnist-784-euclidean", "L2", 60_000),
    ]

    for ds_path, ds_name, ds_metric, full_n in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_metric})  n={full_n:,}")
        print(f"{'='*60}")

        all_data, all_queries, neighbors = load_dataset(ds_path)
        d         = all_data.shape[1]
        data      = all_data[:full_n]
        queries   = all_queries[:1000]
        neighbors = neighbors[:1000] if neighbors is not None else None
        k         = 10
        build_timeout = get_timeout(len(data))

        out_dir = os.path.join("results", timestamp, "speed_recall", ds_name)
        os.makedirs(out_dir, exist_ok=True)

        tunable_results     = {}   # name -> list of {recall, qps, ...params}
        non_tunable_results = {}   # name -> (recall, qps)

        # ── Tunable indexes ───────────────────────────────────────────────
        for spec in make_tunable_specs(d, lancedb_dir):
            # Only process specs that have at least one variant for this metric
            matching = [(nm, mf, fn) for nm, mf, fn in spec["variants"]
                        if mf == ds_metric and _wanted(nm)]
            if not matching:
                continue

            cfg     = load_config(spec["config_file"])
            raw_bkw = cfg.get("args", {})
            qa_dict = cfg.get("query_args", {})
            build_kw, qa_list = adapt_for_dataset(raw_bkw, qa_dict, len(data))

            orig_qa_n = sum(1 for _ in expand_query_args(qa_dict))
            if build_kw != raw_bkw:
                print(f"  [adapt] build args: {raw_bkw} -> {build_kw}")
            if len(qa_list) < orig_qa_n:
                print(f"  [adapt] sweep: {orig_qa_n} -> {len(qa_list)} points")

            for name, _, factory in matching:
                make_fn = lambda f=factory, kw=build_kw: f(**kw)
                print(f"\n[TUNABLE] {name}  build={build_kw}  ({len(qa_list)} pts)")
                try:
                    bt, points = run_sweep(
                        make_fn, data, queries, neighbors, qa_list, k=k,
                        build_timeout=build_timeout, search_timeout=SEARCH_TIMEOUT,
                    )
                    tunable_results[name] = points
                    print(f"  built in {bt:.1f}s")
                    for p in points:
                        param_str = " ".join(f"{pk}={pv}" for pk, pv in p.items()
                                             if pk not in ("recall", "qps"))
                        if p["recall"] is None:
                            print(f"  {param_str}  —  TIMEOUT/ERROR")
                        else:
                            print(f"  {param_str}  recall={p['recall']:.3f}  qps={p['qps']:.1f}")
                except FuturesTimeoutError:
                    print(f"  BUILD TIMEOUT (>{build_timeout}s)")
                    tunable_results[name] = []
                except Exception as e:
                    print(f"  ERROR: {e}")
                    tunable_results[name] = []

        # ── Non-tunable indexes ───────────────────────────────────────────
        for name, metric_filter, factory in make_non_tunable_specs(d, lancedb_dir):
            if metric_filter != ds_metric or not _wanted(name):
                continue

            prefix = name.split("_")[0]
            if prefix in SLOW_NON_TUNABLE_PREFIXES and len(data) > SLOW_SKIP_THRESHOLD:
                print(f"\n[SKIP]    {name}  (n={len(data):,} > {SLOW_SKIP_THRESHOLD:,} threshold)")
                non_tunable_results[name] = (None, None)
                continue

            print(f"\n[SINGLE]  {name}")
            try:
                recall, qps = run_single(factory, data, queries, neighbors,
                                         k=k, timeout=build_timeout)
                non_tunable_results[name] = (recall, qps)
                print(f"  recall={recall:.3f}  qps={qps:.1f}")
            except FuturesTimeoutError:
                print(f"  BUILD TIMEOUT (>{build_timeout}s)")
                non_tunable_results[name] = (None, None)
            except Exception as e:
                print(f"  ERROR: {e}")
                non_tunable_results[name] = (None, None)

        # ── Plot ──────────────────────────────────────────────────────────
        plot_speed_recall(tunable_results, non_tunable_results, ds_name, out_dir)
        print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
