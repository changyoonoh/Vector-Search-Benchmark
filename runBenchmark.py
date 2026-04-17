import os
import time
import argparse
from datetime import datetime
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
    # Qdrant (in-memory mode via :memory:)
    ("Qdrant_L2",         lambda d: QdrantIndex(d, metric_type="l2")),
    ("Qdrant_Angular",     lambda d: QdrantIndex(d, metric_type="cosine")),
    # Chroma (in-memory client)
    ("Chroma_L2",         lambda d: ChromaIndex(d, metric_type="l2")),
    ("Chroma_Angular",     lambda d: ChromaIndex(d, metric_type="cosine")),
    # Annoy
    ("Annoy_L2",          lambda d: AnnoyVecIndex(d, metric_type="l2")),
    ("Annoy_Angular",      lambda d: AnnoyVecIndex(d, metric_type="angular")),
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

            for IndexTypeName, make_index in active_factories:
                try:
                    index = make_index(d)
                    bt, st, I, lat = run_benchmark(index, data, queries, k=10)
                    recall = compute_recall(I, neighbors, k=10) if neighbors is not None else None
                    print(IndexTypeName, bt, st, f"recall={recall:.3f}" if recall is not None else "", f"latency={lat:.3f}ms")
                    results[IndexTypeName]["build"].append(bt)
                    results[IndexTypeName]["search"].append(st)
                    results[IndexTypeName]["recall"].append(recall)
                    results[IndexTypeName]["latency_ms"].append(lat)
                except Exception as e:
                    print(f"ERROR — {IndexTypeName} at n={n}: {e}")
                    results[IndexTypeName]["build"].append(None)
                    results[IndexTypeName]["search"].append(None)
                    results[IndexTypeName]["recall"].append(None)
                    results[IndexTypeName]["latency_ms"].append(None)

        size_labels = ["1M" if n == 1000000 else "1.18M" if n == 1183514 else f"{n//1000}k" if n >= 1000 else str(n) for n in sizes]

        out_dir = os.path.join("results", timestamp, ds_name)
        os.makedirs(out_dir, exist_ok=True)

        # ── Category definitions ──────────────────────────────────────────────
        # label, bg_color (hex, will append "33" for ~20% alpha), index name prefixes
        categories = {
            "Embedded_In_Memory": ("Embedded In-Memory", "#d4e6f1", ["Faiss", "Qdrant", "Chroma", "Annoy", "HNSWLib", "Weaviate"]),
            "Embedded_On_Disk":   ("Embedded On-Disk",   "#d5f5e3", ["LanceDB"]),
            "Server_Batched":     ("Server Batched",      "#fde8d8", ["Milvus"]),
            "Server_Per_Query":   ("Server Per-Query",    "#f9ebea", ["Redis", "ES", "PgVector", "Meili"]),
        }

        def get_category_key(name):
            prefix = name.split("_")[0]
            for cat_key, (_, _, prefixes) in categories.items():
                if prefix in prefixes:
                    return cat_key
            return None

        cat_indexes = {
            cat_key: [n for n, _ in active_factories if get_category_key(n) == cat_key]
            for cat_key in categories
        }
        cat_indexes = {k: v for k, v in cat_indexes.items() if v}

        # ── Color and marker maps ─────────────────────────────────────────────
        n_active = len(active_factories)
        cmap = plt.colormaps["tab10"] if n_active <= 10 else plt.colormaps["tab20"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x", "p", "H", "<", ">", "1", "2", "3", "4", "8"]
        color_map  = {name: cmap(i / max(n_active - 1, 1)) for i, (name, _) in enumerate(active_factories)}
        marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(active_factories)}

        # ── Recall bar chart — all indexes ───────────────────────────────────
        recall_names  = [n for n, _ in active_factories if results[n]["recall"][-1] is not None]
        recall_values = [results[n]["recall"][-1] for n in recall_names]
        recall_colors = [color_map[n] for n in recall_names]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(f"Recall@10 at n={sizes[-1]:,} — {ds_name}")
        bars = ax.bar(recall_names, recall_values, color=recall_colors)
        for bar, val in zip(bars, recall_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, "recall.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ── Recall bar chart — grouped by category ───────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"Recall@10 at n={sizes[-1]:,} — {ds_name} (by category)")
        x = 0
        xtick_positions, xtick_labels = [], []
        for cat_key, idx_names in cat_indexes.items():
            cat_label, cat_color, _ = categories[cat_key]
            valid = [n for n in idx_names if results[n]["recall"][-1] is not None]
            if not valid:
                continue
            x_start = x - 0.5
            for name in valid:
                val = results[name]["recall"][-1]
                ax.bar(x, val, color=color_map[name])
                ax.text(x, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=6)
                xtick_positions.append(x)
                xtick_labels.append(name)
                x += 1
            x_end = x - 0.5
            ax.axvspan(x_start, x_end, alpha=0.15, color=cat_color)
            ax.text((x_start + x_end) / 2, 1.08, cat_label, ha="center", va="bottom", fontsize=8, style="italic")
            x += 0.5
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, "recall_grouped.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ── Speed-recall helpers ──────────────────────────────────────────────
        size_cmap = plt.colormaps["plasma"]
        size_color = {n: size_cmap(i / max(len(sizes) - 1, 1)) for i, n in enumerate(sizes)}
        all_factory_names = [n for n, _ in active_factories]

        def _plot_speed_recall(ax, factory_names):
            for size_idx, n in enumerate(sizes):
                for name in factory_names:
                    recall = results[name]["recall"][size_idx]
                    search_time = results[name]["search"][size_idx]
                    if recall is None or search_time is None or search_time == 0:
                        continue
                    qps = nq / search_time
                    ax.scatter(recall, qps, color=size_color[n], marker=marker_map[name], s=80, zorder=3)
                    ax.annotate(name, (recall, qps), fontsize=5, textcoords="offset points", xytext=(4, 2))
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("Queries per second")
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

        def _add_speed_recall_legends(ax, factory_names):
            size_handles = [ax.scatter([], [], color=size_color[n], label=size_labels[sizes.index(n)], s=60) for n in sizes]
            size_legend = ax.legend(handles=size_handles, title="Dataset size", fontsize=7, title_fontsize=8, loc="upper left")
            ax.add_artist(size_legend)
            marker_handles = [ax.scatter([], [], color="gray", marker=marker_map[name], s=60, label=name) for name in factory_names]
            ax.legend(handles=marker_handles, title="Index", fontsize=6, title_fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

        # ── Speed-recall — all indexes ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.suptitle(f"Speed-Recall Tradeoff — {ds_name}")
        _plot_speed_recall(ax, all_factory_names)
        _add_speed_recall_legends(ax, all_factory_names)
        plt.tight_layout(rect=[0, 0, 0.82, 0.95])
        plt.savefig(os.path.join(out_dir, "speed_recall.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ── Speed-recall — per category ───────────────────────────────────────
        for cat_key, idx_names in cat_indexes.items():
            cat_label, _, _ = categories[cat_key]
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f"Speed-Recall Tradeoff — {ds_name} — {cat_label}")
            _plot_speed_recall(ax, idx_names)
            _add_speed_recall_legends(ax, idx_names)
            plt.tight_layout(rect=[0, 0, 0.82, 0.95])
            plt.savefig(os.path.join(out_dir, f"speed_recall_{cat_key}.png"), dpi=150, bbox_inches="tight")
            plt.close()

        # ── Time-based plots ──────────────────────────────────────────────────
        n_cats = len(cat_indexes)

        def _setup_time_ax(ax, ylabel, sizes, size_labels):
            ax.set_xlabel("Number of Vectors")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels(size_labels, rotation=45)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.grid(True, alpha=0.3)

        for metric_key, ylabel, title_prefix in [
            ("search",     "Search time per query (ms)", "Search Time per Query"),
            ("build",      "Build time (s)",              "Build Time"),
            ("latency_ms", "Latency per query (ms)",      "Single-Query Latency"),
        ]:
            def get_vals(name, mk=metric_key):
                raw = results[name][mk]
                if mk == "search":
                    return [(v / nq * 1000 if v is not None else np.nan) for v in raw]
                return [(v if v is not None else np.nan) for v in raw]

            # All indexes together
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(f"{title_prefix} — {ds_name} ({nq} queries)")
            for name, _ in active_factories:
                vals = get_vals(name)
                if any(not np.isnan(v) for v in vals):
                    ax.plot(sizes, vals, label=name, marker=marker_map[name], color=color_map[name])
            _setup_time_ax(ax, ylabel, sizes, size_labels)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
            plt.tight_layout(rect=[0, 0, 0.82, 0.95])
            plt.savefig(os.path.join(out_dir, f"{metric_key}.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # Grouped — subplots per category on one figure
            if n_cats > 0:
                fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6), sharey=False)
                if n_cats == 1:
                    axes = [axes]
                fig.suptitle(f"{title_prefix} by Category — {ds_name} ({nq} queries)")
                for ax, (cat_key, idx_names) in zip(axes, cat_indexes.items()):
                    cat_label, cat_color, _ = categories[cat_key]
                    ax.set_facecolor(cat_color + "33")
                    for name in idx_names:
                        vals = get_vals(name)
                        if any(not np.isnan(v) for v in vals):
                            ax.plot(sizes, vals, label=name, marker=marker_map[name], color=color_map[name])
                    ax.set_title(cat_label, fontsize=9)
                    _setup_time_ax(ax, ylabel, sizes, size_labels)
                    ax.legend(fontsize=6)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(out_dir, f"{metric_key}_grouped.png"), dpi=150, bbox_inches="tight")
                plt.close()

            # Per category — one figure per category
            for cat_key, idx_names in cat_indexes.items():
                cat_label, cat_color, _ = categories[cat_key]
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.suptitle(f"{title_prefix} — {ds_name} — {cat_label} ({nq} queries)")
                ax.set_facecolor(cat_color + "33")
                for name in idx_names:
                    vals = get_vals(name)
                    if any(not np.isnan(v) for v in vals):
                        ax.plot(sizes, vals, label=name, marker=marker_map[name], color=color_map[name])
                _setup_time_ax(ax, ylabel, sizes, size_labels)
                ax.tick_params(axis="x", rotation=45)
                ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
                plt.tight_layout(rect=[0, 0, 0.82, 0.95])
                plt.savefig(os.path.join(out_dir, f"{metric_key}_{cat_key}.png"), dpi=150, bbox_inches="tight")
                plt.close()

if __name__ == "__main__": # the __name__ is ___main__ here only within this file, so this makes this only runnable directly, not when imported, prevents accidential execution?
    main()