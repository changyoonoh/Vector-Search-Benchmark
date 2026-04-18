import os
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results, active_factories, sizes, size_labels, ds_name, nq, out_dir):
    # ── Category definitions ──────────────────────────────────────────────
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

    # ── Recall bar chart — all indexes ────────────────────────────────────
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

    # ── Recall bar chart — grouped by category ────────────────────────────
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

    def get_vals(name, metric_key):
        raw = results[name][metric_key]
        if metric_key == "search":
            return [(v / nq * 1000 if v is not None else np.nan) for v in raw]
        return [(v if v is not None else np.nan) for v in raw]

    for metric_key, ylabel, title_prefix in [
        ("search",     "Search time per query (ms)", "Search Time per Query"),
        ("build",      "Build time (s)",              "Build Time"),
        ("latency_ms", "Latency per query (ms)",      "Single-Query Latency"),
    ]:
        # All indexes together
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"{title_prefix} — {ds_name} ({nq} queries)")
        for name, _ in active_factories:
            vals = get_vals(name, metric_key)
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
                    vals = get_vals(name, metric_key)
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
                vals = get_vals(name, metric_key)
                if any(not np.isnan(v) for v in vals):
                    ax.plot(sizes, vals, label=name, marker=marker_map[name], color=color_map[name])
            _setup_time_ax(ax, ylabel, sizes, size_labels)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
            plt.tight_layout(rect=[0, 0, 0.82, 0.95])
            plt.savefig(os.path.join(out_dir, f"{metric_key}_{cat_key}.png"), dpi=150, bbox_inches="tight")
            plt.close()