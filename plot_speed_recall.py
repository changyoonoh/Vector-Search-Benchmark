import os
import matplotlib.pyplot as plt

CATEGORIES = {
    "Embedded_In_Memory": ("Embedded In-Memory", "#d4e6f1", ["Faiss", "Annoy", "HNSWLib", "Chroma", "Weaviate"]),
    "Embedded_On_Disk":   ("Embedded On-Disk",   "#d5f5e3", ["LanceDB"]),
    "Server_Batched":     ("Server Batched",      "#fde8d8", ["Milvus"]),
    "Server_Per_Query":   ("Server Per-Query",    "#f9ebea", ["Redis", "ES", "PgVector", "Meili"]),
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x", "p", "H", "<", ">", "1", "2", "3", "4", "8"]


def _get_category(name):
    prefix = name.split("_")[0]
    for cat_key, (_, _, prefixes) in CATEGORIES.items():
        if prefix in prefixes:
            return cat_key
    return None


def _build_color_marker_maps(tunable_results, non_tunable_results):
    all_names = list(tunable_results.keys()) + list(non_tunable_results.keys())
    n = len(all_names)
    cmap = plt.colormaps["tab10"] if n <= 10 else plt.colormaps["tab20"]
    color_map  = {name: cmap(i / max(n - 1, 1)) for i, name in enumerate(all_names)}
    marker_map = {name: MARKERS[i % len(MARKERS)] for i, name in enumerate(all_names)}
    return color_map, marker_map


def _draw_group(ax, tunable_names, non_tunable_names, tunable_results, non_tunable_results, color_map, marker_map):
    for name in tunable_names:
        if name not in tunable_results:
            continue
        # Drop points where timeout/error produced None recall or qps
        pts = [p for p in tunable_results[name]
               if p.get("recall") is not None and p.get("qps") is not None]
        if not pts:
            continue
        recalls = [p["recall"] for p in pts]
        qps_list = [p["qps"] for p in pts]
        ax.plot(recalls, qps_list, color=color_map[name], marker=marker_map[name],
                label=name, linewidth=1.5, markersize=5, zorder=3)

    for name in non_tunable_names:
        if name not in non_tunable_results:
            continue
        recall, qps = non_tunable_results[name]
        if recall is None or qps is None:
            continue
        ax.scatter(recall, qps, color=color_map[name], marker=marker_map[name],
                   s=120, label=name, zorder=3)

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Recall–Queries per Second tradeoff — up and to the right is better",
        fontsize=8, style="italic"
    )


def plot_speed_recall(tunable_results, non_tunable_results, ds_name, out_dir):
    """
    tunable_results  : {index_name: [{"recall": r, "qps": q, ...param}, ...]}
    non_tunable_results: {index_name: (recall, qps)}
    """
    os.makedirs(out_dir, exist_ok=True)

    color_map, marker_map = _build_color_marker_maps(tunable_results, non_tunable_results)

    all_tunable     = list(tunable_results.keys())
    all_non_tunable = list(non_tunable_results.keys())

    def _save_plot(fig, ax, t_names, nt_names, filename, title):
        fig.suptitle(title)
        _draw_group(ax, t_names, nt_names, tunable_results, non_tunable_results, color_map, marker_map)
        visible = [n for n in t_names if n in tunable_results and tunable_results[n]] + \
                  [n for n in nt_names if n in non_tunable_results]
        if visible:
            ax.legend(handles=[
                plt.Line2D([0], [0], color=color_map[n], marker=marker_map[n],
                           linewidth=1.5 if n in tunable_results else 0,
                           markersize=6, label=n)
                for n in visible
            ], fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 0.82, 0.95])
        plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    # All indexes
    fig, ax = plt.subplots(figsize=(13, 7))
    _save_plot(fig, ax, all_tunable, all_non_tunable,
               "speed_recall.png", f"Speed-Recall Tradeoff — {ds_name}")

    # Per category
    for cat_key, (cat_label, cat_color, _) in CATEGORIES.items():
        cat_t  = [n for n in all_tunable     if _get_category(n) == cat_key]
        cat_nt = [n for n in all_non_tunable if _get_category(n) == cat_key]
        if not cat_t and not cat_nt:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor(cat_color + "33")
        _save_plot(fig, ax, cat_t, cat_nt,
                   f"speed_recall_{cat_key}.png",
                   f"Speed-Recall Tradeoff — {ds_name} — {cat_label}")
