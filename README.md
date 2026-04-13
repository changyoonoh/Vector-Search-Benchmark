# Vector Search Benchmark

A systematic benchmark comparing 12 vector search systems across
three real-world datasets, measuring build time, search time,
single-query latency, and recall at scales up to 1,000,000 vectors.

Unlike algorithm-level benchmarks like [ann-benchmarks](https://github.com/erikbern/ann-benchmarks),
this project measures systems as they are actually deployed —
including real network overhead, Docker infrastructure, and ingestion
cost — making it directly relevant to engineers choosing a vector
search system for production.

## Systems Benchmarked

Systems are categorized by architecture, which is the primary driver
of performance differences:

| Category | Systems |
|---|---|
| Embedded · In-Memory | FAISS, Qdrant, Chroma, Annoy, HNSWLib |
| Embedded · In-Memory · Embedded Server | Weaviate |
| Embedded · On-Disk | LanceDB |
| Server · In-Memory · Per-Query | Redis |
| Server · On-Disk · Batched | Milvus |
| Server · On-Disk · Per-Query | Elasticsearch, pgvector, Meilisearch |

All systems implement a consistent `AbstractVectorIndex` interface
with `train()`, `add()`, and `search()` methods, ensuring fair and
comparable measurements across all systems.

## Datasets

Datasets are sourced from [ann-benchmarks](https://ann-benchmarks.com)
in HDF5 format. All datasets are pre-split into train/test sets and
include precomputed ground truth for the top-100 nearest neighbors.

| Dataset | Dimensions | Train size | Test size | Neighbors | Distance | Use Case | Download |
|---|---|---|---|---|---|---|---|
| SIFT | 128 | 1,000,000 | 10,000 | 100 | Euclidean | Image feature vectors | [HDF5 (525MB)](https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5) |
| GloVe | 100 | 1,183,514 | 10,000 | 100 | Angular | Word embeddings | [HDF5 (485MB)](https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5) |
| Fashion-MNIST | 784 | 60,000 | 10,000 | 100 | Euclidean | Image classification | [HDF5 (228MB)](https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/fashion-mnist-784-euclidean.hdf5) |

## Metrics

- **Build time** — time to insert all vectors into the index
- **Search time** — total time to run 1,000 queries at k=10
- **Single-query latency** — median response time for one query in milliseconds, measured over 100 individual queries
- **Recall@10** — fraction of true top-10 nearest neighbors returned, evaluated against precomputed ground truth at full dataset size

## Setup

### Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker Desktop](https://www.docker.com/get-started) (required for Milvus, Redis, Elasticsearch, pgvector, Meilisearch)

### Step 1 — Clone the repo

```bash
git clone https://github.com/changyoonoh/Vector-Search-Benchmark.git
cd Vector-Search-Benchmark
```

### Step 2 — Install dependencies

```bash
poetry install
```

### Step 3 — Download datasets

Download the three datasets and place them in a folder anywhere on
your machine. The fastest way is via terminal:

```bash
wget https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5
wget https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5
wget https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/fashion-mnist-784-euclidean.hdf5
```

> On macOS, if `wget` is not installed: `brew install wget`
> On Windows, use `curl` instead:
> ```
> curl -L -o sift-128-euclidean.hdf5 https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5
> curl -L -o glove-100-angular.hdf5 https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5
> curl -L -o fashion-mnist-784-euclidean.hdf5 https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/fashion-mnist-784-euclidean.hdf5
> ```

### Step 4 — Update dataset paths

Open `benchmark/runBenchmark.py` and update the dataset paths at the
top of `main()` to match where you saved the files:

```python
datasets = [
    ("/your/path/to/sift-128-euclidean.hdf5",         "sift-128-euclidean",         "l2",     [...]),
    ("/your/path/to/glove-100-angular.hdf5",           "glove-100-angular",          "cosine", [...]),
    ("/your/path/to/fashion-mnist-784-euclidean.hdf5", "fashion-mnist-784-euclidean","l2",     [...]),
]
```

### Step 5 — Start Docker services

Make sure Docker Desktop is running, then start all server-based
services with:

```bash
docker-compose up -d
```

This starts the following containers as defined in `docker-compose.yml`:

| Service | Image | Port |
|---|---|---|
| Milvus | milvusdb/milvus:v2.6.11 | 19530 |
| Redis | redis/redis-stack:latest | 6379 |
| Elasticsearch | elasticsearch:9.3.0 | 9200 |
| pgvector | pgvector/pgvector:pg16 | 5432 |
| Meilisearch | getmeili/meilisearch:v1.36.0 | 7700 |

> Milvus also requires `etcd` and `minio` as dependencies — these
> are included in `docker-compose.yml` and start automatically.
> pgvector enables the vector extension automatically on first
> connection — no manual SQL setup needed.
> Weaviate runs as an embedded process and does not require Docker.

### Step 6 — Run the benchmark

```bash
cd benchmark
poetry run python runBenchmark.py
```

Results are saved automatically to `results/{timestamp}/{dataset}/`
as PNG plots. This folder is gitignored and will not be pushed to
GitHub.

### Windows notes

- Use PowerShell or Windows Terminal
- Replace forward slashes with backslashes in dataset paths, or use
  raw strings: `r"C:\Users\you\data\sift-128-euclidean.hdf5"`
- Docker Desktop must be running before `docker-compose up -d`
- If `poetry` is not recognized after install, restart your terminal

## Implementation Notes

**Fault tolerance** — if an index fails at any dataset size, the
error is logged and the benchmark continues to the next index.
Failed results are recorded as `None` and excluded from plots.

**Annoy n_trees** — Annoy's default of 10 trees is insufficient for
accurate results at large scale. This benchmark uses `n_trees=100`
as a minimum, matching the settings used by ann-benchmarks. At 1M
vectors with the default setting, Annoy returns essentially random
results.

**FAISS SQ training** — FAISS ScalarQuantizer requires a training
phase before indexing. Training time is not included in the reported
build time measurements. The `train()` call still happens internally
but is not timed separately.

**Recall methodology** — Recall@10 is only evaluated at the full
dataset size. Recall on partial datasets reflects how many true
neighbors happen to fall within the indexed subset, not algorithm
accuracy. At full scale, exact indexes like FAISS Flat should
achieve Recall@10 ≈ 1.0.

**pgvector extension** — The `pgvector/pgvector:pg16` Docker image
comes with the extension pre-installed. The wrapper enables it
automatically via `CREATE EXTENSION IF NOT EXISTS vector` on first
connection.

## Results

Coming soon.
