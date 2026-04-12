# Vector Search Benchmark

A systematic benchmark comparing 12 vector search systems across 
three real-world datasets, measuring build time, search time, and 
recall across scales up to 500,000 vectors.

## Systems Benchmarked

Systems are categorized by architecture, which is the primary driver 
of performance differences:

| Category | Systems |
|---|---|
| Embedded · In-Memory | FAISS, Qdrant, Chroma, Annoy, HNSWLib |
| Embedded · On-Disk | LanceDB |
| Server · In-Memory · Per-Query | Redis |
| Server · On-Disk · Batched | Milvus, Weaviate |
| Server · On-Disk · Per-Query | Elasticsearch, pgvector, Meilisearch |

All systems implement a consistent `AbstractVectorIndex` interface 
with `train()`, `add()`, and `search()` methods.

## Datasets

Datasets are sourced from [ann-benchmarks](https://ann-benchmarks.com) 
in HDF5 format, each with precomputed ground truth neighbors for 
recall evaluation.

| Dataset | Vectors | Dimension | Metric | Use Case |
|---|---|---|---|---|
| sift-128-euclidean | 1,000,000 | 128 | L2 | Image feature vectors |
| glove-100-angular | 1,183,514 | 100 | Cosine | Word embeddings |
| fashion-mnist-784-euclidean | 60,000 | 784 | L2 | Image classification |

## Metrics

- **Build time** — time to insert all vectors into the index
- **Search time** — time to run 1,000 queries at k=5
- **Recall@5** — fraction of true nearest neighbors returned, 
evaluated against ann-benchmarks ground truth

## Setup
```bash
# Install dependencies
poetry install

# Docker required for: Redis, Elasticsearch, pgvector, Meilisearch
docker-compose up -d

# Run benchmark
poetry run python benchmark/runBenchmark.py
```
