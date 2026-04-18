import pytest
import numpy as np

from src import (
    FaissFlatL2Index, FaissFlatIPIndex, FaissFlatHNSWIndex,
    FaissScalarQuantizerL2Index, FaissScalarQuantizerIPIndex,
    QdrantIndex, ChromaIndex, AnnoyIndex, HNSWLibIndex, WeaviateIndex,
    LanceDBFlatIndex, LanceDBIVFIndex,
    MilvusIndex, RedisIndex, ElasticsearchIndex, PgvectorIndex, MeilisearchIndex,
)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
D   = 32    # dimensionality
N   = 200   # number of database vectors
NQ  = 10    # number of query vectors
K   = 10    # neighbors to retrieve
MIN_RECALL = 0.8  # minimum acceptable recall (exact indexes should hit 1.0)

rng     = np.random.default_rng(42)
DATA    = rng.random((N, D)).astype("float32")
QUERIES = rng.random((NQ, D)).astype("float32")

# ── Ground truth ──────────────────────────────────────────────────────────────
def ground_truth_l2(data, queries, k):
    neighbors = []
    for q in queries:
        dists = np.linalg.norm(data - q, axis=1)
        neighbors.append(np.argsort(dists)[:k])
    return np.array(neighbors)

def ground_truth_cosine(data, queries, k):
    data_n  = data    / (np.linalg.norm(data,    axis=1, keepdims=True) + 1e-9)
    query_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    neighbors = []
    for q in query_n:
        sims = data_n @ q
        neighbors.append(np.argsort(-sims)[:k])
    return np.array(neighbors)

GT_L2     = ground_truth_l2(DATA, QUERIES, K)
GT_COSINE = ground_truth_cosine(DATA, QUERIES, K)

# ── Recall helper ─────────────────────────────────────────────────────────────
def recall(I, ground_truth):
    score = 0
    for i in range(len(I)):
        true_set = set(int(x) for x in ground_truth[i][:K])
        pred_set = set(int(x) for x in I[i][:K])
        score += len(true_set & pred_set) / K
    return score / len(I)

# ── Run helper ────────────────────────────────────────────────────────────────
def run_index(index, data, queries):
    index.train(data)
    index.add(data)
    _, I = index.search(queries, K)
    if hasattr(index, "close"):
        index.close()
    return I

# ── Embedded L2 indexes ───────────────────────────────────────────────────────
@pytest.mark.parametrize("make_index", [
    lambda: FaissFlatL2Index(D),
    lambda: FaissFlatHNSWIndex(D),
    lambda: FaissScalarQuantizerL2Index(D),
    lambda: QdrantIndex(D, metric_type="l2"),
    lambda: ChromaIndex(D, metric_type="l2"),
    lambda: AnnoyIndex(D, metric_type="l2"),
    lambda: HNSWLibIndex(D, metric_type="l2"),
], ids=[
    "FaissFlatL2", "FaissHNSW", "FaissScalarQuantizerL2",
    "Qdrant_L2", "Chroma_L2", "Annoy_L2", "HNSWLib_L2",
])
def test_correctness_l2(make_index):
    I = run_index(make_index(), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

# ── Embedded Angular indexes ──────────────────────────────────────────────────
@pytest.mark.parametrize("make_index", [
    lambda: FaissFlatIPIndex(D),
    lambda: FaissScalarQuantizerIPIndex(D),
    lambda: QdrantIndex(D, metric_type="cosine"),
    lambda: ChromaIndex(D, metric_type="cosine"),
    lambda: AnnoyIndex(D, metric_type="angular"),
    lambda: HNSWLibIndex(D, metric_type="cosine"),
], ids=[
    "FaissFlatIP", "FaissScalarQuantizerIP",
    "Qdrant_Angular", "Chroma_Angular", "Annoy_Angular", "HNSWLib_Angular",
])
def test_correctness_angular(make_index):
    I = run_index(make_index(), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

# ── Embedded Weaviate ─────────────────────────────────────────────────────────
def test_correctness_weaviate_l2():
    I = run_index(WeaviateIndex(D, metric_type="l2"), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

def test_correctness_weaviate_angular():
    I = run_index(WeaviateIndex(D, metric_type="cosine"), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

# ── LanceDB (on-disk, uses a temp directory) ──────────────────────────────────
def test_correctness_lancedb_flat_l2(tmp_path):
    I = run_index(LanceDBFlatIndex(D, metric_type="l2", db_path=str(tmp_path / "flat")), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

def test_correctness_lancedb_flat_angular(tmp_path):
    I = run_index(LanceDBFlatIndex(D, metric_type="cosine", db_path=str(tmp_path / "flat_ang")), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

def test_correctness_lancedb_ivf_l2(tmp_path):
    I = run_index(LanceDBIVFIndex(D, metric_type="l2", db_path=str(tmp_path / "ivf")), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

def test_correctness_lancedb_ivf_angular(tmp_path):
    I = run_index(LanceDBIVFIndex(D, metric_type="cosine", db_path=str(tmp_path / "ivf_ang")), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

# ── Server indexes (require Docker) ──────────────────────────────────────────
# Run with: poetry run pytest -m docker
# Skip with: poetry run pytest -m "not docker"
pytestmark_docker = pytest.mark.docker

@pytest.mark.docker
@pytest.mark.parametrize("make_index", [
    lambda: MilvusIndex(D, metric_type="L2", index_type="FLAT"),
    lambda: MilvusIndex(D, metric_type="IP", index_type="FLAT"),
    lambda: MilvusIndex(D, metric_type="L2", index_type="HNSW"),
    lambda: MilvusIndex(D, metric_type="L2", index_type="IVF_FLAT"),
], ids=["Milvus_FLAT_L2", "Milvus_FLAT_Angular", "Milvus_HNSW_L2", "Milvus_IVF_FLAT_L2"])
def test_correctness_milvus(make_index):
    index = make_index()
    gt = GT_L2 if index.metric_type == "L2" else GT_COSINE
    I = run_index(index, DATA, QUERIES)
    assert recall(I, gt) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_redis_l2():
    I = run_index(RedisIndex(D, metric_type="l2"), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_redis_angular():
    I = run_index(RedisIndex(D, metric_type="cosine"), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_elasticsearch_l2():
    I = run_index(ElasticsearchIndex(D, metric_type="l2"), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_elasticsearch_angular():
    I = run_index(ElasticsearchIndex(D, metric_type="cosine"), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_pgvector_l2():
    I = run_index(PgvectorIndex(D, metric_type="l2"), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_pgvector_angular():
    I = run_index(PgvectorIndex(D, metric_type="cosine"), DATA, QUERIES)
    assert recall(I, GT_COSINE) >= MIN_RECALL

@pytest.mark.docker
def test_correctness_meilisearch():
    I = run_index(MeilisearchIndex(D), DATA, QUERIES)
    assert recall(I, GT_L2) >= MIN_RECALL