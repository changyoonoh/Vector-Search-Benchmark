import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from src.abstract_vector_index import AbstractVectorIndex


class MilvusIndex(AbstractVectorIndex):
    def __init__(
        self,
        d,
        collection_name="bench_vectors",
        host="127.0.0.1",
        port="19530",
        metric_type="L2",      # only the default value, this can be configured to either "L2" or "IP"
        index_type="FLAT",     # start with "FLAT"
        index_params=None, # not really needed for FLAT but maybe for stuff like sq
        ):
        self.d = d
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.index_type = index_type
        if index_params is not None:
            self.index_params = index_params
        else:
            if index_type == "FLAT":
                self.index_params = {}
            elif index_type == "HNSW":
                self.index_params = {"M": 16, "efConstruction": 200}
            elif index_type == "IVF_FLAT":
                self.index_params = {"nlist": 128} #nlist for number of clusters
            elif index_type == "IVF_SQ8":
                self.index_params = {"nlist": 128}
            elif index_type == "IVF_PQ":
                self.index_params = {"nlist": 128, "m": 8, "nbits": 8}
            else:
                raise ValueError(f"Unsupported Milvus index_type: {index_type}")

        connections.connect(host=host, port=port)

        if utility.has_collection(collection_name): #If we don't have this multiple Benchmark runs would produce wrong results, repeated runs would reuse old data
            utility.drop_collection(collection_name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), #id is primary key for each record(each vector from our dataset), we need this in milvus
                FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=d), #
            ],
            description="benchmark vectors",
        )
        self.col = Collection(collection_name, schema)

        self.col.create_index(
            field_name="vec",
            index_params={
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": self.index_params,
            },
        )

    def train(self, data):
        return  # FLAT doesn't train, so not needed for now

    def add(self, data):
        self.col.insert([data.tolist()]) #While our data of vectors are arrays, milvus expects lists, so we will get list of lists, Q: what is the outer list for?
        self.col.flush() #This is needed mostly for benchmarking, because this will get the data perminently stored, not just temporarily in memory// It makes data written to disk & saved in storage
        self.col.load() #loads collection into memory so searching becomes faster

    def search(self, queries, k):

        if self.index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            search_params = {"nprobe": 16}
        elif self.index_type == "HNSW":
            search_params = {"ef": 64}
        else:
            search_params = {}

        res = self.col.search(
            data=queries.tolist(),
            anns_field="vec",
            param={"metric_type": self.metric_type, "params": search_params},
            limit=k,
        )

        D = np.array([[hit.distance for hit in hits] for hits in res], dtype=np.float32)
        I = np.array([[hit.id for hit in hits] for hits in res], dtype=np.int64)
        return D, I
