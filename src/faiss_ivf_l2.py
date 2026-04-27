import faiss
import numpy as np
from src.abstract_vector_index import AbstractVectorIndex


class FaissIVFFlatL2Index(AbstractVectorIndex):

    def __init__(self, d, nlist=128, nprobe=16):
        self.d = d
        self.nprobe = nprobe
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    def train(self, data):
        self.index.train(data)

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        self.index.nprobe = self.nprobe
        return self.index.search(queries, k)

    def set_query_params(self, params):
        self.nprobe = params.get("nprobe", self.nprobe)


class FaissIVFSQ8L2Index(AbstractVectorIndex):

    def __init__(self, d, nlist=128, nprobe=16):
        self.d = d
        self.nprobe = nprobe
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )

    def train(self, data):
        self.index.train(data)

    def add(self, data):
        self.index.add(data)

    def search(self, queries, k):
        self.index.nprobe = self.nprobe
        return self.index.search(queries, k)

    def set_query_params(self, params):
        self.nprobe = params.get("nprobe", self.nprobe)
