import faiss
import numpy as np

d = 4  # dimension
n = 1000  # database size, number of vectors
data = np.random.random((n, d)).astype("float32")
index = faiss.IndexFlatL2(d)  # build the index
index.add(data)  # add vectors to the index

q = np.random.random((5, d)).astype("float32")  # query vectors

k = 5  # number of nearest neighbors
D, I = index.search(q, k)  # actual search, I - indices of neighbors, D - distances
print("Indices of nearest neighbors:\n", I)
print("Distances to nearest neighbors:\n", D)

