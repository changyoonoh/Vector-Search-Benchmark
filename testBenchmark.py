import time
import faiss
import numpy as np
import matplotlib.pyplot as plt

def benchmark(index, data, queries, k=5):
    t0 = time.perf_counter() #vs time.time() /// timer starts with benchmark

    index.add(data)
    index.search(queries, k)

    t1 = time.perf_counter() # timer ends after similarity search
    total_time = t1 - t0

    return total_time

d = 32 #dimension of data
nq = 200 #number of queries
queries = np.random.random((nq, d)).astype("float32")
sizes = [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000] #10 different dataset sizes
times_FlatL2 = []
times_FlatIP = []
times_HNSWFlat = []

for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexFlatL2(d) 
    t = benchmark(index, data, queries, k=5)
    times_FlatL2.append(t)
    print(n, t)

for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexFlatIP(d)  
    t = benchmark(index, data, queries, k=5)
    times_FlatIP.append(t)
    print(n, t)

for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexHNSWFlat(d, 32)   # takes two parameters where M is number of neighboring edges(degree) per node
    t = benchmark(index, data, queries, k=5)
    times_HNSWFlat.append(t)
    print(n, t)

#plotting
plt.plot(sizes, times_FlatL2, label="FlatL2")
plt.plot(sizes, times_FlatIP, label="FlatIP")
plt.plot(sizes, times_HNSWFlat, label="HNSW")

plt.xlabel("Number of Vectors")
plt.ylabel("Total time")
plt.title("FAISS Benchmark")
plt.legend()
plt.xticks(sizes, rotation=45)
plt.show()
