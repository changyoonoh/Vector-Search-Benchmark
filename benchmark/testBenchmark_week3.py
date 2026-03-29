import time
import faiss
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod



def benchmark(index, data, queries, k=5):
    t0 = time.perf_counter() #vs time.time() /// timer starts with benchmark

    index.add(data)
    t1 = time.perf_counter() # timer ends after similarity search
    
    index.search(queries, k)
    t2 = time.perf_counter() # timer ends after similarity search
    
    build_time = t1 - t0
    search_time = t2 - t1
    return build_time, search_time

d = 32 #dimension of data
nq = 20000 #number of queries
queries = np.random.random((nq, d)).astype("float32")
sizes = [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 1000000] #10 different dataset sizes
times_FlatL2 = []
times_FlatIP = []
times_HNSWFlat = []
times_SQ_L2 = []
times_SQ_IP = []
times_FlatL2_build = []
times_FlatIP_build = []
times_HNSWFlat_build = []
times_SQ_L2_build = []
times_SQ_IP_build = []


for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexFlatL2(d) 
    bt, st = benchmark(index, data, queries, k=5)
    times_FlatL2_build.append(bt)
    times_FlatL2.append(st)
    print(n, bt, st)

for n in sizes:
    data = np.random.random((n, d)).astype("float32") #normalization needed?
    index = faiss.IndexFlatIP(d)  
    bt, st = benchmark(index, data, queries, k=5)
    times_FlatIP_build.append(bt)
    times_FlatIP.append(st)
    print(n, bt, st)

for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexHNSWFlat(d, 32)   # takes two parameters where M is number of neighboring edges(degree) per node
    bt, st = benchmark(index, data, queries, k=5)
    times_HNSWFlat_build.append(bt)
    times_HNSWFlat.append(st)
    print(n, bt, st)


for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
    index.train(data)
    bt, st = benchmark(index, data, queries, k=5)
    times_SQ_L2.append(st)
    times_SQ_L2_build.append(bt)
    print(n, bt, st)

for n in sizes:
    data = np.random.random((n, d)).astype("float32")
    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(data)
    bt, st = benchmark(index, data, queries, k=5)
    times_SQ_IP_build.append(bt)
    times_SQ_IP.append(st)
    print(n, bt, st)

#plotting graph
plt.plot(sizes, times_FlatL2, label="FlatL2")
plt.plot(sizes, times_FlatIP, label="FlatIP")
plt.plot(sizes, times_HNSWFlat, label="HNSW")
plt.plot(sizes, times_SQ_L2, label="SQ_L2")
plt.plot(sizes, times_SQ_IP, label="SQ_IP")

#plt.plot(sizes, times_FlatL2_build, label="FlatL2_build")
#plt.plot(sizes, times_FlatIP_build, label="FlatIP_build")
#plt.plot(sizes, times_HNSWFlat_build, label="HNSW_build")
#plt.plot(sizes, times_SQ_L2_build, label="SQ_L2_build")
#plt.plot(sizes, times_SQ_IP_build, label="SQ_IP_build")


plt.xlabel("Number of Vectors")
plt.ylabel("Total time")
plt.title("FAISS Benchmark")
plt.legend()
plt.xticks(sizes, rotation=45)
plt.show()
