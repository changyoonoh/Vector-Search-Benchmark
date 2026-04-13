import pytest 
from benchmark import 

@pytest.parametrize("index_type", ["hnsw", "ivf", "flat"])