import lancedb
import pandas as pd

db = lancedb.connect("./lance_test_db")

data = pd.DataFrame({
    "id": [0, 1, 2],
    "vector": [
        [1.0, 2.0],
        [2.0, 3.0],
        [10.0, 10.0],
    ],
})

table = db.create_table("vectors", data=data, mode="overwrite")

results = table.search([1.1, 2.1]).limit(2).to_list()
print(results)