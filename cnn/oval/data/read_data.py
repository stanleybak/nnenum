import pickle

filename = 'base_100.pkl'

with open(filename,"rb") as f:
    pandas_table = pickle.load(f)

print(pandas_table)

n = pandas_table.to_numpy()

print(f"first row as numpy: {n[0]}")
