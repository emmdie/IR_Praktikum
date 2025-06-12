import msgpack
import pickle

# MessagePack
def load_representatives_msgpack(file_path="representatives.msgpack"):
    with open(file_path, "rb") as f:
        representatives = msgpack.unpackb(f.read(), raw=False)
        return representatives

def save_representatives_msgpack(representatives, file_path="representatives.msgpack"):
    with open(file_path, "wb") as f:
        packed = msgpack.packb(representatives, use_bin_type=True)
        f.write(packed)

# Pickle
def save_representatives_pickle(representatives, file_path="representatives.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(representatives, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_representatives_pickle(file_path="representatives.pkl"):
    with open(file_path, "rb") as f:
        representatives = pickle.load(f)
        return representatives