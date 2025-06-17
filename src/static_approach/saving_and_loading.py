import msgpack
import pickle

# MessagePack
def load_msgpack(file_path="./representatives.msgpack"):
    with open(file_path, "rb") as f:
        file = msgpack.unpackb(f.read(), raw=False)
        return file

def save_msgpack(file, file_path="./representatives.msgpack"):
    with open(file_path, "wb") as f:
        packed = msgpack.packb(file, use_bin_type=True)
        f.write(packed)

# Pickle
def save_pickle(file, file_path="./representatives.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path="./representatives.pkl"):
    with open(file_path, "rb") as f:
        file = pickle.load(f)
        return file