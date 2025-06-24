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
def save_pickle(file, directory=".", file_name="representatives.pkl"):
    file_path = directory + "/" + file_name
    print(f"Saving pickle file to: {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(directory=".", file_name="representatives.pkl"):
    file_path = directory + "/" + file_name
    print(f"Trying to load file from: {file_path}")
    with open(file_path, "rb") as f:
        file = pickle.load(f)
        return file