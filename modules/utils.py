import os
import pickle


def save_object(obj, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)