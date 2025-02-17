import os
import pickle
from typing import Any


def save(obj: Any, path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)