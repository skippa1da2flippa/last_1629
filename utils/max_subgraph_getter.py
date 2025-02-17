import os
import pickle
from typing import Any


def get_max_subgraph(path: str, hops: list[int] = None, is_split: bool = False) -> list[int]:
    hops = [1, 2] if hops is None else hops
    res: list[list[dict]] = [[], []] if is_split else []
    maxi: list[int] = []

    for hop in hops:
        if not is_split:
            full_path: str = os.path.join(path, f'hop_{hop}_transformer.pt')
            with open(full_path, 'rb') as f:
                res.append(pickle.load(f))
        else:
            for idx in range(10):
                full_path: str = os.path.join(path, f'hop_{hop}\\slice_{idx}_hop_{hop}.pt')
                with open(full_path, 'rb') as f:
                    temp = pickle.load(f)

                temp = [tmp["packet"].subgraph_ids.shape[0] for tmp in temp]
                res[hop - 1].extend(temp)

    if not is_split:
        f = lambda d: d["packet"].subgraph_ids.shape[0]
    else:
        f = lambda d: d

    for split in res:
        sample = max(split, key=f)
        maxi.append(sample)

    return maxi

# USE CASE
# path = os.path.join(os.getcwd(), "CITESEER\\raw_dataset")
# train_path = os.path.join(path, "train_hop")
# maxi: list[int] = get_max_subgraph(path=train_path)
