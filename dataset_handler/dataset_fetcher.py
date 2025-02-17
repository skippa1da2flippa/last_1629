import os
import pickle
from enum import Enum
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit


class DatasetType(Enum):
    CORA = 'Cora'
    CITESEER = 'CiteSeer'
    PUBMED = 'PubMed'


planetoid_paths: dict[str, str] = {
    DatasetType.CORA.name: f'/tmp/{DatasetType.CORA.value}',
    DatasetType.CITESEER.name: f'/tmp/{DatasetType.CITESEER.value}',
    DatasetType.PUBMED.name: f'/tmp/{DatasetType.PUBMED.value}'
}

# Each value is a triple representing num_nodes, num_edges and num_feats
dataset_sizes: dict[str, tuple[int, int, int]] = {
    DatasetType.CORA.value: (2708, 10556, 1433),
    DatasetType.CITESEER.value: (3327, 9104, 3703),
    DatasetType.PUBMED.value: (19717, 88648, 500)
}

# Each value represent the sample in the i-th hop which has the highest number of neighbors
dataset_split_max_size: dict[str, list[int]] = {
    DatasetType.CORA.value: [127, 342, 1000],
    DatasetType.CITESEER.value: [93, 250, 441],
    DatasetType.PUBMED.value: [195, 500, 1000]
}


def planetoid_dataset_downloader(
        dataset_name: DatasetType, num_val: float, num_test: float
) -> Tuple[Data, Data, Data]:
    dataset = Planetoid(
        root=planetoid_paths[dataset_name.name], name=dataset_name.value
    )
    data = dataset[0]

    # Split edges for link prediction with neg-pos ratio set to 1
    transform = RandomLinkSplit(
        is_undirected=True, num_val=num_val, num_test=num_test
    )
    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data


def split_dataset_fetcher(path: str, dataset_name: DatasetType) -> Tuple[Data, Data, Data]:
    base_path: str = os.path.join(path, dataset_name.name)
    train_path = os.path.join(base_path, "split", "train_dataset.pt")
    val_path = os.path.join(base_path, "split", "val_dataset.pt")
    test_path = os.path.join(base_path, "split", "test_dataset.pt")

    split: list[Data] = []

    for db_path in [train_path, val_path, test_path]:

        with open(db_path, 'rb') as f:
            split.append(pickle.load(f))

    return split[0], split[1], split[2]


def download_and_save_dataset(
        dataset_name: DatasetType, base_path: str,
        num_val: float = 0.1, num_test: float = 0.2
) -> None:
    train_dataset, val_dataset, test_dataset = planetoid_dataset_downloader(
        dataset_name=dataset_name, num_val=num_val,
        num_test=num_test
    )
    path = os.path.join(base_path, dataset_name.name)
    label_path = os.path.join(path, "split")

    os.makedirs(label_path, exist_ok=True)

    full_path: str = os.path.join(label_path, 'train_dataset.pt')
    with open(full_path, 'wb') as f:
        pickle.dump(train_dataset, f)

    full_path: str = os.path.join(label_path, 'val_dataset.pt')
    with open(full_path, 'wb') as f:
        pickle.dump(val_dataset, f)

    full_path: str = os.path.join(label_path, 'test_dataset.pt')
    with open(full_path, 'wb') as f:
        pickle.dump(test_dataset, f)


# Use-case example
"""
    dataset_name: DatasetType = DatasetType.CORA
    dataset_path: str = os.getcwd()
    
    download_and_save_dataset(dataset_name=dataset_name, dataset_path=dataset_path)
    
"""



