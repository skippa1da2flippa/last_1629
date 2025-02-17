import os
from dataset_handler.dataset_fetcher import download_and_save_dataset, DatasetType, \
    dataset_split_max_size
from preprocessing.preprocess import from_raw_to_processed
from preprocessing.single_hop_multi_process import raw_dataset_preprocess
from utils.max_subgraph_getter import get_max_subgraph


def from_planetoid_to_processed(
    dataset_name: DatasetType, base_path: str,
    num_workers: int = 10, num_val: float = 0.1,
    num_test: float = 0.2, hops: list[int] = None,
    inclusion_map: list[int] = None
) -> None:

    download_and_save_dataset(
        dataset_name=dataset_name,
        base_path=base_path,
        num_val=num_val,
        num_test=num_test
    )

    raw_dataset_preprocess(
        base_path=base_path,
        dataset_name=dataset_name,
        inclusion_map=[1, 0, 0],
        num_workers=num_workers
    )

    train_path = os.path.join(base_path, f"{dataset_name.name}\\raw_dataset\\train_hop")
    maxi: list[int] = get_max_subgraph(path=train_path)

    dataset_split_max_size[dataset_name.value] = maxi

    raw_dataset_preprocess(
        base_path=base_path,
        dataset_name=dataset_name,
        inclusion_map=[0, 1, 1],
        num_workers=num_workers
    )

    dataset_path: str = os.path.join(base_path, f"{dataset_name.name}")

    from_raw_to_processed(
        dataset_path=dataset_path,
        inclusion_map=inclusion_map,
        hops=hops
    )

# use case
"""
path: str = os.getcwd()
from_planetoid_to_processed(base_path=path, dataset_name=DatasetType.PUBMED)
"""