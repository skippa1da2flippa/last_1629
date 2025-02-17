import gc
import os
import time
from typing import Any, Tuple, List
from torch.multiprocessing import Pool
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from dataset_handler.dataset_fetcher import split_dataset_fetcher, DatasetType, dataset_split_max_size
from dataset_handler.transformer_dataset import DatasetItem
from preprocessing.negative_sampling import double_get_ids_neg_sampling, get_features_neg_sampling
from preprocessing.preprocess import preprocess_one_transformer
from utils.data_saver import save


def preprocess_edge_chunk(
        edge_chunk: Tensor, label_chunk: Tensor, data: Data, hop: int,
        index: int, path: str, train: bool = True, device: str = "cuda"
) -> int:
    """Preprocess a chunk of edges."""
    results = []

    print(f"Process number {index} starting preprocessing with edge_chunk of size: {edge_chunk.shape[1]}")
    for idx in range(edge_chunk.shape[1]):
        start_time = time.time()
        result = preprocess_one_transformer(
            data=data,
            target_nodes=edge_chunk[:, idx],
            hop=hop,
            label=label_chunk[idx],
            device=device,
            train=train,
            log=False
        )
        end_time = time.time()

        if idx % 10 == 0:
            print(
                f"Process n. {index} just computed the {idx}_th, link remaining links to compute: {edge_chunk.shape[1] - idx - 1}. "
                f" Expected time for finishing: {((end_time - start_time) * (edge_chunk.shape[1] - idx - 1)) / 3600:.2f}")

        results.append(result)

    full_path = os.path.join(path, f"slice_{index}_hop_{hop}.pt")
    save(results, full_path)

    print(f"Process number {index} finished preprocessing, stored result at: {full_path}")

    return 1


def heavenly_split(edges: Tensor, num_workers: int, labels: Tensor = None) -> Tuple[List[Tensor], List[Tensor]]:
    # Split edges
    evenly_splits_edges: np.ndarray = np.linspace(
        start=0, stop=edges.shape[1], num=num_workers + 1,
        dtype=np.int32
    )
    edge_chunks: List[int] = [
        evenly_splits_edges[i + 1] - evenly_splits_edges[i]
        for i in range(0, len(evenly_splits_edges) - 1)
    ]
    edges_chunk: List[Tensor] = list(torch.split(tensor=edges, split_size_or_sections=edge_chunks, dim=1))

    # Split labels (if provided)
    labels_chunk: List[Tensor] = []
    if labels is not None:
        evenly_splits_labels: np.ndarray = np.linspace(
            start=0, stop=labels.shape[0], num=num_workers + 1,
            dtype=np.int32
        )
        label_chunks: List[int] = [
            evenly_splits_labels[i + 1] - evenly_splits_labels[i]
            for i in range(0, len(evenly_splits_labels) - 1)
        ]
        labels_chunk = list(torch.split(tensor=labels, split_size_or_sections=label_chunks, dim=0))

    return edges_chunk, labels_chunk


# The device must be cpu otherwise the tensors get corrupted and become null-vectors
def parallel_preprocessing(
        data: Data, hop: int, num_workers: int,
        path: str = None, train: bool = True,
        device: str = "cpu"
) -> None:
    """Distribute preprocessing work across workers."""
    # Split the edges into chunks
    data.edge_label_index = data.edge_label_index.to(device).share_memory_()
    data.edge_label = data.edge_label.to(device).share_memory_()
    data.edge_index = data.edge_index.to(device).share_memory_()

    edge_chunks, labels_chunks = heavenly_split(
        edges=data.edge_label_index, labels=data.edge_label, num_workers=num_workers
    )

    args = [
        (edge_chunks[idx], labels_chunks[idx], data, hop, idx, path, train, device)
        for idx in range(len(edge_chunks))
    ]

    results = []

    # Use ProcessPoolExecutor for parallel processing
    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(preprocess_edge_chunk, args)

        for res in worker_results:
            results.append(res)


"""
    This function was written with the purpose of computing the preprocessing 
    just onto a subset of edge_index/edge_label_index
"""


def parallel_preprocessing2(
        data: Data, hop: int,
        edge_label_index: Tensor,
        edge_label: Tensor,
        num_workers: int,
        path: str, train: bool = True,
        device: str = "cpu"
) -> list[dict[str, Any]]:
    """Distribute preprocessing work across workers."""
    # Split the edges into chunks
    edge_label_index = edge_label_index.to(device).share_memory_()
    edge_label = edge_label.to(device).share_memory_()
    data.edge_index = data.edge_index.to(device).share_memory_()

    edge_chunks, labels_chunks = heavenly_split(
        edges=edge_label_index, labels=edge_label, num_workers=num_workers
    )

    args = [
        (edge_chunks[idx], labels_chunks[idx], data, hop, idx, path, train, device)
        for idx in range(len(edge_chunks))
    ]

    results = []

    # Use ProcessPoolExecutor for parallel processing
    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(preprocess_edge_chunk, args)

        for res in worker_results:
            results.extend(res)

    return results


def raw_dataset_preprocess(
        base_path: str, dataset_name: DatasetType,
        inclusion_map: list[int] = None,
        num_workers: int = 10,
        hops: list[int] = None
) -> None:
    inclusion_map = [1, 1, 1] if inclusion_map is None else inclusion_map

    train, val, test = split_dataset_fetcher(path=base_path, dataset_name=dataset_name)

    train_path: str = os.path.join(base_path, f"{dataset_name.name}\\raw_dataset\\train_hop")
    val_path: str = os.path.join(base_path, f"{dataset_name.name}\\raw_dataset\\val_hop")
    test_paths: str = os.path.join(base_path, f"{dataset_name.name}\\raw_dataset\\test_hop")

    for split, split_path, mask in zip([train, val, test], [train_path, val_path, test_paths], inclusion_map):
        if mask:
            hops = [1, 2, 3] if not hops else hops
            for hop in hops:
                full_path: str = os.path.join(split_path, f"hop_{hop}")
                parallel_preprocessing(
                    data=split, hop=hop,
                    num_workers=num_workers,
                    path=full_path
                )

                print(f"Just completed hop: {hop} for {split_path}")


def process_one_hit(
        edge_chunk: Tensor,
        data: Data,
        max_num_neg: int,
        dataset_name: DatasetType,
        alter_labeling: bool,
        hop: int,
        index: int,
        dataset_path: str,
        device: str = "cpu"
) -> None:
    res: list[Tuple[list[DatasetItem], list[Tensor]]] = []

    print(f"Process number {index} starting hit-k-processing with edge_chunk of size: {edge_chunk.shape[1]}")

    for idx in range(edge_chunk.shape[1]):
        start_time = time.time()
        result = double_get_ids_neg_sampling(
            target_u=edge_chunk[0, idx],
            target_v=edge_chunk[1, idx],
            edge_index=data.edge_index,
            dataset_name=dataset_name,
            num_neg_samples=max_num_neg
        )

        negative_set, sub_graphs = get_features_neg_sampling(
            idx_neg_sampled=result, hop=hop,
            data=data, target_nodes=edge_chunk[:, idx],
            dataset_name=dataset_name.value,
            alter_labeling=alter_labeling,
            device=device
        )

        res.append(
            (negative_set, sub_graphs)
        )

        end_time = time.time()
        if idx % 10 == 0:
            elapsed_time = end_time - start_time  # Time for this iteration in seconds
            remaining_links = edge_chunk.shape[1] - idx - 1
            estimated_remaining_time = (elapsed_time * remaining_links) / 3600  # Convert to hours

            print(
                f"Process n. {index} just computed the {idx}_th link. Remaining links to compute: {remaining_links}. "
                f"Expected time to finish: {estimated_remaining_time:.2f} hours.")

        if idx == edge_chunk.shape[1] // 2:
            final_path: str = os.path.join(dataset_path, "HITS", f"process_{index}", f"{0}.pt")
            save(obj=res, path=final_path)

            print(f"Just saved split in path: {final_path} for process: {index}, part: {0}")

            res = []

    final_path: str = os.path.join(dataset_path, "HITS", f"process_{index}", f"{1}.pt")
    save(obj=res, path=final_path)

    print(f"Just saved split in path: {final_path} for process: {index}, part: {1}")
    print(f"Process number {index} finished preprocessing")


"""
This function should receive a test_data with just positive samples
"""


def prepare_hit_k(
        test_data: Data,
        hop: int,
        max_num_neg: int,
        dataset_name: DatasetType,
        base_path: str,
        alter_labeling: bool = False,
        num_workers: int = 10,
        inclusion_map: list[int] = None
) -> None:
    if inclusion_map is None:
        inclusion_map = [1 for _ in range(num_workers)]

    dataset_path: str = os.path.join(base_path, dataset_name.name)
    edge_label_index: Tensor = test_data.edge_label_index
    edge_label: Tensor = test_data.edge_label

    # Split the edges into chunks
    edge_label_index = edge_label_index.cpu().share_memory_()
    test_data.edge_index = test_data.edge_index.cpu().share_memory_()

    edge_chunks, label_chunks = heavenly_split(
        edges=edge_label_index, num_workers=num_workers
    )

    args = [
        (
            edge_chunks[idx], test_data,
            max_num_neg, dataset_name, alter_labeling,
            hop, idx, dataset_path, "cpu"
        )
        for idx in range(len(edge_chunks)) if inclusion_map[idx]
    ]

    with Pool(processes=num_workers) as pool:
        pool.starmap(process_one_hit, args)
