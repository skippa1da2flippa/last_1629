import gc
import os
import pickle
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from dataset_handler.dataset_fetcher import dataset_sizes, DatasetType, dataset_split_max_size
from preprocessing.data.sub_graph_wrapper import SubGraphWrapper
from preprocessing.message_passing import message_passing
from preprocessing.sub_graph_ext import subgraph_extraction
from preprocessing.transformer_pos_enc import get_all_positional
from utils.data_saver import save


def preprocess_one_transformer(
        data: Data, target_nodes: Tensor, hop: int, label: Tensor,
        log: bool = True, train: bool = True,
        dataset_name: str = DatasetType.PUBMED.value,
        mp: bool = False,
        device: str = "cuda"
) -> dict[str, Any]:
    if hop == 1:
        max_num_nodes: int = dataset_split_max_size[dataset_name][0]
    elif hop == 2:
        max_num_nodes: int = dataset_split_max_size[dataset_name][1]
    else:
        max_num_nodes: int = dataset_split_max_size[dataset_name][2]

    packet: SubGraphWrapper = subgraph_extraction(
        graph=data, target_nodes=target_nodes, k_hop=hop,
        subgraph_size=max_num_nodes, graph_size=dataset_sizes[dataset_name][0],
        return_single_subgraphs=True, device=device, remove_edge=train
    )

    pos_mat: Tensor = get_all_positional(
        packet=packet, log=log, device=device,
        num_nodes=dataset_sizes[dataset_name][0]
    )
    adj = to_dense_adj(
        packet.edge_index.to(device),
        max_num_nodes=dataset_sizes[dataset_name][0]
    ).squeeze(dim=0)

    mp_message: list[Tensor] = []

    if mp:
        for ids in range(hop):
            mp_adj = message_passing(adj_mat=adj, layers=ids, message_decay=True, device=device)
            mp_adj = mp_adj[:, packet.subgraph_ids]
            mp_adj = mp_adj[packet.subgraph_ids, :]
            mp_message.append(mp_adj)

    else:
        mp_message = [torch.tensor([]) for _ in range(hop)]

    return {
        "packet": packet,
        "pos_enc": pos_mat,
        "mp": mp_message,
        "target_nodes": target_nodes,
        "label": label.float()
    }


def from_raw_to_processed(
        dataset_path: str,
        inclusion_map: list[int] = None,
        hops: list[int] = None,
) -> None:
    if inclusion_map is None:
        inclusion_mask = [1, 1, 1]
    else:
        inclusion_mask = inclusion_map

    raw_train_path = os.path.join(dataset_path, "raw_dataset\\train_hop")
    raw_val_path = os.path.join(dataset_path, "raw_dataset\\val_hop")
    raw_test_path = os.path.join(dataset_path, "raw_dataset\\test_hop")

    processed_train_path = os.path.join(dataset_path, "processed_dataset\\train_dataset")
    processed_val_path = os.path.join(dataset_path, "processed_dataset\\val_dataset")
    processed_test_path = os.path.join(dataset_path, "processed_dataset\\test_dataset")

    raw_paths = [raw_train_path, raw_val_path, raw_test_path]
    processed_paths = [processed_train_path, processed_val_path, processed_test_path]

    # Filter paths based on inclusion_map
    raw_paths = [raw_paths[idx] for idx in range(len(inclusion_mask)) if inclusion_mask[idx]]
    processed_paths = [processed_paths[idx] for idx in range(len(inclusion_mask)) if inclusion_mask[idx]]

    for raw_path, processed_path in zip(raw_paths, processed_paths):

        if hops is None:
            values = range(1, 4)  # if raw_path != raw_train_path else range(1, 3)
        else:
            values = hops

        target_nodes: list = []
        labels: list = []

        for idj, hop in enumerate(values):
            pos_enc_lst: list = []
            subgraph_lst: list = []
            mp_0: list = []
            mp_1: list = []
            mp_2: list = []
            mp_3: list = []
            full_path = os.path.join(raw_path, f"hop_{hop}_transformer.pt")

            with open(full_path, 'rb') as f:
                res = pickle.load(f)

            for cell in res:
                if idj == 0:
                    target_nodes.append(cell["target_nodes"])
                    labels.append(cell["label"])

                pos_enc_lst.append(cell["pos_enc"])
                subgraph_lst.append(cell["packet"].subgraph_ids)

                for idx, mp in enumerate(cell["mp"]):

                    if idx == 0:
                        mp_0.append(mp)
                    elif idx == 1:
                        mp_1.append(mp)
                    elif idx == 2:
                        mp_2.append(mp)
                    else:
                        mp_3.append(mp)

            base_path = os.path.join(processed_path, f"hop_{hop}")

            for mp_index, data in enumerate([mp_0, mp_1, mp_2, mp_3]):
                if len(data) > 0:
                    full_path = os.path.join(base_path, f"mp_{mp_index}.pt")
                    save(data, full_path)

            save(pos_enc_lst, os.path.join(base_path, "pos_enc.pt"))
            save(subgraph_lst, os.path.join(base_path, "subgraph_ids.pt"))

        split_path: str = os.path.join(dataset_path, r"split\train_dataset.pt")
        with open(split_path, "rb") as f:
            BOW_emb: Tensor = pickle.load(f).x

        save({
            "target_nodes": target_nodes,
            "labels": labels,
            "BOW": BOW_emb
        }, os.path.join(processed_path, "targets_labels.pt"))


def from_splitted_raw_to_processed(
        dataset_path: str,
        inclusion_map: list[int] = None,
        hops: list[int] = None,
) -> None:
    if inclusion_map is None:
        inclusion_mask = [1, 1, 1]
    else:
        inclusion_mask = inclusion_map

    raw_train_path = os.path.join(dataset_path, "raw_dataset\\train_hop")
    raw_val_path = os.path.join(dataset_path, "raw_dataset\\val_hop")
    raw_test_path = os.path.join(dataset_path, "raw_dataset\\test_hop")

    processed_train_path = os.path.join(dataset_path, "processed_dataset\\train_dataset")
    processed_val_path = os.path.join(dataset_path, "processed_dataset\\val_dataset")
    processed_test_path = os.path.join(dataset_path, "processed_dataset\\test_dataset")

    raw_paths = [raw_train_path, raw_val_path, raw_test_path]
    processed_paths = [processed_train_path, processed_val_path, processed_test_path]

    # Filter paths based on inclusion_map
    raw_paths = [raw_paths[idx] for idx in range(len(inclusion_mask)) if inclusion_mask[idx]]
    processed_paths = [processed_paths[idx] for idx in range(len(inclusion_mask)) if inclusion_mask[idx]]

    for raw_path, processed_path in zip(raw_paths, processed_paths):

        if hops is None:
            values = range(1, 4)
        else:
            values = hops

        for hop in values:
            print(f"Computing from raw to processed for hop: {hop}")

            hop_path = os.path.join(raw_path, f"hop_{hop}")
            for file_idx, filename in enumerate(os.listdir(hop_path)):
                pos_enc_lst: list = []
                subgraph_lst: list = []
                target_nodes: list = []
                labels: list = []

                gc.collect()

                full_path = os.path.join(hop_path, filename)
                with open(full_path, "rb") as f:
                    res = pickle.load(f)

                for cell in res:
                    target_nodes.append(cell["target_nodes"])
                    labels.append(cell["label"])
                    pos_enc_lst.append(cell["pos_enc"])
                    subgraph_lst.append(cell["packet"].subgraph_ids)

                base_path = os.path.join(processed_path, f"hop_{hop}")

                save(pos_enc_lst, os.path.join(base_path, f"pos_enc_slice_{file_idx}.pt"))
                save(subgraph_lst, os.path.join(base_path, f"subgraph_ids_slice_{file_idx}.pt"))
                save(target_nodes, os.path.join(base_path, f"target_nodes_slice_{file_idx}.pt"))
                save(labels, os.path.join(base_path, f"labels_slice_{file_idx}.pt"))

                print(f"Just saved slice: {filename}")

        split_path: str = os.path.join(dataset_path, r"split\train_dataset.pt")
        with open(split_path, "rb") as f:
            BOW_emb: Tensor = pickle.load(f).x

        save({
            "BOW": BOW_emb
        }, os.path.join(processed_path, "targets_labels.pt"))
