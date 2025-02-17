from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from dataset_handler.dataset_fetcher import dataset_split_max_size, DatasetType, dataset_sizes
from dataset_handler.transformer_dataset import DatasetItem
from preprocessing.data.sub_graph_wrapper import SubGraphWrapper
from preprocessing.sub_graph_ext import single_subgraph_extraction, subgraph_extraction
from preprocessing.transformer_pos_enc import get_all_positional

"""
This section is written just because I need to compute hits@k
For each positive samples this function shall return a Tensor 
containing the index of the nodes which happened to be randomly
fetched during the sampling operation, which do not have an actual
link with the input node
"""


def single_get_ids_neg_sampling(
        target: Tensor, edge_index: Tensor,
        graph_nodes: int, num_neg_samples: int = 500,
        device: str = "cpu"
) -> Tensor:
    node_mask, _ = single_subgraph_extraction(
        edge_index=edge_index, target_node=target,
        k_hop=1, graph_nodes=graph_nodes, device=device
    )

    non_neighbors: Tensor = torch.nonzero(~node_mask, as_tuple=False).reshape(-1)
    perm: Tensor = torch.randperm(non_neighbors.shape[0])

    return non_neighbors[perm[:num_neg_samples]]


def double_get_ids_neg_sampling(
        target_u: Tensor, target_v: Tensor,
        edge_index: Tensor,
        dataset_name: DatasetType,
        num_neg_samples: int
) -> Tuple[Tensor, Tensor]:
    non_neighboring_u: Tensor = single_get_ids_neg_sampling(
        target=target_u, edge_index=edge_index,
        graph_nodes=dataset_sizes[dataset_name.value][0], num_neg_samples=num_neg_samples
    )

    non_neighboring_v: Tensor = single_get_ids_neg_sampling(
        target=target_v, edge_index=edge_index,
        graph_nodes=dataset_sizes[dataset_name.value][0], num_neg_samples=num_neg_samples
    )

    fst_slice: int = num_neg_samples // 2
    sdn_slice: int = num_neg_samples - (num_neg_samples // 2)

    return non_neighboring_u[:fst_slice], non_neighboring_v[:sdn_slice]


def get_features_neg_sampling(
        idx_neg_sampled: Tuple[Tensor, Tensor], hop: int,
        data: Data, target_nodes: Tensor,
        log: bool = False,
        dataset_name: str = DatasetType.PUBMED.value,
        alter_labeling: bool = False,
        device: str = "cpu"
) -> Tuple[list[DatasetItem], list[Tensor]]:
    if hop == 1:
        max_num_nodes: int = dataset_split_max_size[dataset_name][0]
    elif hop == 2:
        max_num_nodes: int = dataset_split_max_size[dataset_name][1]
    else:
        max_num_nodes: int = dataset_split_max_size[dataset_name][2]

    negative_set: list[DatasetItem] = []
    idx_neg_sampled_u, idx_neg_sampled_v = idx_neg_sampled
    full_idx_neg: Tensor = torch.cat((idx_neg_sampled_u, idx_neg_sampled_v))
    subgraph_ids: list[Tensor] = []

    for count, neg_idx in enumerate(full_idx_neg):
        target_idx: int = 0 if count < len(idx_neg_sampled_u) else 1
        packet: SubGraphWrapper = subgraph_extraction(
            graph=data,
            target_nodes=torch.tensor([target_nodes[target_idx], neg_idx]),
            k_hop=hop, subgraph_size=max_num_nodes,
            graph_size=dataset_sizes[dataset_name][0],
            return_single_subgraphs=True, device=device,
        )

        pos_mat: Tensor = get_all_positional(
            packet=packet, log=log, device=device,
            num_nodes=dataset_sizes[dataset_name][0]
        )

        pad: Tensor = torch.full(
            size=(
                packet.subgraph_ids.shape[0],
                max_num_nodes - packet.subgraph_ids.shape[0]
            ),
            fill_value=0., device=device
        )

        u_mask = (packet.subgraph_ids == target_nodes[target_idx]).int()
        neg_v_mask = (packet.subgraph_ids == neg_idx).int()
        target_footprint: Tensor = u_mask | neg_v_mask

        altered_label: Tensor = torch.tensor([])

        if alter_labeling:
            altered_label = (
                    pos_mat * target_footprint.unsqueeze(dim=0)
            ).mean(dim=1)
            altered_label[torch.nonzero(target_footprint, as_tuple=False).reshape(-1)] = 1

        pos_mat = torch.cat([pos_mat, pad], dim=-1)
        negative_set.append(
            DatasetItem(
                bow_emb=torch.tensor([]),
                pos_emb=pos_mat,
                mp_emb=None,
                labels=torch.tensor(0),
                target_footprint=target_footprint,
                altered_labels=altered_label if alter_labeling else None,
                mask=None
            )
        )

        subgraph_ids.append(packet.subgraph_ids)

    return negative_set, subgraph_ids
