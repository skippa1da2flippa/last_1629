from typing import Tuple
from torch import Tensor
from preprocessing.sub_graph_ext import single_subgraph_extraction


def get_common_neighbor_and_degree(
    edge_index: Tensor, target_nodes: Tensor,
    k_hop: int, graph_nodes: int,
    device: str = "cuda"
) -> Tuple[int, int, int]:
    neighbor_map_u, _ = single_subgraph_extraction(
        edge_index=edge_index, target_node=target_nodes[0],
        k_hop=k_hop, graph_nodes=graph_nodes,
        device=device
    )

    neighbor_map_v, _ = single_subgraph_extraction(
        edge_index=edge_index, target_node=target_nodes[1],
        k_hop=k_hop, graph_nodes=graph_nodes,
        device=device
    )

    cn: int = (neighbor_map_v & neighbor_map_u).sum().item()
    degree_u: int = neighbor_map_u.sum().item()
    degree_v: int = neighbor_map_v.sum().item()

    return cn, degree_u, degree_v
