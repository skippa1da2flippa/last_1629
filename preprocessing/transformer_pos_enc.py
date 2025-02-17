import networkx as nx
import torch
from networkx import Graph
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from preprocessing.data.sub_graph_wrapper import SubGraphWrapper
from utils.neighbor_cntx_manger import NeighborContextManager


def get_all_positional(
        packet: SubGraphWrapper,
        num_nodes: int,
        log: bool = False,
        device: str = "cuda"
) -> Tensor:

    distance_mat: Tensor = torch.eye(
        packet.subgraph_ids.shape[0],
        dtype=torch.float, device=device
    )

    data: Data = Data(edge_index=packet.edge_index, num_nodes=num_nodes)
    graph: Graph = to_networkx(data, to_undirected=True)

    for idx in range(packet.subgraph_ids.shape[0] - 1):
        get_one_positional(
            graph=graph, source_id=idx,
            subgraph_ids=packet.subgraph_ids,
            distance_mat=distance_mat, log=log
        )

    return distance_mat


def get_one_positional(
        graph: Graph, source_id: int, subgraph_ids: Tensor,
        distance_mat: Tensor, log: bool = False
) -> None:

    source: Tensor = subgraph_ids[source_id]
    for target_id in range(source_id + 1, distance_mat.shape[0]):
        target: Tensor = subgraph_ids[target_id]
        with NeighborContextManager(source=source, target=target, log=log):
            target_source_dist: int = nx.shortest_path_length(
                G=graph, source=source.item(), target=target.item()
            )

            if target_source_dist is not None:
                val = 1 / (target_source_dist + 1)
            else:
                val = 0

            distance_mat[source_id, target_id] = val
            distance_mat[target_id, source_id] = val
