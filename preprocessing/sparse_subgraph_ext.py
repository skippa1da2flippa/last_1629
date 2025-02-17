from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from preprocessing.data.sub_graph_wrapper import SubGraphWrapper


def single_subgraph_extraction(
        edge_index: Tensor, target_node: Tensor,
        k_hop: int, graph_nodes: int,
        device: str = "cuda"
) -> Tuple[Tensor, Tensor]:
    node_mask_indices = set()  # Sparse set representation of node_mask
    target_node = target_node.reshape(-1)
    nodes_subset: list[Tensor] = [target_node.to(device)]

    dis_target: Tensor = torch.full(size=(graph_nodes, 1), fill_value=torch.inf, device=device)
    dis_target[target_node, :] = 0

    for hop in range(1, k_hop + 1):
        node_mask_indices.update(nodes_subset[-1].tolist())  # Mark nodes as visited

        # Get neighbors of active nodes
        active_nodes = torch.tensor(list(node_mask_indices), device=device)
        mask = torch.isin(edge_index[1], active_nodes)
        new_nodes = edge_index[0, mask]

        # Compute hop distance updates
        hop_matrix: Tensor = torch.full(size=(new_nodes.shape[0], 1), fill_value=hop, device=device)
        min_vector: Tensor = torch.cat([dis_target[new_nodes], hop_matrix], dim=1)
        dis_target[new_nodes] = torch.min(min_vector, dim=1)[0]

        # Update visited nodes for next iteration
        nodes_subset.append(new_nodes)

    subset = torch.cat(nodes_subset).unique()
    final_node_mask = torch.zeros(graph_nodes, dtype=torch.bool, device=device)
    final_node_mask[subset] = True  # Convert sparse set back to tensor

    return final_node_mask, dis_target




