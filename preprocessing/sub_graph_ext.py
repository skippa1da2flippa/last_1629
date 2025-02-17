from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from preprocessing.data.sub_graph_wrapper import SubGraphWrapper


def sparse_single_subgraph_extraction(
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


def single_subgraph_extraction(
        edge_index: Tensor, target_node: Tensor,
        k_hop: int, graph_nodes: int,
        device: str = "cuda"
) -> Tuple[Tensor, Tensor]:
    node_mask: Tensor = torch.zeros(size=(graph_nodes,), dtype=torch.bool, device=device)
    target_node = target_node.reshape(-1)
    nodes_subset: list[Tensor] = [target_node.to(device)]

    dis_target: Tensor = torch.full(size=(graph_nodes, 1), fill_value=torch.inf, device=device)
    dis_target[target_node, :] = 0

    for hop in range(1, k_hop + 1):
        node_mask[nodes_subset[-1]] = True

        neighbors_mask: Tensor = torch.index_select(
            input=node_mask, dim=0, index=edge_index[1],
        )

        node_mask[edge_index[0, neighbors_mask]] = True
        hop_matrix: Tensor = torch.full(size=(node_mask.sum(), 1), fill_value=hop, device=device)
        min_vector: Tensor = torch.cat(
            tensors=[dis_target[node_mask], hop_matrix],
            dim=1
        )
        dis_target[node_mask.unsqueeze(dim=1)] = torch.min(min_vector, dim=1)[0]

        nodes_subset.append(edge_index[0, neighbors_mask])
        node_mask.fill_(False)

    subset = torch.cat(nodes_subset).unique()
    node_mask[subset] = True

    return node_mask, dis_target


def subgraph_extraction(
        graph: Data, target_nodes: Tensor, k_hop: int, subgraph_size: int,
        graph_size: int, return_single_subgraphs: bool = False,
        remove_edge: bool = False, device: str = "cuda"
) -> SubGraphWrapper:
    if remove_edge:
        lr_mask: Tensor = (graph.edge_index[0, :] == target_nodes[0]) & (graph.edge_index[1, :] == target_nodes[1])
        rl_mask: Tensor = (graph.edge_index[0, :] == target_nodes[1]) & (graph.edge_index[1, :] == target_nodes[0])
        mask = lr_mask | rl_mask
        mask = ~mask
        new_edge_index: Tensor = graph.edge_index[:, mask]
    else:
        new_edge_index: Tensor = graph.edge_index

    u_node_mask, u_distances = single_subgraph_extraction(
        edge_index=new_edge_index, target_node=target_nodes[0],
        k_hop=k_hop, graph_nodes=graph_size, device=device
    )

    v_node_mask, v_distances = single_subgraph_extraction(
        edge_index=new_edge_index, target_node=target_nodes[1],
        k_hop=k_hop, graph_nodes=graph_size, device=device
    )

    uv_node_mask: Tensor = u_node_mask | v_node_mask

    if torch.sum(uv_node_mask) > subgraph_size:
        uv_node_mask[target_nodes] = False

        rand_ids: Tensor = torch.randperm(torch.sum(uv_node_mask).item(), device=device)
        rand_ids = rand_ids[:subgraph_size - 2]
        rand_ids = torch.cat((rand_ids, target_nodes))

        uv_node_mask.fill_(False)
        uv_node_mask[rand_ids] = True

    uv_edge_mask: Tensor = uv_node_mask[new_edge_index[0]] & uv_node_mask[new_edge_index[1]]

    if return_single_subgraphs:
        packet: SubGraphWrapper = SubGraphWrapper(
            edge_index=new_edge_index[:, uv_edge_mask],
            subgraph_mask=uv_node_mask, k_hop=k_hop,
            distances=[u_distances, v_distances],
            single_masks=[u_node_mask, v_node_mask]
        )

    else:
        packet: SubGraphWrapper = SubGraphWrapper(
            edge_index=new_edge_index[:, uv_edge_mask],
            subgraph_mask=uv_node_mask, k_hop=k_hop,
            distances=[u_distances, v_distances]
        )

    return packet
