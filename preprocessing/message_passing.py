from typing import Callable
import torch
from torch import Tensor


def message_passing(adj_mat: Tensor, layers: int, message_decay: bool = False, device: str = "cuda") -> Tensor:
    old_adj: Tensor = torch.zeros(adj_mat.shape, dtype=torch.int, device=device)
    message: Tensor = torch.zeros(adj_mat.shape[0], dtype=torch.int, device=device)
    new_adj: Tensor = torch.zeros(adj_mat.shape, dtype=torch.int, device=device)

    f_w: Callable[[int], float] = lambda layer: 1 / layer if message_decay else 1.

    my_adj: Tensor = adj_mat.clone()

    for layer in range(layers):
        new_adj = my_adj.clone()
        for row in range(my_adj.shape[0]):
            neighbors_map: Tensor = my_adj[row, :] >= 1
            message = torch.sum(
                my_adj[neighbors_map, :] - old_adj[neighbors_map, :], dim=0
            )
            new_adj[row, :] += f_w(layer + 1) * message

        old_adj = my_adj
        my_adj = new_adj

    return my_adj
