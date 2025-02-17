import torch
from torch import Tensor
from torch_geometric.data import Data


class SubGraphWrapper(Data):
    def __init__(
            self, edge_index: Tensor, subgraph_mask: Tensor, k_hop: int,
            distances: list[Tensor], single_masks: list[Tensor] = None,
            x: Tensor = None
    ) -> None:
        super().__init__(edge_index=edge_index, x=x)

        self.subgraph_mask: Tensor = subgraph_mask

        self.subgraph_ids: Tensor = torch.nonzero(
            self.subgraph_mask, as_tuple=False
        ).reshape(-1)

        self.k_hop: int = k_hop
        self.u_distance, self._v_distance = distances

        if single_masks is not None:
            self.u_mask, self.v_mask = single_masks
            self.u_subgraph: Tensor = torch.nonzero(self.u_mask, as_tuple=False).reshape(-1)
            self.v_subgraph: Tensor = torch.nonzero(self.v_mask, as_tuple=False).reshape(-1)

    def __iter__(self):
        yield self.subgraph_ids
        yield self.edge_index
        yield self.u_distance, self._v_distance
        if hasattr(self, "u_mask"):
            yield self.u_subgraph, self.u_subgraph
