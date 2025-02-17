from typing import Tuple, Callable
import torch
from torch import Tensor
import torch.nn as nn
from custom_blocks.layers.masked_linear import MaskedLinear


def find_split(multi: int, target: int) -> int:
    while target % multi:
        multi += 1

    return multi


class SuperLinear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            alpha: float = 0.5, decomposed: bool = False,
            num_target: int = 2, device: str = "cuda"
    ) -> None:
        super().__init__()

        reduction_layer_out: int = find_split(int(out_features * alpha), out_features)

        self.device: str = device
        self.decomposed: bool = decomposed

        self.linear: nn.Linear = nn.Linear(
            in_features=in_features, out_features=reduction_layer_out,
            device=device
        )
        self.red_out: int = reduction_layer_out
        self.num_target: int = num_target
        self.alpha: float = reduction_layer_out / out_features

        amplify_layer_dim: int = out_features // reduction_layer_out
        V_std_dev: float = 1 / out_features
        self.V: nn.Parameter = nn.Parameter(data=torch.randn(amplify_layer_dim, device=device) * V_std_dev)

        b_V_size: tuple[int] = (out_features // 2,) if decomposed else (out_features,)
        self.bias_V: nn.Parameter = nn.Parameter(data=torch.randn(b_V_size, device=device) * V_std_dev)

        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, batch: Tensor) -> Tensor:
        reduction_out: Tensor = self.linear(batch)
        tensor_prod: Tensor = torch.tensordot(reduction_out, self.V, dims=0)

        if tensor_prod.shape[-1] == 1:
            tensor_prod = tensor_prod.squeeze(dim=-1)

        if self.decomposed:
            tensor_prod = tensor_prod.view(tensor_prod.shape[0], tensor_prod.shape[-1], -1)
        else:
            dims = (tensor_prod.shape[0], self.num_target, -1) if self.num_target > 1 else (tensor_prod.shape[0], -1)
            tensor_prod = tensor_prod.view(*dims)

        return tensor_prod + self.bias_V
