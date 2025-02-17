from typing import Tuple
from torch import nn, Tensor
from custom_blocks.layers.superlinear import SuperLinear


class DecomposerLayer(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            tensor_dot: bool = True,
            num_target: int = 1, device: str = "cuda"
    ) -> None:

        super().__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.tensor_dot: bool = tensor_dot
        self.num_target: int = num_target

        if self.tensor_dot:
            self.decompose: SuperLinear = SuperLinear(
                in_features=in_features,
                out_features=num_target * out_features,
                decomposed=True, num_target=1, device=device
            )

        else:
            self.decompose: nn.Linear = nn.Linear(
                in_features=in_features,
                out_features=num_target * out_features,
                device=device
            )

    def forward(self, batch: Tensor) -> Tensor:
        if self.num_target >= 2:
            dims: Tuple = batch.shape[0], self.num_target, -1
        else:
            dims: Tuple = (batch.shape[0], -1)

        x_batch = self.decompose(batch)
        x_batch = x_batch.reshape(*dims)

        return x_batch
