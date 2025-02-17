from typing import Type, Callable, Tuple, Union
from torch import nn, Tensor
from custom_blocks.layers.superlinear import SuperLinear


class SuperBlock(nn.Module):
    def __init__(
            self, in_features: int, mid_out: int, out_features: int,
            alpha: float = 0.5, f: Type[nn.Module] = nn.Sigmoid, dropout: float = .0,
            num_target: int = 2, device: str = "cuda"
    ) -> None:

        super().__init__()

        if dropout > .0:
            self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        else:
            self.dropout: Callable[[Tensor], Tensor] = lambda x: x

        self.super_linear: SuperLinear = SuperLinear(
            in_features=in_features, out_features=mid_out, alpha=alpha,
            device=device, num_target=num_target,
        )

        self.f: nn.Module = f()

        self.masked_linear: nn.Linear = nn.Linear(
            in_features=mid_out, out_features=out_features,
            device=device
        )

    def forward(self, batch: Tensor) -> Tensor:
        dropped_out: Tensor = self.dropout(batch)

        x_batch = self.super_linear(dropped_out)
        x_batch = self.f(x_batch)

        return self.masked_linear(x_batch)
