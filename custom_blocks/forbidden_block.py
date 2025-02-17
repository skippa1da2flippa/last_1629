from typing import Callable, Type
from torch import nn, Tensor
from custom_blocks.layers.aggregator import AggregatorLayer
from custom_blocks.layers.decomposer import DecomposerLayer


class ForbiddenBlock(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            aggregator: Callable[[Tensor], Tensor], mid_out: int = None,
            tensor_dot: bool = True, num_target: int = 2, dropout: float = .0,
            f: Type[nn.Module] = nn.Sigmoid, device: str = "cuda"
    ) -> None:

        super().__init__()

        if dropout > .0:
            self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        else:
            self.dropout: Callable[[Tensor], Tensor] = lambda x: x

        self.in_features: int = in_features
        self.out_features: int = out_features

        self.aggregator_layer: AggregatorLayer = AggregatorLayer(
            in_features=in_features, out_features=out_features,
            aggregator=aggregator, mid_out=mid_out, device=device,
        )

        self.f: nn.Module = f()

        self.decompose_layer: DecomposerLayer = DecomposerLayer(
            in_features=out_features, out_features=out_features,
            tensor_dot=tensor_dot, num_target=num_target, device=device,
        )

    def forward(self, batch: Tensor) -> Tensor:
        x_batch = self.dropout(batch)

        x_batch = self.aggregator_layer(x_batch)
        x_batch = self.f(x_batch)

        return self.decompose_layer(x_batch)
