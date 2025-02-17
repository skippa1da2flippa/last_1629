from typing import Callable, Union, Any
from torch import nn, Tensor


class AggregatorLayer(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            aggregator: Callable[[Tensor], Tensor], mid_out: int = None,
            device: str = "cuda"
    ) -> None:
        super().__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features

        if aggregator is None:
            raise Exception(f"A None is found where substance was sought. Prithee, provide that which is not None!")

        self.aggregator: Callable[[Any], Tensor] = aggregator

        self.mid_out = mid_out if mid_out is not None else in_features

        self.linear: nn.Linear = nn.Linear(
            in_features=self.mid_out, out_features=out_features,
            device=device
        )

    def forward(self, batch: Tensor) -> Tensor:
        aggregated_batch: Tensor = self.aggregator(batch)
        return self.linear(aggregated_batch)
