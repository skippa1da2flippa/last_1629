from enum import Enum
from typing import Callable, Union
import torch
from torch import Tensor


class AggregatorType(Enum):
    SUM = "sum"
    HADAMARD = "hadamard"
    CONCAT = 'concatenation'
    NONE = "empty"
    AVERAGE = "average"


AGGREGATOR_FUNCTIONS_ss: dict[str, Callable[[Tensor], Tensor]] = {
    AggregatorType.SUM.name: lambda x: torch.sum(x, dim=1),
    AggregatorType.HADAMARD.name: lambda x: torch.prod(x, dim=1),
    AggregatorType.NONE.name: lambda x: x,
    AggregatorType.AVERAGE.name: lambda x: torch.mean(x, dim=1),
}

AGGREGATOR_FUNCTIONS_ds: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    AggregatorType.SUM.name: lambda x, y: x + y,
    AggregatorType.HADAMARD.name: lambda x, y: x * y,
    AggregatorType.AVERAGE.name: lambda x, y: 0.5 * (x + y),
    AggregatorType.NONE.name: lambda x, y: x
}

MASKED_AGGREGATOR_FUNCTION_ss: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    AggregatorType.SUM.name: lambda x, mask: masked_sum(x, mask),
    AggregatorType.HADAMARD.name: lambda x, mask: masked_prod(x, mask),
    AggregatorType.NONE.name: lambda x, mask: x,
    AggregatorType.AVERAGE.name: lambda x, mask: masked_avg(x, mask),
}


def masked_prod(x: Tensor, mask: Tensor) -> Tensor:
    mask = mask.unsqueeze(dim=-1).int()
    masked_x: Tensor = x * mask
    return torch.prod(masked_x + (1 - mask), dim=1)


def masked_sum(x: Tensor, mask: Tensor) -> Tensor:
    mask = mask.unsqueeze(dim=-1).int()
    masked_x: Tensor = x * mask
    return torch.sum(masked_x, dim=1)


def masked_avg(x: Tensor, mask: Tensor) -> Tensor:
    summed_mask: Tensor = torch.sum(mask, dim=1)
    masked_x: Tensor = masked_sum(x, mask)
    return masked_x / summed_mask.unsqueeze(dim=-1)


def base_aggregator_wrapper(aggregator_name: str, masked: bool = False) -> Union[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor]
]:
    if masked:
        return MASKED_AGGREGATOR_FUNCTION_ss[AggregatorType[aggregator_name]]
    else:
        return AGGREGATOR_FUNCTIONS_ss[AggregatorType[aggregator_name]]

