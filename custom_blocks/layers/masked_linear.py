from typing import Tuple, Callable
import torch
from torch import nn, Tensor


class MaskedLinear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            decomposed: bool = False, num_target: int = 2,
            ignore_mask: bool = False,
            device: str = "cuda"
    ) -> None:
        super().__init__()

        self.device: str = device
        self.num_target: int = num_target
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.decomposed: bool = decomposed
        self.ignore_mask: bool = ignore_mask

        std_dev: float = 1 / out_features
        self.W: nn.Parameter = nn.Parameter(
            data=torch.randn(out_features, in_features, device=device) * std_dev
        )
        self.bias: nn.Parameter = nn.Parameter(
            data=torch.randn(out_features, device=device) * std_dev
        )

        def get_mask(batch_size: int, masks: Tensor = None) -> Tensor:
            if masks is not None and self.out_features > 2:
                if self.decomposed:
                    return torch.cat((masks, masks), dim=1)
                else:
                    if masks.shape[-1] == self.bias.shape[-1]:
                        return masks
                    else:
                        return torch.ones(
                            (batch_size, self.bias.shape[-1]),
                            device=self.device
                        )

            elif self.out_features <= 2:
                return torch.ones((1,), device=self.device)
            else:
                dim = (batch_size, 2 * self.bias.shape[0]) if self.decomposed else (batch_size, self.bias.shape[0])
                return torch.ones(dim, device=self.device)

        self.get: Callable[[int, Tensor], Tensor] = get_mask

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x_batch, masks = batch

        if self.ignore_mask:
            masked_b = self.bias
        else:
            padded_masks = self.get(x_batch.shape[0], masks)
            masked_b: Tensor = self.bias * padded_masks

        if self.num_target > 1:
            masked_b = masked_b.unsqueeze(dim=1)

        return x_batch @ self.W.T + masked_b, masks
