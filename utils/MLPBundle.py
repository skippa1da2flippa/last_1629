from typing import Type, Callable, Any
from torch import nn, Tensor
from custom_blocks.forbidden_block import ForbiddenBlock
from custom_blocks.layers.aggregator import AggregatorLayer
from custom_blocks.layers.decomposer import DecomposerLayer
from custom_blocks.superblock import SuperBlock
from utils.layer_type import LayerType


class MLPBundle:
    def __init__(
            self, l_type: LayerType, index: int, in_features: int = None,
            out_features: int = None, mid_out: int = None, alpha: float = None,
            f: Type[nn.Module] = None, dropout: float = .0, emb_size: int = None,
            aggregator: Callable[[Tensor], Tensor] = None, tensor_dot: bool = None,
            num_target: int = 2
    ) -> None:

        self._index: int = index

        if l_type == LayerType.DROPOUT:
            self._type: Type[nn.Dropout] = nn.Dropout
            self.name: str = f"dropout_{self._index}"
            self._bundle: dict[str, Any] = {"p": dropout}

        elif l_type == LayerType.LAYER_NORM:
            self._type: Type[nn.LayerNorm] = nn.LayerNorm
            self.name = f"layer_norm_{self._index}"
            self._bundle: dict[str, Any] = {"normalized_shape": emb_size}

        elif l_type == LayerType.LINEAR:
            self._type: Type[nn.Linear] = nn.Linear
            self.name = f"linear_{self._index}"
            self._bundle: dict[str, Any] = {
                "in_features": in_features,
                "out_features": out_features
            }

        elif l_type == LayerType.SUPER_BLOCK:
            self._type: Type[SuperBlock] = SuperBlock
            self.name: str = f"super_block_{self._index}"
            self._bundle: dict[str, Any] = {
                "in_features": in_features,
                "mid_out": mid_out,
                "out_features": out_features,
                "alpha": alpha,
                "f": f,
                "dropout": dropout,
                "num_target": num_target
            }
        elif l_type == LayerType.FORBIDDEN_BLOCK:
            self._type: Type[ForbiddenBlock] = ForbiddenBlock
            self.name: str = f"forbidden_block_{self._index}"
            self._bundle: dict[str, Any] = {
                "in_features": in_features,
                "out_features": out_features,
                "aggregator": aggregator,
                "mid_out": mid_out,
                "tensor_dot": tensor_dot,
                "num_target": num_target,
                "dropout": dropout,
                "f": f
            }
        elif l_type == LayerType.AGGREGATOR:
            self._type: Type[AggregatorLayer] = AggregatorLayer
            self.name: str = f"aggregator_ly_{self._index}"
            self._bundle: dict[str, Any] = {
                "in_features": in_features,
                "out_features": out_features,
                "aggregator": aggregator,
                "mid_out": mid_out
            }
        elif l_type == LayerType.DECOMPOSER:
            self._type: Type[DecomposerLayer] = DecomposerLayer
            self.name: str = f"decomposer_ly_{self._index}"
            self._bundle: dict[str, Any] = {
                "in_features": in_features,
                "out_features": out_features,
                "tensor_dot": tensor_dot,
                "num_target": num_target
            }

    def get_bundle(self) -> dict[str, Any]:
        return self._bundle

    def get_type(self) -> Type[nn.Module]:
        return self._type
